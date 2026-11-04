"""Build a release after each parent release that it needs.

Use ``plan_release_build`` to plan a build, and ``run_plan`` to run the plan.
"""

from collections.abc import Mapping, Sequence
from typing import cast

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import Plan, Step, plan_steps
from timenet.manifest import PayloadSource
from timenet.progress import StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.sources import ReleaseSource
from timenet.types.references import DatasetRef, split_ref


def plan_release_build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    store: LocalRegistry,
    parents_from: Sequence[ReleaseSource] = (),
    force: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    open_access_only: bool = False,
) -> Plan:
    """Plan a build of a release into the local store, and run nothing.

    The requested build comes last, unless the store holds an intact copy and ``force`` is false.
    Planning leaves the registry unchanged.

    Args:
        dataset_id: A dataset ID such as ``org/name``, or an exact release such as ``org/name@1.2.0``.
        store: The local registry to write to.
        parents_from: Sources to copy or build missing dependencies from, in preference order.
        force: Plan a forced rebuild and its parents even if the requested release already exists.
        pinned_by: File pins grouped by dependent release that the build must preserve.
        open_access_only: Require ``access: open`` for every needed release.

    Returns:
        The plan: its steps in the order in which they run.

    Raises:
        TimeNetDatasetNotFoundError: If no installed backend builds the dataset.
        TimeNetVersionUnavailableError: If ``dataset`` names another version than the card of the
            connector declares.
        TimeNetMissingParentError: If no source provides a parent or a payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared and the plan
    dataset_id, version = split_ref(dataset_id)
    declared(dataset_id, version)
    _, steps = plan_steps(
        dataset_id,
        store,
        parents_from=parents_from,
        force_build=force,
        open_access_only=open_access_only,
        pinned_by=pinned_by,
    )
    return Plan(tuple(steps))


def run_plan(plan: Plan, store: LocalRegistry, *, options: BuildOptions | None = None) -> None:
    """Run a plan's copy and build steps in order.

    Args:
        plan: The plan, for example from :func:`plan_release_build`.
        store: The local registry to copy and build into.
        options: Build settings, including the progress callback for copies and builds.

    Raises:
        TimeNetVersionConflictError: If a copy step finds that ``store`` holds a release that it
            copies, or a payload source of it, with other files.
        TimeFFormatError: If a build does not reproduce the files that other releases pin.
        TimeNetBuildError: If a build runs but fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the copy and the backends
    options = options or BuildOptions()
    for step in plan.steps:
        if step.action == "copy":
            # The planner gives each copy step a registry as its entry.
            copy_release(step.release, cast("BaseRegistry", step.entry), store, progress=options.progress)
        else:
            _build(step, store, options=options)


def _build(step: Step, store: LocalRegistry, *, options: BuildOptions) -> None:
    """Build a step's exact release and check that its required files are available.

    Args:
        step: The build step. Its ``force`` builds the release again, even if ``store`` holds it.
        store: The local registry to build into.
        options: The build settings.

    Raises:
        TimeNetDatasetNotFoundError: If no backend builds this version or the result lacks a release or payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared and prepare_local_copy
    builder = declared(step.release.dataset_id, step.release.version).backend
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    builder.build(step.release, store, force=step.force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the store must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
