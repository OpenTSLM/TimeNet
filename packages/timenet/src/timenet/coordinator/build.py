"""Plan and run release builds with dependencies first."""

from collections.abc import Sequence
from typing import cast

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import Plan, Step, plan_steps
from timenet.progress import StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.registry.origin import record_origin
from timenet.sources import ReleaseSource
from timenet.types.references import split_ref


def plan_release_build(
    dataset_id: str,
    *,
    store: LocalRegistry,
    parents_from: Sequence[ReleaseSource] = (),
    force: bool = False,
    open_access_only: bool = False,
) -> Plan:
    """Plan a release build in a local registry.

    The plan reuses an intact release unless ``force`` is true.
    Planning leaves the registry unchanged.

    Args:
        dataset_id: A dataset ID such as ``org/name``, or an exact release such as ``org/name@1.2.0``.
        store: The local registry to write to.
        parents_from: Sources to copy or build missing dependencies from, in preference order.
        force: Plan a forced rebuild and its parents even if the requested release already exists.
        open_access_only: Require ``access: open`` for every needed release.

    Returns:
        A plan with steps in execution order.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's card.
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
    )
    return Plan(tuple(steps))


def run_plan(plan: Plan, store: LocalRegistry, *, options: BuildOptions | None = None) -> None:
    """Run the copy and build steps of a plan in order.

    A read step runs nothing, because its release stays where it is. Each release that a copy or a
    build puts into ``store`` gets a provenance record (see :mod:`timenet.registry.origin`).

    Args:
        plan: The steps to run.
        store: The local registry to copy and build into.
        options: Build settings, including the progress callback.

    Raises:
        TimeNetVersionConflictError: If a copied release conflicts with the files in ``store``.
        TimeFFormatError: If a build changes files pinned by other releases.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the copy and the backends
    options = options or BuildOptions()
    for step in plan.steps:
        if step.action == "copy":
            # The planner gives each copy step a registry as its entry.
            registry = cast("BaseRegistry", step.entry)
            for release in copy_release(step.release, registry, store, force=step.force, progress=options.progress):
                record_origin(store, release, "copy", registry.location)
        elif step.action == "build":
            _build(step, store, options=options)


def _build(step: Step, store: LocalRegistry, *, options: BuildOptions) -> None:
    """Build an exact release and check its required files.

    Args:
        step: The build step, with ``force`` set to rebuild an existing release.
        store: The local registry to build into.
        options: The build settings.

    Raises:
        TimeNetDatasetNotFoundError: If no installed backend builds the exact release, or if the
            backend does not put the release or a payload source of it into ``store``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared and prepare_local_copy
    found = declared(step.release.dataset_id, step.release.version)
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    found.backend.build(step.release, store, force=step.force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the store must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    record_origin(store, step.release, "build", found.package)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
