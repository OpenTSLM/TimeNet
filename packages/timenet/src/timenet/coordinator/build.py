"""Plan and run release builds with dependencies first."""

from collections.abc import Mapping, Sequence
from typing import assert_never

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import BuildStep, CopyStep, Plan, plan_steps
from timenet.manifest import PayloadSource
from timenet.progress import StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.sources import ReleaseSource
from timenet.types.references import DatasetRef, is_latest, split_ref


def plan_release_build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    store: LocalRegistry,
    parents_from: Sequence[ReleaseSource] = (),
    force: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
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
        pinned_by: File pins grouped by dependent release that the build must preserve.
        open_access_only: Require ``access: open`` for every needed release.

    Returns:
        A plan with steps in execution order.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's card.
        TimeNetMissingParentError: If no source provides a parent or a payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared and the plan
    dataset_id, version = split_ref(dataset_id)
    _, steps = plan_steps(
        # A connector builds only the version of its card, so "latest" asks for that version.
        declared(dataset_id, None if is_latest(version) else version),
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
        plan: The steps to run in dependency order.
        store: The local registry to copy and build into.
        options: Build settings, including the progress callback for copies and builds.

    Raises:
        TimeNetVersionConflictError: If ``store`` holds different files for a copied release.
        TimeFFormatError: If a build fails to reproduce pinned files.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the copy and the backends
    options = options or BuildOptions()
    for step in plan.steps:
        match step:
            case CopyStep():
                copy_release(
                    step.release,
                    step.registry,
                    store,
                    with_payload_sources=step.with_payload_sources,
                    progress=options.progress,
                )
            case BuildStep():
                _build(step, store, options=options)
            case _:
                assert_never(step)


def _build(step: BuildStep, store: LocalRegistry, *, options: BuildOptions) -> None:
    """Build an exact release with the builder of its step and check its required files.

    Args:
        step: The build step, with ``force`` set to rebuild an existing release.
        store: The local registry to build into.
        options: The build settings.

    Raises:
        TimeNetDatasetNotFoundError: If the result lacks the release or a payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by prepare_local_copy
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    step.builder.backend.build(step.release, store, force=step.force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the local registry must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
