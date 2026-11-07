"""Plan and run release builds with dependencies first."""

from collections.abc import Sequence
from typing import assert_never

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import BuildStep, CopyStep, Plan, ReadStep, plan_steps
from timenet.progress import StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.registry.origin import record_origin
from timenet.sources import ReleaseSource
from timenet.types.references import is_latest, split_ref


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
    _, steps = plan_steps(
        # A connector builds only the version of its card, so "latest" asks for that version.
        declared(dataset_id, None if is_latest(version) else version),
        store,
        parents_from=parents_from,
        force_build=force,
        open_access_only=open_access_only,
    )
    return Plan(tuple(steps))


def run_plan(plan: Plan, store: LocalRegistry, *, options: BuildOptions | None = None) -> None:
    """Run a plan's copy and build steps in order.

    Each copied or built release gets a record of its source.

    Args:
        plan: The plan to run.
        store: The local registry to copy and build into.
        options: Build settings, such as the progress callback.

    Raises:
        TimeNetVersionConflictError: If the local registry holds the same release with different files.
        TimeFFormatError: If a build changes files that other releases pin.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the copy and the backends
    options = options or BuildOptions()
    for step in plan.steps:
        match step:
            case ReadStep():
                # A read opens the release where it is, so it has nothing to run.
                pass
            case CopyStep():
                copies = copy_release(
                    step.release,
                    step.registry,
                    store,
                    with_payload_sources=step.with_payload_sources,
                    force=step.force,
                    progress=options.progress,
                )
                for release in copies:
                    record_origin(store, release, "copy", step.registry.location)
            case BuildStep():
                _build(step, store, options=options)
            case _:
                assert_never(step)


def _build(step: BuildStep, store: LocalRegistry, *, options: BuildOptions) -> None:
    """Build the requested release with the builder of its step and record the builder's package.

    Args:
        step: The build step, with ``force`` set to rebuild an existing release.
        store: The local registry to build into.
        options: The build settings.

    Raises:
        TimeNetDatasetNotFoundError: If the build omits the release or a payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by prepare_local_copy
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    step.builder.backend.build(step.release, store, force=step.force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the local registry must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    record_origin(store, step.release, "build", step.builder.package)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
