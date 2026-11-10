"""Plan and run release builds with dependencies first."""

from collections.abc import Sequence
from typing import assert_never

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import BuildStep, CopyStep, Plan, ReadStep, plan_steps
from timenet.progress import StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
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
    """Run the copy and build steps of a plan in order.

    Skip read steps.

    Args:
        plan: The steps to run.
        store: The local registry to copy and build into.
        options: Build settings, including the progress callback.

    Raises:
        TimeNetRegistryError: If the plan copies or builds a release and ``store`` is read-only.
        TimeNetVersionConflictError: If a copied release conflicts with the files in ``store``.
        TimeFFormatError: If a build changes files pinned by other releases.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the copy and the backends
    if any(not isinstance(step, ReadStep) for step in plan.steps):
        store.check_writable("the copies and builds of a plan")
    options = options or BuildOptions()
    for step in plan.steps:
        match step:
            case ReadStep():
                # A read opens the release where it is, so it has nothing to run.
                pass
            case CopyStep():
                copy_release(
                    step.release,
                    step.registry,
                    store,
                    with_payload_sources=step.with_payload_sources,
                    force=step.force,
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
