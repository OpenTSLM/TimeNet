"""Prepare parent releases and build a dataset into a local registry."""

from collections.abc import Mapping

from timenet.builders import BuildOptions, declared
from timenet.coordinator.plan import Plan, Step, plan_parents
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest, PayloadSource
from timenet.progress import ProgressCallback, StepFinished, StepStarted
from timenet.registry import copy_release
from timenet.registry._paths import release_dir
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.registry.shared import complete_copy
from timenet.types.references import DatasetRef, split_ref


def build_release(  # ruff: ignore[too-many-arguments]
    dataset: str,
    *,
    store: LocalRegistry,
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    open_access_only: bool = False,
    options: BuildOptions | None = None,
) -> Plan:
    """Prepare a release's dependencies and build it into a local registry.

    Copy missing dependencies from ``fetch_from``, or build them if ``build_parents`` is set.
    Reuse an existing release unless ``force`` is set.

    Args:
        dataset: A dataset ID such as ``org/name``, or an exact release such as ``org/name@1.2.0``.
        store: The local registry to build into.
        fetch_from: The registry to copy missing parents from, or ``None`` to copy nothing.
        build_parents: Build missing parents with installed connectors.
        force: Rebuild the requested release and prepare its parents even if the release exists.
        pinned_by: Releases that already read the requested release's files, with the pins the new build must match.
        open_access_only: Require ``access: open`` for all releases and their lineage before running the plan.
        options: Build settings, or ``None`` for defaults.

    Returns:
        The completed plan, with steps in execution order.

    Raises:
        TimeNetDatasetNotFoundError: If no installed backend builds the dataset or a missing parent.
        TimeNetVersionUnavailableError: If the connector declares a different version.
        TimeNetVersionConflictError: If ``store`` holds a different copy of a release being copied.
        TimeFValidationError: If the dataset reference, backend, or dependency plan is invalid.
        TimeNetMissingParentError: If a parent is missing and ``build_parents`` is not set.
        TimeNetAccessError: If a release or its lineage lacks required open access.
        TimeNetInvalidCardError: If an installed connector's card cannot be read.
        TimeNetFileError: If ``store`` holds a damaged parent and no replacement is available.
        TimeFFormatError: If a build omits or changes a pinned file.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the plan, the copy and the backends
    options = options or BuildOptions()
    dataset_id, version = split_ref(dataset)
    found = declared(dataset_id, version)
    card, requested = found.card, found.release
    steps = plan_parents(
        dataset_id,
        store,
        fetch_from=fetch_from,
        build_parents=build_parents,
        force_build=force,
        open_access_only=open_access_only,
    )
    if force or not store.exists(requested.dataset_id, str(requested.version)):
        planned = {step.release for step in steps}
        depends_on = tuple(parent for parent in card.parents.values() if parent in planned)
        steps.append(Step(requested, "build", depends_on=depends_on, pinned_by=pinned_by or {}))
    if fetch_from is None and any(step.action == "copy" for step in steps):
        raise TimeFValidationError(f"the plan of {requested} copies a release, but has no registry to copy from")
    for step in steps:
        if step.action == "build":
            _build(step, store, force=force and step.release == requested, options=options)
        elif fetch_from is not None:
            _copy(step.release, fetch_from, store, progress=options.progress)
    return Plan(tuple(steps))


def _copy(
    release: DatasetRef, fetch_from: BaseRegistry, store: LocalRegistry, *, progress: ProgressCallback | None
) -> None:
    """Copy a release and its required payload sources into the store.

    If the release is intact, copy only its missing or damaged payload sources.

    Args:
        release: The release to copy.
        fetch_from: The registry to copy from.
        store: The destination registry.
        progress: The callback for copy progress events, or ``None``.
    """
    held = _intact_manifest(store, release)
    if held is None:
        copy_release(release, fetch_from, store, progress=progress)
        return
    for pinned in held.payload_sources.values():
        if _intact_manifest(store, pinned.release) is None:
            copy_release(pinned.release, fetch_from, store, progress=progress)
    # The copied payload sources must hold the files that the release pins.
    prepare_local_copy(store.root, release)


def _intact_manifest(store: LocalRegistry, reference: DatasetRef) -> Manifest | None:
    """Read a release's manifest and check its file sizes.

    Args:
        store: The local registry.
        reference: The release to check.

    Returns:
        The manifest, or ``None`` if the release or a file is missing or a file has the wrong size.
    """
    try:
        manifest = store.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError:
        return None
    return manifest if complete_copy(release_dir(store.root, reference), manifest) else None


def _build(step: Step, store: LocalRegistry, *, force: bool, options: BuildOptions) -> None:
    """Build a planned release and check its required files.

    Args:
        step: The build step.
        store: The local registry to build into.
        force: Rebuild even if ``store`` already holds the release.
        options: The build settings.

    Raises:
        TimeNetDatasetNotFoundError: If no backend builds this version or the result lacks a release or payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared and prepare_local_copy
    builder = declared(step.release.dataset_id, step.release.version).backend
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    builder.build_prepared(step.release, store, force=force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the store must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
