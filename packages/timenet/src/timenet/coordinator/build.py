"""Build a release after each parent release that it needs.

A copy step uses :func:`~timenet.registry.copy_release`. A build step uses ``build_prepared()`` of
the backend that builds exactly that release.
"""

from collections.abc import Mapping

from timenet.builders import BuildOptions, exact_builder, find_card
from timenet.coordinator.plan import Plan, Step, plan_parents
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
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
    """Build a release into the local store, after each release that its build needs.

    The function copies each missing parent, and each missing payload source of a parent, from
    ``fetch_from``, or builds it if ``build_parents`` is set. If ``store`` already holds the
    requested release, the function builds it again only if ``force`` is set.

    Args:
        dataset: The dataset ID, as ``org/name``, or one exact release, as ``org/name@1.2.0``.
        store: The local registry to build into.
        fetch_from: The registry to copy missing parents from, or ``None`` to copy nothing.
        build_parents: Build missing parents with the backends that build them.
        force: Build the requested release again, even if ``store`` holds it, and plan its parents.
        pinned_by: The releases that already read files of the requested release, with their pins.
            The build publishes the release only if it writes each pinned file again.
        open_access_only: Refuse each release of the plan that does not declare ``access: open``,
            the requested release included, before any step runs.
        options: The options of each build, such as the progress callback.

    Returns:
        The plan that ran, with its steps in the order in which they ran.

    Raises:
        TimeNetDatasetNotFoundError: If no installed backend builds the dataset or a missing parent.
        TimeNetVersionUnavailableError: If ``dataset`` names another version than the card declares.
        TimeNetVersionConflictError: If ``store`` holds a copied release with other files.
        TimeFValidationError: If ``dataset`` is not a valid reference, or the parents form a cycle.
        TimeNetMissingParentError: If a parent is missing and ``build_parents`` is not set.
        TimeNetAccessError: If a release of the plan is not open where open access is required.
        TimeNetInvalidCardError: If the card of an installed connector cannot be read.
        TimeNetFileError: If ``store`` holds a damaged parent that the plan cannot copy again.
        TimeFFormatError: If a build does not write again the files that other releases pin.
        TimeNetBuildError: If a build runs but fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the plan, the copy and the backends
    options = options or BuildOptions()
    dataset_id, version = split_ref(dataset)
    card = find_card(dataset_id)
    if card is None:
        raise TimeNetDatasetNotFoundError(f"no installed connector builds {dataset_id!r}")
    if version is not None:
        wanted = DatasetRef(dataset_id=dataset_id, version=version)
        if wanted.version != card.dataset_version:
            raise TimeNetVersionUnavailableError(wanted, card.dataset_version)
    requested = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
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
    """Run one copy step.

    If ``store`` holds an intact copy of the release, the function copies only each payload source
    that ``store`` lacks or holds damaged.

    Args:
        release: The release of the copy step.
        fetch_from: The registry to copy from.
        store: The local registry to copy to.
        progress: Receives the progress events of each release that is copied, or ``None``.
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
    """Return the manifest of a release that the store holds with every file at its recorded size.

    Args:
        store: The local registry.
        reference: The release.

    Returns:
        The manifest, or ``None`` if ``store`` lacks the release or a file of it, or holds a file of
        another size.
    """
    try:
        manifest = store.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError:
        return None
    return manifest if complete_copy(release_dir(store.root, reference), manifest) else None


def _build(step: Step, store: LocalRegistry, *, force: bool, options: BuildOptions) -> None:
    """Run one build step with the backend that builds exactly its release.

    Args:
        step: The build step.
        store: The local registry to build into.
        force: Build the release again, even if ``store`` holds it.
        options: The options of the build.

    Raises:
        TimeNetDatasetNotFoundError: If no installed backend builds the exact release, or if the
            backend does not put the release or a payload source of it into ``store``.
    """
    builder = exact_builder(step.release)
    if builder is None:
        raise TimeNetDatasetNotFoundError(f"no installed connector builds {step.release}")
    if options.progress is not None:
        options.progress(StepStarted(step.release, "build"))
    builder.build_prepared(step.release, store, force=force, pinned_by=step.pinned_by, options=options)
    # After a successful build, the store must hold the exact release with its payload sources.
    prepare_local_copy(store.root, step.release)
    if options.progress is not None:
        options.progress(StepFinished(step.release, "build"))
