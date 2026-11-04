"""Copy a release from one registry to another, with the payload sources that it needs.

A payload source is a release whose payload files, such as Signal values, another release reads. A
published version cannot change, so a copy never replaces a release that the destination holds with
other files.
"""

from pathlib import Path
import tempfile

from timenet.errors import (
    TimeFFormatError,
    TimeNetDatasetNotFoundError,
    TimeNetMissingParentError,
    TimeNetVersionConflictError,
)
from timenet.manifest import Manifest
from timenet.progress import ProgressCallback, StepFinished, StepStarted
from timenet.registry._paths import release_dir
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_pins, prepare_local_copy, validate_manifest_reference
from timenet.registry.local import LocalRegistry
from timenet.registry.shared import complete_copy
from timenet.registry.writable import WritableRegistry
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef


def copy_release(
    reference: DatasetRef,
    source: BaseRegistry,
    dest: WritableRegistry,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Copy a release and the payload sources that it needs to another registry.

    Payload sources go first, and a complete copy in ``dest`` is reused. A damaged local copy is
    fetched again, and if a check fails, nothing is copied.

    Args:
        reference: The exact release to copy.
        source: The registry to copy from.
        dest: The registry to copy to.
        progress: Receives :class:`~timenet.progress.StepStarted` and
            :class:`~timenet.progress.StepFinished` for each release that is copied or repaired.

    Raises:
        TimeNetVersionConflictError: If ``dest`` holds the release, or a payload source of it, with
            other files.
        TimeNetAccessError: If a release of the copy is not open, and ``dest`` requires open access
            or ``source`` requires it for a release that the copy fetches.
        TimeNetDatasetNotFoundError: If ``source`` does not hold the release.
        TimeNetMissingParentError: If a payload source is neither in ``source`` nor complete in
            ``dest``.
        TimeFFormatError: If a manifest names another release, a payload source differs from the
            files that a release pins, or the payload sources form a cycle.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
    manifests = _manifests(reference, source, dest)
    # Check every release before anything is copied, so that a refusal leaves dest unchanged. A payload
    # source that only dest holds is complete there, so it is never pending.
    pending = [ref for ref, manifest in manifests.items() if _needs_copy(reference, ref, manifest, source, dest)]
    _check_access(manifests, set(pending), source, dest)
    if not pending:
        return
    # Only a copy from a non-local source to a non-local dest downloads into this directory, each
    # pending release on its own.
    with tempfile.TemporaryDirectory(prefix="timenet-copy-", ignore_cleanup_errors=True) as staging:
        for ref in pending:
            if progress is not None:
                progress(StepStarted(ref, "copy"))
            _copy_one(ref, manifests[ref], source, dest, Path(staging))
            if progress is not None:
                progress(StepFinished(ref, "copy"))


def _copy_one(
    reference: DatasetRef, manifest: Manifest, source: BaseRegistry, dest: WritableRegistry, staging: Path
) -> None:
    """Copy one release to ``dest``, or repair the local copy that ``dest`` holds.

    Args:
        reference: The release.
        manifest: The manifest of the release in ``source``.
        source: The registry to copy from.
        dest: The registry to copy to.
        staging: A temporary directory for a download from a non-local ``source``.
    """
    # A local registry is a folder that the copy reads or writes directly.
    if isinstance(dest, LocalRegistry):
        prepare_local_copy(dest.local_root, reference, fetch_from=source, manifest=manifest)
        return
    if isinstance(source, LocalRegistry):
        directory = release_dir(source.root, reference)
    else:
        # put_release() needs only the release directory, and the walk checked the pins. So the payload
        # sources, which dest may hold already, are not downloaded. The download checks each file by size
        # and checksum.
        directory = release_dir(staging, reference)
        source.download_release(reference.dataset_id, str(reference.version), directory, manifest=manifest)
    dest.put_release(directory)


def _check_access(
    manifests: dict[DatasetRef, Manifest], fetched: set[DatasetRef], source: BaseRegistry, dest: WritableRegistry
) -> None:
    """Refuse a release that is not open for a registry that requires open access.

    ``dest`` must hold each release of the copy, and ``source`` must serve each release that the copy
    fetches.

    Args:
        manifests: The manifest of each release of the copy, by release.
        fetched: The releases that the copy fetches from ``source``.
        source: The registry to copy from.
        dest: The registry to copy to.

    Raises:
        TimeNetAccessError: If the card or the lineage of such a release declares an access that is not
            ``open``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by require_open_access
    for reference, manifest in manifests.items():
        dest_needs_open = dest.requires_open_access
        source_needs_open = reference in fetched and source.requires_open_access
        if dest_needs_open or source_needs_open:
            require_open_access(manifest.metadata, manifest.lineage)


def _manifests(reference: DatasetRef, source: BaseRegistry, dest: WritableRegistry) -> dict[DatasetRef, Manifest]:
    """Return the manifests of a release and of every payload source that it needs.

    A payload source that ``source`` does not hold comes from a complete copy in ``dest``. Each
    payload source must match the pins of each release that reads it.

    Args:
        reference: The release.
        source: The registry to copy from.
        dest: The registry to copy to.

    Returns:
        Each manifest by its release, once for each release. Each payload source comes before the
        releases that read it, and the requested release comes last.

    Raises:
        TimeNetMissingParentError: If a payload source is neither in ``source`` nor complete in ``dest``.
        TimeFFormatError: If the payload sources form a cycle.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _payload_source_manifest
    manifests: dict[DatasetRef, Manifest] = {}
    # The releases whose payload sources are still being added: a release that reads one of them is a cycle.
    open_releases: set[DatasetRef] = set()

    def add(release: DatasetRef, manifest: Manifest) -> None:
        open_releases.add(release)
        for pinned in manifest.payload_sources.values():
            if pinned.release in open_releases:
                raise TimeFFormatError(f"the payload sources of {reference} form a cycle at {pinned.release}")
            known = manifests.get(pinned.release)
            held = known if known is not None else _payload_source_manifest(source, dest, pinned.release, release)
            check_pins(release, pinned, held)
            if known is None:
                add(pinned.release, held)
        open_releases.remove(release)
        manifests[release] = manifest

    manifest = source.get_manifest(reference.dataset_id, str(reference.version))
    validate_manifest_reference(manifest, reference)
    add(reference, manifest)
    return manifests


def _payload_source_manifest(
    source: BaseRegistry, dest: WritableRegistry, reference: DatasetRef, needed_by: DatasetRef
) -> Manifest:
    """Return the manifest of a payload source in ``source``, or of its complete copy in ``dest``.

    Args:
        source: The registry to copy from.
        dest: The registry to copy to.
        reference: The payload source.
        needed_by: The release that reads the payload source.

    Returns:
        The manifest of the payload source.

    Raises:
        TimeNetMissingParentError: If the payload source is neither in ``source`` nor complete in
            ``dest``.
    """
    try:
        manifest = source.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError as error:
        try:
            held = dest.get_manifest(reference.dataset_id, str(reference.version))
        except TimeNetDatasetNotFoundError:
            held = None
        if held is None or not _complete(reference, held, dest):
            raise TimeNetMissingParentError(reference, needed_by, str(source), payload_source=True) from error
        manifest = held
    validate_manifest_reference(manifest, reference)
    return manifest


def _needs_copy(
    requested: DatasetRef,
    reference: DatasetRef,
    manifest: Manifest,
    source: BaseRegistry,
    dest: WritableRegistry,
) -> bool:
    """Return whether ``dest`` lacks a release, or holds a local copy of it with a missing or resized file.

    Args:
        requested: The release that the copy is for, for the error.
        reference: The release to check: ``requested`` or a payload source that it needs.
        manifest: The manifest of the release in ``source``, or in ``dest`` if only ``dest`` holds it.
        source: The registry to copy from, for the error.
        dest: The registry to copy to.

    Returns:
        ``True`` if the release must be copied or repaired.

    Raises:
        TimeNetVersionConflictError: If ``dest`` holds the version with other files.
    """
    try:
        held = dest.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError:
        return True
    if _file_records(held) != _file_records(manifest):
        needed_by = None if reference == requested else requested
        raise TimeNetVersionConflictError(reference, str(dest), str(source), needed_by)
    return not _complete(reference, manifest, dest)


def _complete(reference: DatasetRef, manifest: Manifest, dest: WritableRegistry) -> bool:
    """Return whether ``dest`` holds a release complete, if it holds the release.

    A non-local registry publishes a version only as a whole, so it holds each release complete. A
    local copy is complete if each file of the manifest is present with its recorded size.

    Args:
        reference: The release.
        manifest: The manifest of the release, with the files that ``dest`` holds.
        dest: The registry to copy to.

    Returns:
        ``False`` only for a local copy with a missing or resized file.
    """
    return not isinstance(dest, LocalRegistry) or complete_copy(release_dir(dest.root, reference), manifest)


def _file_records(manifest: Manifest) -> frozenset[tuple[str, int, str]]:
    """Return the path, size and checksum of each file that a manifest records.

    Args:
        manifest: The manifest.

    Returns:
        One ``(path, size, checksum)`` entry for each file.
    """
    return frozenset((part.path, part.size, part.checksum) for part in manifest.files.all_files())
