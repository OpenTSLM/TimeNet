"""Copy a release from one registry to another, with each release whose payload files it reads.

A release can read payload files, such as Signal values, that another release stores. That other
release is a payload source. A copy brings each payload source along, so that the destination can
open the copied release. A published version cannot change. So a copy never replaces a release that
the destination already holds with other files. It raises
:class:`~timenet.errors.TimeNetVersionConflictError` instead, before it copies anything.
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
    """Copy a release and its payload sources to a registry, without changing a published version.

    The copy covers the release and every payload source that it needs: each payload source that its
    manifest lists, their own payload sources, and so on. Each payload source goes before the releases
    that read it, so that ``dest`` never holds a release without its payload sources. Each release is
    handled once, even if several releases read it. Two copies of a release are the same if their
    manifests record the same files: the same paths, sizes and checksums. Other manifest fields do not
    count.

    ``source`` must hold the release. A payload source can come from ``dest`` instead: if ``source``
    does not hold it, but ``dest`` holds a complete copy, this function uses that copy and does not
    copy it again. That copy must also match the files that each release that reads it pins. A local
    copy is complete if each file of its manifest is present with its recorded size.

    This function checks every release first, and then copies. For each release, it does one of these:

    - If ``dest`` does not hold the release, it copies the release. A local ``dest`` gets the release
      with :func:`~timenet.registry.dependencies.prepare_local_copy`. Another ``dest`` gets it with
      :meth:`~timenet.registry.WritableRegistry.put_release`, from the release directory of a local
      ``source``, or from a temporary download of only this release that this function removes
      afterwards.
    - If a local ``dest`` holds the same release, but a file is missing or has another size, this
      function fetches the release again from ``source``, its manifest included. Another ``dest``
      holds the release complete, because it publishes a version only as a whole.
    - If ``dest`` holds the version with other files, it raises an error before it copies anything.

    A non-local registry serves files only for releases whose card and lineage declare access
    ``open``. So if ``dest`` is non-local, each release of the copy must be open. If ``source`` is
    non-local, each release that this function fetches from it must be open. This function makes
    sure that each of these releases is open before it copies anything.

    Args:
        reference: The exact release to copy.
        source: The registry to copy from.
        dest: The registry to copy to.
        progress: Receives :class:`~timenet.progress.StepStarted` and
            :class:`~timenet.progress.StepFinished` for each release that is copied or repaired.

    Raises:
        TimeNetVersionConflictError: If ``dest`` holds the release, or a payload source that it needs, with
            other files. Nothing is copied then.
        TimeNetAccessError: If ``dest`` is non-local and a release of the copy is not open, or if
            ``source`` is non-local and a release that this function fetches from it is not open. A
            release is open if its card and each card in its lineage declare access ``open``. Nothing
            is copied then.
        TimeNetDatasetNotFoundError: If ``source`` does not hold the release.
        TimeNetMissingParentError: If a payload source that the release needs is neither in ``source``
            nor complete in ``dest``. Nothing is copied then.
        TimeFFormatError: If a manifest names another release, a payload source in ``source``, or the
            copy in ``dest`` that this function uses, differs from the files that a release that reads
            it pins, or the payload sources form a cycle. Nothing is copied then.
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
        source._download_release(reference.dataset_id, str(reference.version), directory, manifest=manifest)
    dest.put_release(directory)


def _check_access(
    manifests: dict[DatasetRef, Manifest], fetched: set[DatasetRef], source: BaseRegistry, dest: WritableRegistry
) -> None:
    """Refuse a release that is not open for a non-local registry that must serve it.

    ``dest`` serves each release of the copy. ``source`` serves each release that the copy fetches.

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
        if not isinstance(dest, LocalRegistry) or (reference in fetched and not isinstance(source, LocalRegistry)):
            require_open_access(manifest.metadata, manifest.lineage)


def _manifests(reference: DatasetRef, source: BaseRegistry, dest: WritableRegistry) -> dict[DatasetRef, Manifest]:
    """Return the manifests of a release and of every payload source that it needs.

    Each manifest comes from ``source``. A payload source that ``source`` does not hold comes from
    ``dest``, if ``dest`` holds a complete copy. Each payload source must match the files that each
    release that reads it pins.

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
            raise TimeNetMissingParentError(reference, needed_by, True, str(source)) from error
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
