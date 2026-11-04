"""Copy releases and their payload sources between registries.

A payload source is a release whose data files another release reads.
Manifests must match for existing releases, except for their build environments.
"""

from pathlib import Path
import tempfile

from timenet.errors import TimeFFormatError, TimeNetMissingParentError, TimeNetVersionConflictError
from timenet.manifest import Manifest, same_release
from timenet.progress import ProgressCallback, StepFinished, StepStarted
from timenet.registry._paths import release_dir
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_pins, prepare_local_copy, validate_manifest_reference
from timenet.registry.local import LocalRegistry
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
    """Copy a release and its payload sources to another registry.

    Copy payload sources first, reuse complete releases in ``dest``, and fetch damaged local copies again.
    All checks run before the first copy, so a failed check leaves ``dest`` unchanged.

    Args:
        reference: The exact release to copy.
        source: The registry to copy from.
        dest: The registry to copy to.
        progress: Receives a ``StepStarted`` and a ``StepFinished`` event for each copied release.

    Raises:
        TimeNetVersionConflictError: If ``dest`` holds a release of the copy with other files or another card.
        TimeNetAccessError: If a release is not open, and ``dest`` or ``source`` requires open access.
        TimeNetDatasetNotFoundError: If ``source`` does not hold the release.
        TimeNetMissingParentError: If neither registry holds a complete payload source.
        TimeFFormatError: If a manifest names another release, a pin fails, or sources form a cycle.
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
    """Copy one release to ``dest`` or repair its local copy.

    Args:
        reference: The release to copy.
        manifest: The release's manifest from ``source``.
        source: The registry to copy from.
        dest: The registry to copy to.
        staging: Temporary download root when both registries are remote.
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
        source.download_release(reference, directory, manifest=manifest)
    dest.put_release(directory)


def _check_access(
    manifests: dict[DatasetRef, Manifest], fetched: set[DatasetRef], source: BaseRegistry, dest: WritableRegistry
) -> None:
    """Check release access against the source and destination requirements.

    Check all releases against ``dest`` and fetched releases against ``source``.

    Args:
        manifests: Release references mapped to their manifests.
        fetched: The releases that the copy fetches from ``source``.
        source: The registry to copy from.
        dest: The registry to copy to.

    Raises:
        TimeNetAccessError: If a release or its lineage lacks open access where a registry requires it.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by require_open_access
    for reference, manifest in manifests.items():
        dest_needs_open = dest.requires_open_access
        source_needs_open = reference in fetched and source.requires_open_access
        if dest_needs_open or source_needs_open:
            require_open_access(manifest.metadata, manifest.lineage)


def _manifests(reference: DatasetRef, source: BaseRegistry, dest: WritableRegistry) -> dict[DatasetRef, Manifest]:
    """Collect manifests for a release and all its payload sources.

    Use a complete copy in ``dest`` when ``source`` lacks a payload source.
    Each payload source must match the file records pinned by every release that reads it.

    Args:
        reference: The requested release.
        source: The registry to copy from.
        dest: The registry to copy to.

    Returns:
        Manifests keyed by release, with each payload source before the releases that read it.

    Raises:
        TimeNetMissingParentError: If ``source`` lacks a payload source and ``dest`` lacks a complete copy.
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
    """Find a payload source's manifest in ``source`` or a complete copy in ``dest``.

    Args:
        source: The registry to copy from.
        dest: The registry to copy to.
        reference: The payload source to find.
        needed_by: The release that reads the payload source.

    Returns:
        The payload source's manifest.

    Raises:
        TimeNetMissingParentError: If ``source`` lacks the payload source and ``dest`` lacks a complete copy.
    """
    manifest = source.find_manifest(reference)
    if manifest is not None:
        return manifest
    held = dest.find_manifest(reference)
    if held is None or not _complete(reference, held, dest):
        raise TimeNetMissingParentError(reference, needed_by, f"{source} or {dest}", payload_source=True)
    return held


def _needs_copy(
    requested: DatasetRef,
    reference: DatasetRef,
    manifest: Manifest,
    source: BaseRegistry,
    dest: WritableRegistry,
) -> bool:
    """Check whether a release needs copying or local repair.

    Args:
        requested: The requested release, for the error message.
        reference: The release to check.
        manifest: The release's manifest from ``source``, or from ``dest`` if only ``dest`` holds it.
        source: The source registry, for the error message.
        dest: The registry to copy to.

    Returns:
        ``True`` if ``dest`` lacks the release or its local copy has a missing or resized file.

    Raises:
        TimeNetVersionConflictError: If ``dest`` holds the same version with other files or another card.
    """
    held = dest.find_manifest(reference)
    if held is None:
        return True
    files_differ = _file_records(held) != _file_records(manifest)
    if not files_differ and same_release(held, manifest):
        return not _complete(reference, manifest, dest)
    needed_by = None if reference == requested else requested
    raise TimeNetVersionConflictError(reference, str(dest), str(source), needed_by, "files" if files_differ else "card")


def _complete(reference: DatasetRef, manifest: Manifest, dest: WritableRegistry) -> bool:
    """Check if a stored release is complete.

    A local copy is complete when every file has its recorded size.
    Treat releases in non-local registries as complete.

    Args:
        reference: The release to check.
        manifest: The manifest that records the files in ``dest``.
        dest: The registry that holds the release.

    Returns:
        ``True`` for a complete local copy or any release in a non-local registry.
    """
    return not isinstance(dest, LocalRegistry) or dest.damage(reference, manifest) is None


def _file_records(manifest: Manifest) -> frozenset[tuple[str, int, str]]:
    """Collect each file's path, size, and checksum from a manifest.

    Args:
        manifest: The manifest to read.

    Returns:
        A frozen set of ``(path, size, checksum)`` entries.
    """
    return frozenset((part.path, part.size, part.checksum) for part in manifest.files.all_files())
