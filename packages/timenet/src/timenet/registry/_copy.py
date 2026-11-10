"""Copy releases and their payload sources between registries.

A payload source is a release whose data files another release reads.
Manifests must match for existing releases, except for their build environments.
"""

from collections.abc import Callable
from pathlib import Path
import tempfile

from timenet.errors import TimeFValidationError, TimeNetMissingParentError, TimeNetVersionConflictError
from timenet.manifest import Manifest
from timenet.progress import BytesMoved, ProgressCallback, StepFinished, StepStarted
from timenet.registry._paths import release_dir
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import find_payload_sources, validate_manifest_reference
from timenet.registry.local import LocalRegistry
from timenet.registry.writable import WritableRegistry
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef


def copy_release(  # ruff: ignore[too-many-arguments]
    reference: DatasetRef,
    source: BaseRegistry,
    dest: WritableRegistry,
    *,
    with_payload_sources: bool = True,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[DatasetRef, ...]:
    """Copy a release and its payload sources to another registry.

    Copy payload sources first, reuse complete copies, and repair damaged local copies.
    Check all releases before copying any files.

    Args:
        reference: The exact release to copy.
        source: The registry to copy from.
        dest: The registry to copy to.
        with_payload_sources: Also copy the payload sources that the manifest of the release lists.
        force: Fetch ``reference`` again even if the local destination has a complete copy.
        progress: Receives step events for each copy and byte counts for local destinations.

    Returns:
        The copied or repaired releases, in copy order.

    Raises:
        TimeNetRegistryError: If ``dest`` is read-only.
        TimeFValidationError: If ``force`` is set and ``dest`` is not local.
        TimeNetVersionConflictError: If ``dest`` holds the same release with other files or another card.
        TimeNetAccessError: If a release lacks open access and either registry requires it.
        TimeNetBuildError: If ``dest`` is not local and a copied release records TimeNet code with local changes.
        TimeNetDatasetNotFoundError: If ``source`` lacks the release.
        TimeNetMissingParentError: If neither registry holds a complete payload source.
        TimeFFormatError: If a manifest names the wrong release or a pin fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
    dest.check_writable(f"a copy of {reference}")
    if force and not isinstance(dest, LocalRegistry):
        raise TimeFValidationError(f"force=True needs a local dest: {dest} never writes a published version again")
    manifests = _manifests(reference, source, dest, with_payload_sources=with_payload_sources)
    # Check every release before anything is copied, so that a refusal leaves dest unchanged. A payload
    # source that only dest holds is complete there, so it is never pending. _needs_copy() runs first,
    # so that other files raise also with force.
    pending = [
        ref
        for ref, manifest in manifests.items()
        if _needs_copy(reference, ref, manifest, source, dest) or (force and ref == reference)
    ]
    _check_access(manifests, set(pending), source, dest)
    for ref in pending:
        dest.require_committed_code(manifests[ref])
    if not pending:
        return ()
    # Only a copy from a non-local source to a non-local dest downloads into this directory, each
    # pending release on its own.
    with tempfile.TemporaryDirectory(prefix="timenet-copy-", ignore_cleanup_errors=True) as staging:
        for ref in pending:
            if progress is not None:
                progress(StepStarted(ref, "copy"))
            again = force and ref == reference
            _copy_one(ref, manifests[ref], source, dest, Path(staging), force=again, progress=progress)
            if progress is not None:
                progress(StepFinished(ref, "copy"))
    return tuple(pending)


def _copy_one(  # ruff: ignore[too-many-arguments]
    reference: DatasetRef,
    manifest: Manifest,
    source: BaseRegistry,
    dest: WritableRegistry,
    staging: Path,
    *,
    force: bool,
    progress: ProgressCallback | None,
) -> None:
    """Copy or repair one release without its payload sources in the destination registry.

    Args:
        reference: The release to copy.
        manifest: The release manifest from ``source``.
        source: The registry to copy from.
        dest: The registry to copy to.
        staging: A temporary directory for downloads from a remote source.
        force: Fetch the release again, even if ``dest`` has a complete local copy.
        progress: Receives byte progress events when ``dest`` is local.
    """
    # A local registry is a folder that the copy reads or writes directly.
    if isinstance(dest, LocalRegistry):
        report = None if progress is None else _bytes_of(reference, manifest, progress)
        directory = release_dir(dest.root, reference)
        source.download_release(reference, directory, force=force, manifest=manifest, progress_cb=report)
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


def _bytes_of(reference: DatasetRef, manifest: Manifest, progress: ProgressCallback) -> Callable[[int], None]:
    """Create a callback that reports cumulative bytes for a release.

    Args:
        reference: The release to report.
        manifest: The manifest whose file sizes define the total.
        progress: The callback that receives ``BytesMoved`` events.

    Returns:
        A callback that accepts each file or chunk's byte count and reports the bytes moved so far.
    """
    total = sum(part.size for part in manifest.files.all_files())
    done = 0

    def report(count: int) -> None:
        nonlocal done
        done += count
        progress(BytesMoved(str(reference), done, total))

    return report


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


def _manifests(
    reference: DatasetRef, source: BaseRegistry, dest: WritableRegistry, *, with_payload_sources: bool
) -> dict[DatasetRef, Manifest]:
    """Collect the manifests of a release and of the payload sources that its manifest lists.

    Use a complete copy in ``dest`` when ``source`` does not serve a payload source.

    Args:
        reference: The requested release.
        source: The registry to copy from.
        dest: The registry to copy to.
        with_payload_sources: Also collect the payload sources of the release.

    Returns:
        Manifests keyed by release, with the payload sources before the release.

    Raises:
        TimeNetAccessError: If ``source`` lists a payload source without serving it and ``dest`` lacks a complete copy.
        TimeNetMissingParentError: If neither ``source`` nor ``dest`` holds a complete payload source.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by require_open_access
    manifest = source.get_manifest(reference.dataset_id, str(reference.version))
    validate_manifest_reference(manifest, reference)
    if not with_payload_sources:
        return {reference: manifest}
    sources = find_payload_sources(
        reference, manifest, source, find=source.find_manifest, held=lambda ref: _complete_copy(ref, dest)
    )
    for missing in sources.missing:
        listed = source.find_manifest(missing)
        if listed is not None:
            # The source lists the payload source, but does not serve its files.
            require_open_access(listed.metadata, listed.lineage)
        raise TimeNetMissingParentError(missing, reference, f"{source} or {dest}", payload_source=True)
    return {**sources.fetched, **sources.held, reference: manifest}


def _complete_copy(reference: DatasetRef, dest: WritableRegistry) -> Manifest | None:
    """Return the manifest of a complete copy in a registry.

    Args:
        reference: The release.
        dest: The registry to look in.

    Returns:
        The manifest, or ``None`` if ``dest`` lacks the release or a local file is missing or resized.
    """
    held = dest.find_manifest(reference)
    return held if held is not None and _complete(reference, held, dest) else None


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
    difference = held.difference_from(manifest)
    if difference is None:
        return not _complete(reference, manifest, dest)
    needed_by = None if reference == requested else requested
    raise TimeNetVersionConflictError(reference, str(dest), dest.location, str(source), difference, needed_by)


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
