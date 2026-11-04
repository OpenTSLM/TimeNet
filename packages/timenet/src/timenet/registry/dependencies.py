"""Open releases with their payload sources, and prepare complete local copies for downloads.

Opening is eager but reads only metadata. It checks that every file the release reads is listed
and present, and that each payload source is the release it was written against. Values stay
lazy, and nothing is hashed: :meth:`~timenet.reader.TimeFReader.verify` hashes the files.

A release lists every payload source it reads from, flattened by the writer, so opening resolves
only those and never walks a source's own sources.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import FilePart, Manifest, PayloadSource, read_manifest
from timenet.registry._paths import release_dir
from timenet.registry.shared import complete_copy
from timenet.registry.version import DatasetVersion
from timenet.types.access import Access, effective_access
from timenet.types.reference_keys import SourceKey
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


def open_local_version(path: str | Path, *, sources_root: str | Path | None = None) -> DatasetVersion:
    # The checks below raise TimeFFormatError and TimeNetDatasetNotFoundError.
    # ruff: ignore[docstring-extraneous-exception]
    """Open a local release directory and its payload sources.

    Opening reads only metadata. The manifest must be valid, with a supported format version, and
    every file it lists must be present with its recorded size. Each payload source must be found
    under the sources root, match the files the release pins, and hold each of them with its
    recorded size.

    Args:
        path: The release directory.
        sources_root: The registry root to find payload sources under. By default this is the
            root that holds ``path`` as ``<root>/<org>/<name>/<version>``, so a release and its
            sources can move together. A release without payload sources opens from any directory.

    Returns:
        A handle with each payload source opened. No catalog or values file is opened.

    Raises:
        FileNotFoundError: If ``path`` has no manifest.
        TimeFValidationError: If the release has payload sources, is not stored as
            ``<root>/<org>/<name>/<version>``, and no ``sources_root`` is given.
    """
    directory = Path(path).expanduser()
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = read_manifest(manifest_path)
    if not manifest.payload_sources:
        return open_local_release(directory, manifest, sources_root=None)
    if sources_root is None:
        sources_root = _layout_root(directory, manifest)
    return open_local_release(directory, manifest, sources_root=Path(sources_root).expanduser())


def open_local_release(directory: Path, manifest: Manifest, *, sources_root: Path | None) -> DatasetVersion:
    """Open a local release whose manifest is already read, with its payload sources.

    Args:
        directory: The release directory.
        manifest: The release's manifest.
        sources_root: The registry root of the payload sources, required if the release has any.

    Returns:
        A handle with each payload source opened and checked.

    Raises:
        TimeNetFileError: If a required file is missing or its size differs from its manifest record.
        TimeFFormatError: If a payload source differs from the files the release pins.
        TimeNetMissingParentError: If a payload source is not under ``sources_root``.
        TimeFValidationError: If the release has payload sources but no ``sources_root``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the checks this function calls
    version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(directory))
    check_file_sizes(version, manifest.files.all_files(), owner=str(_reference(manifest)))
    if not manifest.payload_sources:
        return version
    if sources_root is None:
        raise TimeFValidationError(f"{_reference(manifest)} reuses payloads of other releases and needs a sources root")
    root = sources_root

    def open_source(reference: DatasetRef) -> DatasetVersion:
        source_dir = release_dir(root, reference)
        source_manifest = source_dir / MANIFEST_FILE
        if not source_manifest.is_file():
            raise TimeNetDatasetNotFoundError(f"no manifest at {source_manifest}")
        return DatasetVersion(read_manifest(source_manifest), version.filesystem, str(source_dir))

    return attach_payload_sources(version, open_source, searched=str(root))


def attach_payload_sources(
    version: DatasetVersion, open_source: Callable[[DatasetRef], DatasetVersion], *, searched: str
) -> DatasetVersion:
    """Open a release's payload sources and check each against the files the release pins.

    Only the release's own ``payload_sources`` are opened, since the writer lists every release
    whose files the release reads. A source's own sources are not opened.

    Args:
        version: The opened release, without its payload sources.
        open_source: Open one exact release, without its payload sources. It raises
            :class:`~timenet.errors.TimeNetDatasetNotFoundError` when the release is absent.
        searched: Where ``open_source`` looks, for the error that names a missing source.

    Returns:
        The release with each payload source attached.

    Raises:
        TimeNetMissingParentError: If a payload source is absent.
        TimeNetFileError: If a pinned local file is missing or has another size.
        TimeFFormatError: If a payload source's manifest names another release, lacks a pinned file,
            or records it differently.
    """  # ruff: ignore[docstring-extraneous-exception] - mismatches are raised by the checks this function calls
    release = _reference(version.manifest)
    sources: dict[SourceKey, DatasetVersion] = {}
    for key, pinned in version.manifest.payload_sources.items():
        try:
            source = open_source(pinned.release)
        except TimeNetDatasetNotFoundError as error:
            raise TimeNetMissingParentError(pinned.release, release, searched, payload_source=True) from error
        validate_manifest_reference(source.manifest, pinned.release)
        check_pins(release, pinned, source.manifest)
        check_file_sizes(source, pinned.files, owner=f"payload source {pinned.release} of {release}")
        sources[key] = source
    return replace(version, payload_sources=sources)


@dataclass(frozen=True)
class CopySources:
    """The payload sources of a release, by where a copy of the release finds them."""

    fetched: dict[DatasetRef, Manifest] = field(default_factory=dict)
    """The payload sources that the registry serves, with their manifests there. The copy brings them."""
    held: dict[DatasetRef, Manifest] = field(default_factory=dict)
    """The other payload sources that the destination holds as complete copies, with their manifests."""
    missing: dict[DatasetRef, PayloadSource] = field(default_factory=dict)
    """The payload sources that neither holds, with the pins of the release on each."""


def find_payload_sources(
    release: DatasetRef,
    manifest: Manifest,
    registry: "BaseRegistry",
    *,
    find: Callable[[DatasetRef], Manifest | None],
    held: Callable[[DatasetRef], Manifest | None],
) -> CopySources:
    """Find each payload source of a release in the registry to copy from, else in the destination.

    A release needs exactly the payload sources that its manifest lists, never their own payload sources.

    Args:
        release: The release to copy.
        manifest: The manifest of ``release``.
        registry: The registry to copy from.
        find: Return the manifest of a release in ``registry``, or ``None`` if it lacks the release.
        held: Return the manifest of a complete copy in the destination, or ``None``.

    Returns:
        The payload sources, by where the copy finds them.

    Raises:
        TimeFFormatError: If a payload source that the copy finds differs from the files that ``release`` pins.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
    sources = CopySources()
    for pinned in manifest.payload_sources.values():
        source = pinned.release
        found = find(source)
        if found is not None and _serves(registry, found):
            sources.fetched[source] = found
        elif (found := held(source)) is not None:
            sources.held[source] = found
        else:
            sources.missing[source] = pinned
            continue
        check_pins(release, pinned, found)
    return sources


def _serves(registry: "BaseRegistry", manifest: Manifest) -> bool:
    """Check whether a registry serves the files of a release that it holds.

    A registry that requires open access serves a release only if its card and each card in its lineage are open.

    Args:
        registry: The registry that holds the release.
        manifest: The manifest of the release in ``registry``.

    Returns:
        ``True`` if the registry serves the files.
    """
    open_access = effective_access(manifest.metadata, manifest.lineage) is Access.OPEN
    return open_access or not registry.requires_open_access


def check_pins(release: DatasetRef, pinned: PayloadSource, manifest: Manifest) -> None:
    """Check that a payload source still records each pinned file as the release pins it.

    Only the two manifests are compared; no file is read.

    Args:
        release: The release that pins the files, for the error message.
        pinned: The release's pins on one payload source.
        manifest: The payload source's own manifest.

    Raises:
        TimeFFormatError: Naming the source and the first differing file, in path order.
    """
    recorded = {part.path: part for part in manifest.files.all_files()}
    for part in sorted(pinned.files, key=lambda item: item.path):
        actual = recorded.get(part.path)
        if actual == part:
            continue
        found = "does not list it" if actual is None else f"records {_describe(actual)}"
        raise TimeFFormatError(
            f"payload source {pinned.release} differs from the release {release} was written against: "
            f"{release} pins {part.path!r} as {_describe(part)}, but {pinned.release} {found}"
        )


def check_file_sizes(version: DatasetVersion, parts: Iterable[FilePart], *, owner: str) -> None:
    """Check that each file of a local release is present with its recorded size.

    Remote releases are not checked here: their sizes come from the manifest, and remote reads and
    downloads check them.

    Args:
        version: The release that stores the files.
        parts: The manifest records of the files to check.
        owner: The release, or payload source, that the files belong to, for the error message.

    Raises:
        TimeNetFileError: Naming the first missing or resized file, its recorded and actual size.
    """
    if not isinstance(version.filesystem, pafs.LocalFileSystem):
        return
    root = Path(version.root)
    release = _reference(version.manifest)
    for part in parts:
        try:
            size = (root / part.path).stat().st_size
        except FileNotFoundError:
            raise TimeNetFileError(release, owner, part.path, f"{part.size:,} bytes", None) from None
        if size != part.size:
            raise TimeNetFileError(release, owner, part.path, f"{part.size:,} bytes", f"{size:,} bytes")


def prepare_local_copy(  # ruff: ignore[too-many-arguments]
    root: str | Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None" = None,
    manifest: Manifest | None = None,
    force: bool = False,
    progress_cb: Callable[[int], None] | None = None,
    before_fetch: Callable[[Manifest], object] | None = None,
) -> DatasetVersion:
    """Ensure complete local copies of a release and of each of its payload sources.

    Only the release's own payload sources are prepared, never their sources. Build-only parents stay
    recorded in the manifest and are not prepared. A payload source must be a complete copy, not
    only the files that the release pins. Then the source opens and passes its checks as a release
    of its own.

    For each release, this function reuses a complete local copy. If no complete local copy exists,
    it fetches the release from ``fetch_from``. A local copy is damaged if its manifest is present
    but a file is missing or has another size. A fetch repairs a damaged copy. If a local copy is
    damaged and ``fetch_from`` is not given, this function raises an error that names the first
    missing or resized file.

    Args:
        root: The local registry or cache directory.
        reference: The exact dataset ID and version to prepare.
        fetch_from: Fetch missing releases from this registry. ``None`` disables fetching.
        manifest: An already-loaded manifest for the requested release.
        force: Fetch the requested release again, even if it is cached.
        progress_cb: Report downloaded bytes.
        before_fetch: A callback that gets the manifest of each release before this function
            downloads the files of the release. For example, the callback can check the access
            requirement of the release.

    Returns:
        The requested release with its payload sources attached and checked against its pins.

    Raises:
        TimeNetDatasetNotFoundError: If the requested release is absent and this function cannot
            fetch it.
        TimeNetMissingParentError: If a payload source is absent and this function cannot fetch it.
        TimeNetFileError: If a local copy is damaged and ``fetch_from`` is not given, or if a
            fetched file does not match its manifest record.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
    root = Path(root)

    def open_complete(ref: DatasetRef, *, requested: bool) -> DatasetVersion:
        directory = release_dir(root, ref)
        release = _ensure_dataset_version_available_locally(
            directory,
            ref,
            fetch_from=fetch_from,
            manifest=manifest if requested else None,
            force=force and requested,
            progress_cb=progress_cb,
            before_fetch=before_fetch,
        )
        return DatasetVersion(release, pafs.LocalFileSystem(), str(directory))

    version = open_complete(reference, requested=True)
    return attach_payload_sources(version, lambda ref: open_complete(ref, requested=False), searched=str(root))


def open_cached_or_remote(
    root: Path,
    reference: DatasetRef,
    open_remote_version: Callable[[DatasetRef], DatasetVersion],
    *,
    searched: str,
) -> DatasetVersion:
    """Open a release and its payload sources from complete cached copies or remotely.

    Args:
        root: The cache directory, checked for complete releases before using remote files.
        reference: The exact dataset ID and version to open.
        open_remote_version: Open one remote release without its payload sources.
        searched: The remote registry, for the error that names a missing source.

    Returns:
        A local or remote handle with each payload source attached and checked against its pins.
        No catalogs or payload files are opened here.
    """

    def open_one(ref: DatasetRef) -> DatasetVersion:
        directory = release_dir(root, ref)
        cached = _load_complete_local_manifest(directory, ref)
        if cached is not None:
            return DatasetVersion(cached, pafs.LocalFileSystem(), str(directory))
        return open_remote_version(ref)

    version = open_one(reference)
    validate_manifest_reference(version.manifest, reference)
    return attach_payload_sources(version, open_one, searched=f"{searched} or its cache {root}")


def _ensure_dataset_version_available_locally(  # ruff: ignore[too-many-arguments]
    directory: Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None",
    manifest: Manifest | None,
    force: bool,
    progress_cb: Callable[[int], None] | None,
    before_fetch: Callable[[Manifest], object] | None = None,
) -> Manifest:
    """Reuse or fetch one complete release, and repair a damaged copy only by fetching it.

    Returns:
        The release manifest. File hashes are checked during transfer, not on every read.

    Raises:
        TimeNetDatasetNotFoundError: If the release is absent and this function cannot fetch it.
        TimeNetFileError: If a local file is damaged and ``fetch_from`` is not given, or a fetched file does not match.
        TimeFFormatError: If a manifest identifies a different release.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
    if manifest is not None:
        validate_manifest_reference(manifest, reference)
    if not force and (directory / MANIFEST_FILE).is_file():
        local = manifest if manifest is not None else read_manifest(directory / MANIFEST_FILE)
        validate_manifest_reference(local, reference)
        try:
            check_file_sizes(
                DatasetVersion(local, pafs.LocalFileSystem(), str(directory)),
                local.files.all_files(),
                owner=str(reference),
            )
        except TimeNetFileError:
            if fetch_from is None:
                raise
        else:
            return local
    if fetch_from is not None:
        return _download_dataset_version(
            directory, reference, fetch_from, manifest=manifest, progress_cb=progress_cb, before_fetch=before_fetch
        )
    raise TimeNetDatasetNotFoundError(f"complete release {reference} is missing at {directory}")


def _load_complete_local_manifest(
    directory: Path,
    reference: DatasetRef,
    *,
    manifest: Manifest | None = None,
) -> Manifest | None:
    """Load and validate a local release, returning None if files are missing or incomplete.

    Returns:
        The manifest if it exists and every listed file has the recorded size, otherwise ``None``.
    """
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    if manifest is None:
        manifest = read_manifest(manifest_path)
    validate_manifest_reference(manifest, reference)
    return manifest if complete_copy(directory, manifest) else None


def _download_dataset_version(  # ruff: ignore[too-many-arguments]
    directory: Path,
    reference: DatasetRef,
    registry: "BaseRegistry",
    *,
    manifest: Manifest | None,
    progress_cb: Callable[[int], None] | None,
    before_fetch: Callable[[Manifest], object] | None = None,
) -> Manifest:
    """Fetch one complete release, checking its identity before downloading its files.

    Returns:
        The manifest after the registry has verified and stored the downloaded files.
    """
    if manifest is None:
        manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
    validate_manifest_reference(manifest, reference)
    if before_fetch is not None:
        before_fetch(manifest)
    registry.download_release(
        reference.dataset_id,
        str(reference.version),
        directory,
        force=True,
        manifest=manifest,
        progress_cb=progress_cb,
    )
    return manifest


def validate_manifest_reference(manifest: Manifest, reference: DatasetRef) -> None:
    """Reject a manifest stored under the wrong release.

    Raises:
        TimeFFormatError: If the manifest's ID or version differs from the reference.
    """
    if _reference(manifest) != reference:
        raise TimeFFormatError(f"manifest for {reference} declares a different dataset ID or version")


def _layout_root(directory: Path, manifest: Manifest) -> Path:
    """Return the registry root that holds a release directory as ``<root>/<org>/<name>/<version>``.

    Raises:
        TimeFValidationError: If the directory does not end in the release's dataset ID and version.
    """
    reference = _reference(manifest)
    expected = Path(reference.dataset_id, str(reference.version)).parts
    resolved = directory.resolve()
    if resolved.parts[-len(expected) :] != expected:
        raise TimeFValidationError(
            f"{reference} at {directory} reuses payloads of other releases, which are found relative to its "
            f"registry root; open it from <root>/{reference.dataset_id}/{reference.version} or pass sources_root="
        )
    return resolved.parents[len(expected) - 1]


def _reference(manifest: Manifest) -> DatasetRef:
    """Return the exact release a manifest describes."""
    return DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)


def _describe(part: FilePart) -> str:
    """Describe a file's manifest record for an error message.

    Returns:
        Its size and checksum.
    """
    return f"{part.size} bytes with checksum {part.checksum}"
