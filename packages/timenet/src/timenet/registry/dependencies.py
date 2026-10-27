"""Open releases with their payload sources, and prepare complete local copies for downloads.

Opening is eager but reads only metadata. It checks that every file the release reads is listed
and present, and that each payload source is the release it was written against. Values stay
lazy, and nothing is hashed: :meth:`~timenet.reader.TimeFReader.verify` hashes the files.

A release lists every payload source it reads from, flattened by the writer, so opening resolves
only those and never walks a source's own sources.
"""

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import FilePart, Manifest, PayloadSource
from timenet.registry._paths import release_dir
from timenet.registry.shared import complete_copy
from timenet.registry.version import DatasetVersion
from timenet.types.reference_keys import SourceKey
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


def open_local_version(path: str | Path, *, sources_root: str | Path | None = None) -> DatasetVersion:
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
    """  # noqa: DOC502 - the checks below raise TimeFFormatError and TimeNetDatasetNotFoundError
    directory = Path(path).expanduser()
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = Manifest.model_validate_json(manifest_path.read_text())
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
        sources_root: The registry root that holds its payload sources. Only a release without
            payload sources may omit it.

    Returns:
        A handle with each payload source opened and checked.

    Raises:
        TimeFFormatError: If a file is missing or has another size than recorded, or a payload source
            differs from the files the release pins.
        TimeNetDatasetNotFoundError: If a payload source is not under ``sources_root``.
        TimeFValidationError: If the release has payload sources but no ``sources_root``.
    """  # noqa: DOC502 - raised by the checks this function calls
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
        return DatasetVersion(
            Manifest.model_validate_json(source_manifest.read_text()), version.filesystem, str(source_dir)
        )

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
        TimeNetDatasetNotFoundError: If a payload source is absent.
        TimeFFormatError: If a payload source's manifest names another release, lacks a pinned file,
            records it differently, or a pinned local file is missing or has another size.
    """  # noqa: DOC502 - mismatches are raised by the checks this function calls
    release = _reference(version.manifest)
    sources: dict[SourceKey, DatasetVersion] = {}
    for key, pinned in version.manifest.payload_sources.items():
        try:
            source = open_source(pinned.release)
        except TimeNetDatasetNotFoundError as error:
            raise TimeNetDatasetNotFoundError(
                f"payload source {pinned.release} of {release} is not in {searched}"
            ) from error
        validate_manifest_reference(source.manifest, pinned.release)
        check_pins(release, pinned, source.manifest)
        check_file_sizes(source, pinned.files, owner=f"payload source {pinned.release} of {release}")
        sources[key] = source
    return replace(version, payload_sources=sources)


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
        TimeFFormatError: Naming the first missing or resized file, its recorded and actual size.
    """
    if not isinstance(version.filesystem, pafs.LocalFileSystem):
        return
    root = Path(version.root)
    for part in parts:
        try:
            size = (root / part.path).stat().st_size
        except FileNotFoundError:
            raise TimeFFormatError(f"{owner}: file {part.path!r} is missing; expected {part.size} bytes") from None
        if size != part.size:
            raise TimeFFormatError(f"{owner}: file {part.path!r} has {size} bytes; expected {part.size}")


def prepare_local_copy(  # noqa: PLR0913
    root: str | Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None" = None,
    manifest: Manifest | None = None,
    force: bool = False,
    progress_cb: Callable[[int], None] | None = None,
) -> DatasetVersion:
    """Ensure complete local copies of a release and of each of its payload sources.

    Only the release's own payload sources are prepared, never their sources. Build inputs stay
    recorded in the manifest and are not prepared.

    Args:
        root: The local registry or cache directory.
        reference: The exact dataset ID and version to prepare.
        fetch_from: Fetch missing releases from this registry. ``None`` disables fetching.
        manifest: An already-loaded manifest for the requested release.
        force: Fetch the requested release again, even if it is cached.
        progress_cb: Report downloaded bytes.

    Returns:
        The requested release with its payload sources attached and checked against its pins.
    """
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


def _ensure_dataset_version_available_locally(  # noqa: PLR0913
    directory: Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None",
    manifest: Manifest | None,
    force: bool,
    progress_cb: Callable[[int], None] | None,
) -> Manifest:
    """Reuse or fetch one complete release.

    Returns:
        The release manifest. File hashes are checked during transfer, not on every read.

    Raises:
        TimeNetDatasetNotFoundError: If a complete release is unavailable.
        TimeFFormatError: If a manifest identifies a different release.
    """  # noqa: DOC502 - helpers validate manifest references
    if manifest is not None:
        validate_manifest_reference(manifest, reference)
    if not force:
        cached = _load_complete_local_manifest(directory, reference, manifest=manifest)
        if cached is not None:
            return cached
    if fetch_from is not None:
        return _download_dataset_version(directory, reference, fetch_from, manifest=manifest, progress_cb=progress_cb)
    raise TimeNetDatasetNotFoundError(f"complete release {reference} is missing at {directory}")


def _load_complete_local_manifest(
    directory: Path,
    reference: DatasetRef,
    *,
    manifest: Manifest | None = None,
) -> Manifest | None:
    """Load and validate a local release, returning None if files are missing or incomplete.

    Returns:
        The manifest if all declared files exist with their expected sizes, otherwise None.
        Invalid manifests raise an error instead of being treated as missing releases.
    """
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    if manifest is None:
        manifest = Manifest.model_validate_json(manifest_path.read_text())
    validate_manifest_reference(manifest, reference)
    return manifest if complete_copy(directory, manifest) else None


def _download_dataset_version(
    directory: Path,
    reference: DatasetRef,
    registry: "BaseRegistry",
    *,
    manifest: Manifest | None,
    progress_cb: Callable[[int], None] | None,
) -> Manifest:
    """Fetch one complete release, checking its identity before downloading its files.

    Returns:
        The manifest after the registry has verified and stored the downloaded files.
    """
    if manifest is None:
        manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
    validate_manifest_reference(manifest, reference)
    registry._download_release(
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
