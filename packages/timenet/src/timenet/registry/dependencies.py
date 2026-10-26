"""Resolve read dependencies and prepare complete local releases for downloads and builds."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest
from timenet.registry._paths import release_dir, split_release_dir
from timenet.registry.shared import complete_copy
from timenet.registry.version import DatasetVersion
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


def open_local_version(root: str | Path) -> "DatasetVersion":
    """Prepare a local version and its physical payload sources before reading.

    Args:
        root: The version directory (``<registry>/<org>/<name>/<version>``).

    Returns:
        A handle with all payload sources resolved. Preparation never opens DuckDB catalogs.

    Raises:
        FileNotFoundError: If ``root`` has no manifest.
    """
    root = Path(root)
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    registry_root, reference = split_release_dir(root)
    return prepare_dependency_graph(registry_root, reference, manifest=manifest)


def prepare_dependency_graph(  # noqa: PLR0913
    root: str | Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None" = None,
    build: Callable[[DatasetRef], None] | None = None,
    manifest: Manifest | None = None,
    force: bool = False,
    progress_cb: Callable[[int], None] | None = None,
) -> DatasetVersion:
    """Ensure the requested release and its physical payload sources are complete locally.

    Connector builds prepare declared inputs separately. An existing release needs
    only the files reached through ``payload_sources``, regardless of its build provenance.

    Args:
        root: The local registry or cache directory.
        reference: The exact dataset ID and version to prepare.
        fetch_from: Fetch missing releases from this registry. ``None`` disables fetching.
        build: Build a missing release when fetching cannot find it. ``None`` disables building.
        manifest: An already-loaded manifest for the requested release.
        force: Fetch the requested release again, even if it is cached.
        progress_cb: Report downloaded bytes.

    Returns:
        The requested version with its physical sources already resolved.
    """
    root = Path(root)

    def open_complete(ref: DatasetRef) -> DatasetVersion:
        directory = release_dir(root, ref)
        release = _ensure_dataset_version_available_locally(
            directory,
            ref,
            fetch_from=fetch_from,
            build=build,
            manifest=manifest if ref == reference else None,
            force=force and ref == reference,
            progress_cb=progress_cb,
        )
        return DatasetVersion(release, pafs.LocalFileSystem(), str(directory))

    return _open_release_graph(reference, open_complete)


def resolve_read_dependency_graph(
    root: Path,
    reference: DatasetRef,
    open_remote_version: Callable[[DatasetRef], DatasetVersion],
) -> DatasetVersion:
    """Resolve physical sources without downloading complete releases or build-only inputs.

    Args:
        root: The cache directory, checked for complete releases before using remote files.
        reference: The exact dataset ID and version to open.
        open_remote_version: Open one remote release without resolving its dependencies.

    Returns:
        A local or remote handle with all physical source manifests resolved upfront.
        No catalogs or payload files are opened here.
    """

    def open_cached_or_remote(ref: DatasetRef) -> DatasetVersion:
        directory = release_dir(root, ref)
        cached = _load_complete_local_manifest(directory, ref)
        if cached is not None:
            return DatasetVersion(cached, pafs.LocalFileSystem(), str(directory))
        return open_remote_version(ref)

    return _open_release_graph(reference, open_cached_or_remote)


def _open_release_graph(reference: DatasetRef, open_release: Callable[[DatasetRef], DatasetVersion]) -> DatasetVersion:
    """Open a release and, recursively, the physical payload sources it reads from.

    Each release is opened once, however many releases share it.

    Args:
        reference: The release to open.
        open_release: Open one release without resolving its payload sources.

    Returns:
        The release with every payload source resolved.

    Raises:
        TimeFFormatError: If payload sources form a cycle or a manifest names a different release.
    """  # noqa: DOC502 - reference validation happens in a helper
    opened: dict[DatasetRef, DatasetVersion] = {}
    visiting: set[DatasetRef] = set()

    def visit(ref: DatasetRef) -> DatasetVersion:
        if ref in visiting:
            raise TimeFFormatError(f"dataset dependency cycle through {ref}")
        if ref not in opened:
            visiting.add(ref)
            version = open_release(ref)
            _validate_manifest_reference(version.manifest, ref)
            sources = {owner: visit(source) for owner, source in version.manifest.payload_sources.items()}
            opened[ref] = replace(version, payload_sources=sources)
            visiting.discard(ref)
        return opened[ref]

    return visit(reference)


def _ensure_dataset_version_available_locally(  # noqa: PLR0913
    directory: Path,
    reference: DatasetRef,
    *,
    fetch_from: "BaseRegistry | None",
    build: Callable[[DatasetRef], None] | None,
    manifest: Manifest | None,
    force: bool,
    progress_cb: Callable[[int], None] | None,
) -> Manifest:
    """Reuse, fetch, or build one complete release.

    Returns:
        The release manifest. File hashes are checked during transfer, not on every read.

    Raises:
        TimeNetDatasetNotFoundError: If a complete release is unavailable.
        TimeFFormatError: If a manifest identifies a different release.
    """  # noqa: DOC502 - helpers validate manifest references
    if manifest is not None:
        _validate_manifest_reference(manifest, reference)
    if not force:
        cached = _load_complete_local_manifest(directory, reference, manifest=manifest)
        if cached is not None:
            return cached

    if fetch_from is not None:
        try:
            return _download_dataset_version(
                directory,
                reference,
                fetch_from,
                manifest=manifest,
                progress_cb=progress_cb,
            )
        except TimeNetDatasetNotFoundError:
            if build is None:
                raise

    if build is not None:
        build(reference)
        built = _load_complete_local_manifest(directory, reference)
        if built is not None:
            return built

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
    _validate_manifest_reference(manifest, reference)
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
    _validate_manifest_reference(manifest, reference)
    registry._download_release(
        reference.dataset_id,
        str(reference.version),
        directory,
        force=True,
        manifest=manifest,
        progress_cb=progress_cb,
    )
    return manifest


def _validate_manifest_reference(manifest: Manifest, reference: DatasetRef) -> None:
    """Reject a manifest stored under the wrong release.

    Raises:
        TimeFFormatError: If the manifest's ID or version differs from the reference.
    """
    if manifest.dataset_id != reference.dataset_id or manifest.metadata.dataset_version != reference.version:
        raise TimeFFormatError(f"manifest for {reference} declares a different dataset ID or version")
