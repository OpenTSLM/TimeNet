"""Resolve read dependencies and prepare complete local releases for downloads and builds."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest
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
    reference = DatasetRef.parse(f"{root.parent.parent.name}/{root.parent.name}@{root.name}")
    return prepare_dependency_graph(root.parents[2], reference, manifest=manifest)


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

    Raises:
        TimeFFormatError: If references form a cycle or a manifest identifies the wrong release.
    """  # noqa: DOC502 - preparation helpers raise the documented errors
    root = Path(root)
    prepared: dict[str, DatasetVersion] = {}
    visiting: set[str] = set()

    def prepare_dataset(ref: DatasetRef) -> DatasetVersion:
        key = str(ref)
        if key in visiting:
            raise TimeFFormatError(f"dataset dependency cycle through {ref}")
        if key in prepared:
            return prepared[key]
        visiting.add(key)
        directory = root / ref.dataset_id / str(ref.version)
        release = _ensure_dataset_version_available_locally(
            directory,
            ref,
            fetch_from=fetch_from,
            build=build,
            manifest=manifest if ref == reference else None,
            force=force and ref == reference,
            progress_cb=progress_cb,
        )
        sources = {key: prepare_dataset(source) for key, source in release.payload_sources.items()}
        version = DatasetVersion(release, pafs.LocalFileSystem(), str(directory), sources)
        prepared[str(ref)] = version
        visiting.remove(str(ref))
        return version

    return prepare_dataset(reference)


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

    Raises:
        TimeFFormatError: If physical sources form a cycle or a manifest has the wrong identity.
    """  # noqa: DOC502 - reference validation happens in a helper
    resolved: dict[str, DatasetVersion] = {}
    visiting: set[str] = set()

    def resolve(ref: DatasetRef) -> DatasetVersion:
        key = str(ref)
        if key in visiting:
            raise TimeFFormatError(f"dataset dependency cycle through {ref}")
        if key in resolved:
            return resolved[key]
        visiting.add(key)
        directory = root / ref.dataset_id / str(ref.version)
        cached = _load_complete_local_manifest(directory, ref)
        version = (
            DatasetVersion(cached, pafs.LocalFileSystem(), str(directory))
            if cached is not None
            else open_remote_version(ref)
        )
        _validate_manifest_reference(version.manifest, ref)
        sources = {name: resolve(source) for name, source in version.manifest.payload_sources.items()}
        resolved[key] = replace(version, payload_sources=sources)
        visiting.remove(key)
        return resolved[key]

    return resolve(reference)


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

    raise TimeNetDatasetNotFoundError(
        f"complete release {reference} is missing at {directory}; "
        "fetch it first or enable --fetch-parents or --build-parents"
    )


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
