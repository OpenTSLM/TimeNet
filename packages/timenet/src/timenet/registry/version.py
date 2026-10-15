"""Prepare complete local releases, then read through simple, picklable version handles."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE, check_relative_path
from timenet.manifest import Manifest
from timenet.registry.shared import complete_copy, dependency_references
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


@dataclass(frozen=True)
class DatasetVersion:
    """A prepared release: its manifest, filesystem, and already-opened payload sources."""

    manifest: Manifest
    filesystem: pafs.FileSystem
    root: str
    payload_sources: dict[str, "DatasetVersion"] = field(default_factory=dict, compare=False, repr=False)

    def resolve_payload_source(self, key: str | None) -> "DatasetVersion":
        """Return the prepared version that stores a chunk's bytes.

        Args:
            key: The chunk's ``owner_key``, naming an entry in ``manifest.payload_sources``.
                ``None`` means the bytes belong to this version.

        Returns:
            This version or an already-prepared payload source. No files are opened here.

        Raises:
            TimeFFormatError: If the catalog refers to an unknown payload source.
        """
        if key is None:
            return self
        try:
            return self.payload_sources[key]
        except KeyError as error:
            raise TimeFFormatError(f"unknown payload source {key!r} in {self.manifest.dataset_id}") from error

    def resolve_file_path(self, relative_path: str) -> str:
        """Return a file's filesystem path without checking or downloading the file.

        Args:
            relative_path: A path within this version, such as ``time_series/part-000.parquet``.
                Absolute paths and paths that escape the version directory are rejected.

        Returns:
            The path to use with :attr:`filesystem`.
        """
        check_relative_path("relative_path", relative_path)
        return f"{self.root}/{relative_path}"

    def store_uri(self, relpath: str) -> str:
        """Return a location for a store-oriented backend such as Zarr.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The local path, or a scheme-qualified URI for an S3 filesystem.
        """
        path = self.resolve_file_path(relpath)
        return f"s3://{path}" if isinstance(self.filesystem, pafs.S3FileSystem) else path

    @classmethod
    def open_local(cls, root: str | Path) -> "DatasetVersion":
        """Prepare a local version and all its dependencies before reading.

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
    """Ensure the requested dataset and all its dependencies have complete local releases.

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
        for parent in dependency_references(release):
            prepare_dataset(parent)
        sources = {key: prepare_dataset(source) for key, source in release.payload_sources.items()}
        version = DatasetVersion(release, pafs.LocalFileSystem(), str(directory), sources)
        prepared[str(ref)] = version
        visiting.remove(str(ref))
        return version

    return prepare_dataset(reference)


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
