"""A registry backed by a local ``<root>/<dataset_id>/<version>/`` directory tree."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
from typing import BinaryIO
import uuid

import pyarrow.fs as pafs
from pydantic import ValidationError

from timenet._fs import delete_and_measure, same_directory, staged_directory
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetOriginError,
)
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest, PayloadSource, read_manifest
from timenet.registry._paths import release_dir
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_file_sizes, validate_manifest_reference
from timenet.registry.origin import Origin, OriginKind, without_credentials
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.sources import FoundInRegistry, NotFound, ReleaseLookup, ReleaseSource
from timenet.types import DatasetMetadata, Version
from timenet.types._model import _first_problem
from timenet.types.metadata_fields import validate_dataset_id
from timenet.types.references import DatasetRef, is_latest
from timenet.writer import TimeFWriter, WriteProgressEvent


_ORIGIN_FILE = ".origin.json"
"""The provenance record of a release, in the folder of the release."""


class LocalRegistry(WritableRegistry):
    """Read and write dataset versions in a local directory."""

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        """Open a registry in a local directory.

        Args:
            root: The directory containing ``<dataset_id>/<version>/`` folders, with ``~`` expanded.
            read_only: Refuse every write to this registry.
        """
        self._root = Path(root).expanduser()
        self._read_only = read_only
        # (dataset_id, version) -> (manifest mtime, parsed manifest). This avoids re-parsing on repeat
        # reads, for example list_datasets then search's schema filter. An mtime change invalidates it.
        self._manifest_cache: dict[tuple[str, str], tuple[float, Manifest]] = {}

    def __str__(self) -> str:
        return f"local registry {self._root}"

    @property
    def location(self) -> str:
        """The registry's absolute root path."""
        return str(self._root.resolve())

    @property
    def root(self) -> Path:
        """The directory this registry reads from and writes to."""
        return self._root

    @property
    def requires_committed_code(self) -> bool:
        """Whether local releases must record committed code.

        Returns:
            ``False``, since a local registry accepts a build with local changes.
        """
        return False

    @property
    def requires_open_access(self) -> bool:
        """Whether local releases must declare open access.

        Returns:
            ``False``, since a local registry accepts any access level.
        """
        return False

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the latest-version metadata of every dataset, sorted by id.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        latest: dict[str, Version] = {}
        for release in self.releases():
            # The releases come sorted by version, so the last one of a dataset is its latest.
            latest[release.dataset_id] = release.version
        return [self.get_manifest(dataset_id, str(version)).metadata for dataset_id, version in latest.items()]

    def releases(self) -> list[DatasetRef]:
        """List the committed releases below the root.

        A release is a ``<org>/<name>/<version>/manifest.json`` outside hidden folders and unfinished writes.

        Returns:
            The releases, sorted by dataset ID and version.

        Raises:
            OSError: If a folder cannot be read.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _walk)
        return self._walk()[0]

    def release_at(self, path: Path) -> DatasetRef | None:
        """Return the release whose folder holds a path.

        Args:
            path: A path, whether or not it exists.

        Returns:
            The release, or ``None`` if no release folder of this registry holds the path.
        """
        resolved = path.expanduser().resolve()
        release_depth = 3  # <org>/<name>/<version> below the root
        for folder in (resolved, *resolved.parents):
            if len(folder.parents) < release_depth or not same_directory(folder.parents[release_depth - 1], self._root):
                continue
            try:
                release = DatasetRef.parse(f"{folder.parent.parent.name}/{folder.parent.name}@{folder.name}")
            except TimeFValidationError:
                return None
            return release if folder.name == str(release.version) else None
        return None

    def check_no_linked_folders(self) -> None:
        """Check that no dataset or release folder is a symlink, which :meth:`releases` would skip.

        Raises:
            TimeFValidationError: If a dataset or release folder is a symlink.
            OSError: If a folder cannot be read.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _walk)
        linked = self._walk()[1]
        if linked:
            raise TimeFValidationError(
                f"{linked[0]} is a symlink, so the clear cannot see the releases behind it or the "
                "releases that they need. Replace the link with its folder, then clear again."
            )

    def _walk(self) -> tuple[list[DatasetRef], list[Path]]:
        """Walk the dataset and release folders once, without following a symlink.

        Returns:
            The releases, sorted by dataset ID and version, and the linked dataset or release folders in walk order.

        Raises:
            OSError: If a folder cannot be read.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by os.walk)
        found: list[DatasetRef] = []
        linked: list[Path] = []
        if not self._root.is_dir():
            return found, linked
        release_depth = 3  # <org>/<name>/<version>

        def fail(error: OSError) -> None:
            raise error  # os.walk otherwise skips a folder that it cannot read

        for parent, dirs, files in os.walk(self._root, onerror=fail):
            folder = Path(parent)
            parts = folder.relative_to(self._root).parts
            if len(parts) == release_depth:
                dirs.clear()  # the walk stops at the releases
                release = self._release_at_parts(parts) if MANIFEST_FILE in files else None
                if release is not None:
                    found.append(release)
                continue
            dirs[:] = sorted(name for name in dirs if not _is_hidden_or_staging(name))
            linked.extend(folder / name for name in dirs if (folder / name).is_symlink())
        return sorted(found, key=lambda release: (release.dataset_id, release.version)), linked

    def _release_at_parts(self, parts: tuple[str, ...]) -> DatasetRef | None:
        """Return the release of a ``<org>/<name>/<version>`` folder, or ``None`` if the names are not one."""
        try:
            release = DatasetRef.parse(f"{parts[0]}/{parts[1]}@{parts[2]}")
        except TimeFValidationError:
            return None  # for example a directory whose name is not a version
        if release_dir(self._root, release) != self._root.joinpath(*parts):
            return None  # for example 01.0.0, which parses as 1.0.0
        return release

    def remove_release(self, release: DatasetRef) -> int:
        """Delete a release and the dataset folders that this leaves empty.

        Args:
            release: The release to delete.

        Returns:
            The bytes freed.

        Raises:
            TimeNetRegistryError: If the registry is read-only.
            OSError: If the release folder cannot be removed completely.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by check_writable and delete_and_measure)
        self.check_writable(f"a removal of {release}")
        directory = release_dir(self._root, release)
        freed = delete_and_measure(directory)
        parent = directory.parent
        while self._root in parent.parents and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return freed

    def record_origin(self, release: DatasetRef, kind: OriginKind, source: str) -> None:
        """Record how a stored release reached this registry, with the current time.

        The record replaces an earlier one. A URL in ``source`` loses its user info, query and fragment, so no
        credential lands in the record.

        Args:
            release: The release.
            kind: How the release reached this registry.
            source: The registry location, or the name and version of the backend package.

        Raises:
            TimeNetDatasetNotFoundError: If this registry does not hold the release.
        """
        if not self.exists(release.dataset_id, str(release.version)):
            raise TimeNetDatasetNotFoundError(f"the local registry holds no {release}, so it gets no provenance record")
        origin = Origin(kind=kind, source=without_credentials(source), at=datetime.now(UTC).replace(microsecond=0))
        directory = release_dir(self._root, release)
        # Write next to the record, then rename, so a reader never sees half a record.
        staging = directory / f"{_ORIGIN_FILE}.tmp-{uuid.uuid4().hex}"
        try:
            staging.write_text(origin.model_dump_json() + "\n")
            staging.replace(directory / _ORIGIN_FILE)
        finally:
            staging.unlink(missing_ok=True)

    def read_origin(self, release: DatasetRef) -> Origin | None:
        """Read the provenance record of a stored release.

        Args:
            release: The release.

        Returns:
            The record, or ``None`` if the release has none.

        Raises:
            TimeNetOriginError: If the record is invalid.
        """
        path = release_dir(self._root, release) / _ORIGIN_FILE
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        try:
            return Origin.model_validate_json(text)
        except ValidationError as error:
            raise TimeNetOriginError(release, str(path), _first_problem(error)) from error

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest (latest version if unspecified).

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` / ``"latest"`` for the latest.

        Returns:
            The dataset's manifest.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version has no committed manifest.
            TimeFFormatError: If the manifest is unreadable, invalid, or names a different dataset.
        """
        resolved = self._latest_version(dataset_id) if is_latest(version) else version
        if resolved is None:
            raise TimeNetDatasetNotFoundError(f"{self} holds no version of {dataset_id!r}")
        path = release_dir(self._root, DatasetRef.parse(f"{dataset_id}@{resolved}")) / MANIFEST_FILE
        if not path.exists():
            raise TimeNetDatasetNotFoundError(f"no manifest for {dataset_id!r} version {resolved!r}")
        mtime = path.stat().st_mtime
        cached = self._manifest_cache.get((dataset_id, resolved))
        if cached is not None and cached[0] == mtime:
            return cached[1]
        manifest = read_manifest(path)
        if manifest.metadata.dataset_id != dataset_id:
            raise TimeFFormatError(
                f"manifest under {dataset_id!r}/{resolved} declares dataset id "
                f"{manifest.metadata.dataset_id!r}; the on-disk layout is inconsistent"
            )
        self._manifest_cache[dataset_id, resolved] = (mtime, manifest)
        return manifest

    def resolve(self, dataset_id: str, version: Version | None) -> ReleaseLookup:
        """Check whether the local registry holds a dataset release.

        Local releases are available at any access level.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None`` for the latest version.

        Returns:
            ``FoundInRegistry`` with the committed version, or ``NotFound`` if the release is absent.

        Raises:
            TimeFValidationError: If ``dataset_id`` is invalid.
        """
        try:
            validate_dataset_id(dataset_id)
        except ValueError as error:
            raise TimeFValidationError(f"invalid dataset ID {dataset_id!r}: {error}") from error
        resolved = self._latest_version(dataset_id) if version is None else str(version)
        if resolved is None:
            return NotFound()
        reference = DatasetRef.parse(f"{dataset_id}@{resolved}")
        if not (release_dir(self._root, reference) / MANIFEST_FILE).exists():
            return NotFound()
        return FoundInRegistry(reference.version, self)

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Open one file of a dataset version for binary reading.

        Args:
            dataset_id: The dataset id.
            version: The version string.
            relpath: The file path relative to the version directory.

        Returns:
            An open binary file object.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset version directory does not exist.
            ValueError: If ``relpath`` escapes the dataset version directory.
        """
        version_dir = release_dir(self._root, DatasetRef.parse(f"{dataset_id}@{version}"))
        if not version_dir.is_dir():
            raise TimeNetDatasetNotFoundError(f"no dataset {dataset_id!r} version {version!r}")
        base = version_dir.resolve()
        target = (version_dir / relpath).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"relpath {relpath!r} escapes dataset {dataset_id!r} version {version!r}")
        return target.open("rb")

    def put_dataset(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> DatasetRef:
        """Write a dataset to this local registry.

        Return an existing release's reference unless ``force`` is set. Reused data must match the
        source releases stored in this registry.

        Args:
            dataset: The dataset to write.
            force: Replace an existing version.
            values_backend: Storage format for values (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional callback for write progress.
            pinned_by: Releases that already read this release's files, with the pins that the new files must match.

        Returns:
            A reference to the stored release.

        Raises:
            TimeNetRegistryError: If this registry is read-only.
            TimeNetDatasetNotFoundError: If a release that stores reused data is missing from this registry.
            TimeFFormatError: If a source release or the new output fails a file pin check.
        """  # ruff: ignore[docstring-extraneous-exception] - the writer checks payload sources before it publishes
        self.check_writable("put_dataset()")
        if dataset.schema is None:
            dataset.derive_schema()
        version = str(dataset.metadata.dataset_version)
        dataset_id = dataset.metadata.dataset_id
        # DatasetRef validates the release before a write can touch its directory.
        reference = DatasetRef.parse(f"{dataset_id}@{version}")
        if self.exists(dataset_id, version) and not force:
            return reference
        with TimeFWriter(
            self._root,
            dataset,
            values_backend=values_backend,
            progress_cb=progress_cb,
            replace=force,
            registry=self,
            pinned_by=pinned_by,
        ) as writer:
            writer.write()
        return reference

    def put_release(self, from_directory: Path) -> DatasetRef:
        """Copy a release directory into this registry.

        The directory must contain ``manifest.json`` and every file it lists. This method copies
        those files before replacing an existing version.

        Args:
            from_directory: The release directory.

        Returns:
            A reference to the published release.

        Raises:
            TimeNetRegistryError: If this registry is read-only.
            TimeFFormatError: If ``manifest.json`` is missing, unreadable, or invalid.
            TimeNetFileError: If a listed file is missing or its size differs from the manifest.
            OSError: If copying or renaming fails, with the backup path if restoring the old directory also fails.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _read_release and publish_directory
        self.check_writable("put_release()")
        manifest = self._read_release(from_directory)
        reference = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        with staged_directory(release_dir(self._root, reference)) as staging:
            for relpath in (*sorted({part.path for part in manifest.files.all_files()}), MANIFEST_FILE):
                target = staging / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(from_directory / relpath, target)
        return reference

    def damage(self, release: DatasetRef, manifest: Manifest | None = None) -> TimeNetFileError | None:
        """Return the first missing or resized file of a stored release, or ``None`` for a complete copy.

        Args:
            release: The exact release.
            manifest: Its manifest, or ``None`` to read the stored one.

        Returns:
            The error that names the file, or ``None``.
        """
        if manifest is None:
            manifest = self.get_manifest(release.dataset_id, str(release.version))
        version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(release_dir(self._root, release)))
        try:
            check_file_sizes(version, manifest.files.all_files(), owner=str(release))
        except TimeNetFileError as error:
            return error
        return None

    def open_release(self, reference: DatasetRef, *, manifest: Manifest | None = None) -> DatasetVersion:
        """Open a local release without attaching its payload sources.

        This checks that every file exists with its recorded size.

        Args:
            reference: The dataset ID and exact version to open.
            manifest: A manifest already read from this registry, or ``None`` to fetch it.

        Returns:
            A handle to the release's directory on the local filesystem.

        Raises:
            TimeNetDatasetNotFoundError: If the registry lacks a committed manifest for the release.
            TimeFFormatError: If the manifest declares a different release.
            TimeNetFileError: If a file is missing or has an incorrect size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by get_manifest and the checks
        if manifest is None:
            manifest = self.get_manifest(reference.dataset_id, str(reference.version))
        validate_manifest_reference(manifest, reference)
        version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(release_dir(self._root, reference)))
        check_file_sizes(version, manifest.files.all_files(), owner=str(reference))
        return version

    def _dataset_dir(self, dataset_id: str) -> Path:
        """Return the directory holding a dataset's versions.

        Args:
            dataset_id: The dataset id, used as a relative path under the root.

        Returns:
            The ``<root>/<dataset_id>/`` directory.
        """
        validate_dataset_id(dataset_id)
        return self._root / dataset_id

    def _latest_version(self, dataset_id: str) -> str | None:
        """Return the highest committed version string for a dataset, or ``None``."""
        dataset_dir = self._dataset_dir(dataset_id)
        if not dataset_dir.is_dir():
            return None
        versions = [
            entry.name
            for entry in dataset_dir.iterdir()
            if entry.is_dir() and _is_version(entry.name) and (entry / MANIFEST_FILE).exists()
        ]
        if not versions:
            return None
        return max(versions, key=Version.parse)


def is_same_registry(source: ReleaseSource, registry: BaseRegistry) -> bool:
    """Check whether a source matches a registry.

    Args:
        source: The source to compare.
        registry: The registry to compare against.

    Returns:
        ``True`` for the same object or local registries over the same directory.
    """
    if source is registry:
        return True
    return (
        isinstance(source, LocalRegistry)
        and isinstance(registry, LocalRegistry)
        and same_directory(source.root, registry.root)
    )


def _is_hidden_or_staging(name: str) -> bool:
    """Check whether a folder name is hidden or names an unfinished write.

    Args:
        name: The folder name.

    Returns:
        ``True`` if the name starts with ``.`` or holds ``.tmp-``.
    """
    return name.startswith(".") or ".tmp-" in name


def _is_version(name: str) -> bool:
    """Return whether a directory name is a parseable version.

    This skips non-version siblings such as a crashed build's ``<version>.tmp-<uuid>`` staging
    directory. Without the skip, parsing that name crashes :meth:`LocalRegistry._latest_version`.

    Args:
        name: The directory name.

    Returns:
        ``True`` if ``name`` parses as a ``major.minor.patch`` version.
    """
    try:
        Version.parse(name)
    except ValueError:
        return False
    return True
