"""A registry backed by a local ``<root>/<dataset_id>/<version>/`` directory tree."""

from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
from typing import BinaryIO

import pyarrow.fs as pafs

from timenet._fs import staged_directory
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError, TimeNetFileError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest, PayloadSource, read_manifest
from timenet.registry._paths import release_dir
from timenet.registry.dependencies import check_file_sizes, open_local_release
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata, Version
from timenet.types.metadata_fields import validate_dataset_id
from timenet.types.references import DatasetRef, is_latest
from timenet.writer import TimeFWriter, WriteProgressEvent


class LocalRegistry(WritableRegistry):
    """Read and publish datasets in a local directory."""

    def __init__(self, root: Path) -> None:
        """Open a local registry rooted at a directory.

        Args:
            root: The directory containing ``<dataset_id>/<version>/`` layouts. This expands ``~``.
        """
        self._root = Path(root).expanduser()
        # (dataset_id, version) -> (manifest mtime, parsed manifest). This avoids re-parsing on repeat
        # reads, for example list_datasets then search's schema filter. An mtime change invalidates it.
        self._manifest_cache: dict[tuple[str, str], tuple[float, Manifest]] = {}

    def __str__(self) -> str:
        return f"local registry {self._root}"

    @property
    def root(self) -> Path:
        """The directory this registry reads from and writes to."""
        return self._root

    @property
    def local_root(self) -> Path:
        """The directory that holds complete local releases."""
        return self._root

    @property
    def requires_open_access(self) -> bool:
        """Whether local releases must declare open access.

        Returns:
            ``False``, since a local registry accepts any access level.
        """
        return False

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the latest-version metadata of every dataset, sorted by id.

        Search at any depth. This finds both flat (``hello_world``) and namespaced (``org/name``)
        layouts. A dataset id is the path from the root to a version directory's parent.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        if not self._root.is_dir():
            return []
        latest_versions: dict[str, str] = {}
        for manifest_path in self._root.rglob(MANIFEST_FILE):
            version_dir = manifest_path.parent
            parts = version_dir.relative_to(self._root).parts
            if any(part.startswith(".") or ".tmp-" in part for part in parts):
                continue
            dataset_id = "/".join(parts[:-1])
            version = parts[-1]
            if not dataset_id or not _is_version(version):
                continue
            current = latest_versions.get(dataset_id)
            if current is None or Version.parse(version) > Version.parse(current):
                latest_versions[dataset_id] = version
        return [
            self.get_manifest(dataset_id, version).metadata for dataset_id, version in sorted(latest_versions.items())
        ]

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
            known = ", ".join(m.dataset_id for m in self.list_datasets()) or "(none)"
            raise TimeNetDatasetNotFoundError(
                f"no committed version for dataset {dataset_id!r}; this registry has: {known}. "
                "Build one with `timenet-build build <id>`."
            )
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
            TimeNetDatasetNotFoundError: If a release that stores reused data is missing from this registry.
            TimeFFormatError: If a source release or the new output fails a file pin check.
        """  # ruff: ignore[docstring-extraneous-exception] - the writer checks payload sources before it publishes
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
            TimeFFormatError: If ``manifest.json`` is missing, unreadable, or invalid.
            TimeNetFileError: If a listed file is missing or its size differs from the manifest.
            OSError: If copying or renaming fails, with the backup path if restoring the old directory also fails.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _read_release and publish_directory
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

    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Open a committed version as a local, random-access handle.

        Reuses the manifest :meth:`get_manifest` already parsed and validated, so a missing version or a
        misplaced artifact surfaces there. Roots the handle at the version's directory over a
        :class:`pyarrow.fs.LocalFileSystem`, which is zero network and already seekable. Opening reads
        only metadata, as :func:`~timenet.registry.open_local_version` does, and finds payload sources
        in this registry.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            A handle to the committed version's manifest and files.
        """
        manifest = self.get_manifest(dataset_id, version)
        reference = DatasetRef(dataset_id=dataset_id, version=manifest.metadata.dataset_version)
        return open_local_release(release_dir(self._root, reference), manifest, sources_root=self._root)

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
