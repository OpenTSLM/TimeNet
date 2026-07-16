"""A registry backed by a local ``<root>/<dataset_id>/<version>/`` directory tree."""

from collections.abc import Callable
from pathlib import Path
import shutil
from typing import BinaryIO

from timenet.dataset import TimeFDataset
from timenet.errors import DatasetNotFoundError
from timenet.manifest import Manifest
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata, Version
from timenet.writer import TimeFWriter, WriteProgressEvent
from timenet.writer.constants import MANIFEST_FILE


class LocalRegistry(WritableRegistry):
    """Serves datasets from a local directory (the output of curation is itself a valid one)."""

    def __init__(self, root: Path) -> None:
        """Open a local registry rooted at a directory.

        Args:
            root: The directory containing ``<dataset_id>/<version>/`` layouts. ``~`` is expanded.
        """
        self._root = Path(root).expanduser()
        # (dataset_id, version) -> (manifest mtime, parsed manifest); avoids re-parsing on repeat reads
        # (e.g. list_datasets then search's schema filter). Invalidated when the file's mtime changes.
        self._manifest_cache: dict[tuple[str, str], tuple[float, Manifest]] = {}

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the latest-version metadata of every dataset, sorted by id.

        Discovers datasets depth-agnostically so both flat (``hello_world``) and namespaced
        (``org/name``) layouts are found. A dataset id is the path from the root to a version
        directory's parent.

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
            DatasetNotFoundError: If the dataset id or version has no committed manifest.
        """
        resolved = self._latest_version(dataset_id) if version in (None, "", "latest") else version
        if resolved is None:
            raise DatasetNotFoundError(f"no committed version for dataset {dataset_id!r}")
        path = self._root / dataset_id / resolved / MANIFEST_FILE
        if not path.exists():
            raise DatasetNotFoundError(f"no manifest for {dataset_id!r} version {resolved!r}")
        mtime = path.stat().st_mtime
        cached = self._manifest_cache.get((dataset_id, resolved))
        if cached is not None and cached[0] == mtime:
            return cached[1]
        manifest = Manifest.from_json(path.read_text())
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
            DatasetNotFoundError: If the dataset version directory does not exist.
            ValueError: If ``relpath`` escapes the dataset version directory.
        """
        version_dir = self._root / dataset_id / version
        if not version_dir.is_dir():
            raise DatasetNotFoundError(f"no dataset {dataset_id!r} version {version!r}")
        base = version_dir.resolve()
        target = (version_dir / relpath).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"relpath {relpath!r} escapes dataset {dataset_id!r} version {version!r}")
        return target.open("rb")

    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Compile a dataset and write it into this registry's directory tree.

        Streams the dataset through a :class:`~timenet.writer.TimeFWriter`, which stages under
        ``<version>.tmp-*`` and publishes with a single atomic rename. An already-committed version is
        skipped unless ``force`` is set.

        Args:
            dataset: The populated dataset to store.
            force: Overwrite an already-committed version instead of skipping it.
            values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional writer progress callback.

        Returns:
            The stored version string.
        """
        if dataset.schema is None:
            dataset.derive_schema()
        version = str(dataset.metadata.dataset_version)
        if self.exists(dataset.metadata.dataset_id, version) and not force:
            return version
        if force:
            final_dir = self._root / dataset.metadata.dataset_id / version
            if final_dir.exists():
                shutil.rmtree(final_dir)
        with TimeFWriter(self._root, dataset, values_backend=values_backend, progress_cb=progress_cb) as writer:
            writer.write()
        return version

    def _latest_version(self, dataset_id: str) -> str | None:
        """Return the highest committed version string for a dataset, or ``None``."""
        dataset_dir = self._root / dataset_id
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

    Skips non-version siblings such as a crashed build's ``<version>.tmp-<uuid>`` staging directory,
    which would otherwise crash :meth:`LocalRegistry._latest_version` when parsed.

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
