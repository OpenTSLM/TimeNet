"""Define the writable-registry contract for a backend that build publishes datasets into.

This class extends the read-only :class:`~timenet.registry.BaseRegistry` with two write primitives,
:meth:`WritableRegistry.put_dataset` and :meth:`WritableRegistry.put_release`. Backends differ only in where a
release lands, for example a local directory, an S3 prefix, or a remote service. So these two are the
abstract methods, and :meth:`exists` is shared.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

import pyarrow.fs as pafs

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_file_sizes
from timenet.registry.version import DatasetVersion
from timenet.types.references import DatasetRef
from timenet.writer import WriteProgressEvent


class WritableRegistry(BaseRegistry, ABC):
    """A registry that build can publish datasets into, not just read from."""

    @abstractmethod
    def put_dataset(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> DatasetRef:
        """Compile a dataset and publish it to this registry.

        If the dataset has no schema, this method derives one first. If the version is already
        committed, this method skips it unless ``force`` is set.

        Args:
            dataset: The populated dataset to store.
            force: Overwrite an already-committed version instead of skipping it.
            values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional writer progress callback.
            pinned_by: The releases that already read files of this release, with their pins. The
                writer publishes only if it reproduced each pinned file.

        Returns:
            The stored release.
        """

    @abstractmethod
    def put_release(self, from_directory: Path) -> DatasetRef:
        """Publish a release directory to this registry.

        The directory must hold ``manifest.json`` and every file that it lists. The method publishes
        only those files and does not change the directory.

        Args:
            from_directory: The release directory.

        Returns:
            The published release.

        Raises:
            TimeFFormatError: If ``manifest.json`` is missing or cannot be read.
            TimeNetFileError: If a listed file is missing or has another size. Nothing is published
                then.
        """

    @staticmethod
    def _read_release(directory: Path) -> Manifest:
        """Read the manifest of a release directory and check the size of each file that it lists.

        Args:
            directory: The release directory.

        Returns:
            The manifest of the release.

        Raises:
            TimeFFormatError: If ``manifest.json`` is missing or cannot be read.
            TimeNetFileError: If a listed file is missing or has another size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_file_sizes
        manifest_path = directory / MANIFEST_FILE
        try:
            manifest = Manifest.model_validate_json(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise TimeFFormatError(f"cannot read the release manifest {manifest_path}: {error}") from error
        reference = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(directory))
        check_file_sizes(version, manifest.files.all_files(), owner=str(reference))
        return manifest

    def exists(self, dataset_id: str, version: str) -> bool:
        """Return whether a committed version already exists in this registry.

        Args:
            dataset_id: The dataset id.
            version: The version string.

        Returns:
            ``True`` if the version has a committed manifest, else ``False``.
        """
        try:
            self.get_manifest(dataset_id, version)
        except TimeNetDatasetNotFoundError:
            return False
        return True
