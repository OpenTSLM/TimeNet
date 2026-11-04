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
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetFileError
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
    ) -> str:
        """Compile a dataset and publish it to this registry.

        If the dataset has no schema, this method derives one first. If a version is already
        committed, this method skips it unless ``force`` is set.

        Args:
            dataset: The populated dataset to store.
            force: Overwrite an already-committed version instead of skipping it.
            values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional writer progress callback.
            pinned_by: The releases that already read files of this release, with their pins. The
                writer publishes only if it reproduced each pinned file.

        Returns:
            The stored version string.
        """

    @abstractmethod
    def put_release(self, from_directory: Path, manifest: Manifest) -> None:
        """Publish a complete local release directory to this registry as it is.

        This method only reads ``from_directory``. It does not change, move or delete a file in it.
        Before it publishes anything, this method makes sure that ``from_directory`` holds a
        ``manifest.json`` equal to ``manifest`` and every file that the manifest declares, with its
        recorded size. It publishes only those files and ``manifest.json``. It does not check the
        payload sources of the release, and it does not compare with a copy that this registry
        already holds. The caller decides whether this registry may receive the version.

        Args:
            from_directory: The local release directory to read from, with ``manifest.json`` and
                every declared file.
            manifest: The manifest of the release.

        Raises:
            TimeNetFileError: If ``manifest.json`` is missing from ``from_directory`` or differs from
                ``manifest``, or a declared file is missing or has another size than recorded.
                Nothing is published then.
        """

    @staticmethod
    def _check_release_files(directory: Path, manifest: Manifest) -> None:
        """Make sure that a local directory holds the manifest and every file that it declares.

        Args:
            directory: The local release directory.
            manifest: The manifest of the release.

        Raises:
            TimeNetFileError: If ``manifest.json`` is missing or differs from ``manifest``, or a
                declared file is missing or has another size than recorded.
        """
        reference = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        owner = str(reference)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise TimeNetFileError(reference, owner, MANIFEST_FILE, "the release manifest", None)
        try:
            on_disk = Manifest.model_validate_json(manifest_path.read_text())
        except ValueError:
            on_disk = None
        if on_disk != manifest:
            raise TimeNetFileError(
                reference, owner, MANIFEST_FILE, "the manifest that was passed", "another or unreadable manifest"
            )
        version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(directory))
        check_file_sizes(version, manifest.files.all_files(), owner=owner)

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
