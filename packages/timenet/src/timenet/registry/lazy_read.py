"""Shared opening logic for registries that read releases remotely."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
import tempfile
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.manifest import PayloadSource
from timenet.registry._paths import release_dir
from timenet.registry.dependencies import open_cached_or_remote
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types.references import DatasetRef, is_latest
from timenet.writer import TimeFWriter, WriteProgressEvent


class LazyReadRegistry(WritableRegistry, ABC):
    """A registry that reads remote files on demand and reuses complete cached releases.

    Backends differ only in how one remote release is bound to a filesystem, so
    :meth:`_open_remote_version` is the one method they implement for reading.
    """

    _staging_prefix: ClassVar[str] = "timenet-publish-"
    """The name prefix of the local folder that :meth:`put_dataset` writes into before it publishes."""

    def put_dataset(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> DatasetRef:
        """Write a dataset locally and publish it to this registry.

        Return the existing release's reference unless ``force`` is set.

        Args:
            dataset: The dataset to write.
            force: Publish even if the version already exists.
            values_backend: Storage format for values (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional callback for write progress.
            pinned_by: Releases that already read this release's files, with the pins that the new files must match.

        Returns:
            A reference to the stored release.
        """
        if dataset.schema is None:
            dataset.derive_schema()
        dataset_id = dataset.metadata.dataset_id
        version = str(dataset.metadata.dataset_version)
        reference = DatasetRef.parse(f"{dataset_id}@{version}")
        if not force and self.exists(dataset_id, version):
            return reference
        staging = Path(tempfile.mkdtemp(prefix=self._staging_prefix))
        try:
            # The writer stages locally but checks every payload source against this registry.
            with TimeFWriter(
                staging,
                dataset,
                values_backend=values_backend,
                progress_cb=progress_cb,
                registry=self,
                pinned_by=pinned_by,
            ) as writer:
                writer.write()
            return self.put_release(release_dir(staging, reference))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Open a release and its payload sources, then read remote files on demand.

        Opening reads only manifests. Each payload source must be in this registry and match the
        files the release pins.

        Args:
            dataset_id: The dataset ID.
            version: The release version, or ``None`` for the latest.

        Returns:
            A handle that prefers complete cached releases and otherwise uses remote range reads.
        """
        if is_latest(version):
            version = str(self.get_manifest(dataset_id, version).metadata.dataset_version)
        return open_cached_or_remote(
            self.local_root, DatasetRef.parse(f"{dataset_id}@{version}"), self._open_remote_version, searched=str(self)
        )

    @abstractmethod
    def _open_remote_version(self, reference: DatasetRef) -> DatasetVersion:
        """Bind one remote release to a range-capable filesystem, without its payload sources.

        Returns:
            A handle whose payload sources :meth:`open_version` attaches.
        """
