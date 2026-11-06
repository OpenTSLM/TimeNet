"""Shared opening logic for registries that read releases remotely."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
import tempfile
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest, PayloadSource
from timenet.registry._paths import release_dir
from timenet.registry.dependencies import open_cached_or_remote_release
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types.references import DatasetRef
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

        If the version already exists, return its reference. A published version never changes.

        Args:
            dataset: The dataset to write.
            force: Must be false, because this registry never writes a published version again.
            values_backend: Storage format for values (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional callback for write progress.
            pinned_by: Releases that already read this release's files, with the pins that the new files must match.

        Returns:
            A reference to the stored release.

        Raises:
            TimeFValidationError: If ``force`` is set.
        """
        if force:
            raise TimeFValidationError(
                f"force=True needs a local registry: {self} never writes a published version again"
            )
        if dataset.schema is None:
            dataset.derive_schema()
        dataset_id = dataset.metadata.dataset_id
        version = str(dataset.metadata.dataset_version)
        reference = DatasetRef.parse(f"{dataset_id}@{version}")
        if self.exists(dataset_id, version):
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

    def open_release(self, reference: DatasetRef, *, manifest: Manifest | None = None) -> DatasetVersion:
        """Open a cached or remote release without attaching its payload sources.

        Args:
            reference: The dataset ID and exact version to open.
            manifest: A manifest already read from this registry, or ``None`` to fetch it.

        Returns:
            A handle to a complete copy of the same release in :attr:`local_root`, or to remote files.
        """
        if manifest is None:
            manifest = self.get_manifest(reference.dataset_id, str(reference.version))
        return open_cached_or_remote_release(
            self.local_root, reference, manifest, lambda ref: self._open_remote_version(ref, manifest=manifest)
        )

    @abstractmethod
    def _open_remote_version(self, reference: DatasetRef, *, manifest: Manifest | None = None) -> DatasetVersion:
        """Open a remote release on a filesystem that serves range reads.

        Args:
            reference: The dataset ID and exact version to open.
            manifest: A manifest already read from this registry, or ``None`` to fetch it.

        Returns:
            A handle to the remote release without attached payload sources.
        """
