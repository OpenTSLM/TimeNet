"""Shared opening logic for registries that read releases remotely."""

from abc import ABC, abstractmethod

from timenet.registry.dependencies import open_cached_or_remote
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types.references import DatasetRef, is_latest


class LazyReadRegistry(WritableRegistry, ABC):
    """A registry that reads remote files on demand and reuses complete cached releases.

    Backends differ only in how one remote release is bound to a filesystem, so
    :meth:`_open_remote_version` is the one method they implement for reading.
    """

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
