"""Shared opening logic for registries that read releases remotely."""

from abc import ABC, abstractmethod

from timenet.registry.dependencies import resolve_read_dependency_graph
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types.references import DatasetRef, is_latest


class LazyReadRegistry(WritableRegistry, ABC):
    """A registry that reads remote files on demand and reuses complete cached releases.

    Backends differ only in how one remote release is bound to a filesystem, so
    :meth:`_open_remote_version` is the one method they implement for reading.
    """

    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Resolve physical source manifests, then read remote files on demand.

        Args:
            dataset_id: The dataset ID.
            version: The release version, or ``None`` for the latest.

        Returns:
            A handle that prefers complete cached releases and otherwise uses remote range reads.
        """
        if is_latest(version):
            version = str(self.get_manifest(dataset_id, version).metadata.dataset_version)
        return resolve_read_dependency_graph(
            self.local_root, DatasetRef.parse(f"{dataset_id}@{version}"), self._open_remote_version
        )

    @abstractmethod
    def _open_remote_version(self, reference: DatasetRef) -> DatasetVersion:
        """Bind one remote release to a range-capable filesystem, without its payload sources.

        Returns:
            A handle whose payload sources the dependency resolver fills in.
        """
