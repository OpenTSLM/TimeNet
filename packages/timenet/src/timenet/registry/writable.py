"""Define the writable-registry contract for a backend that build publishes datasets into.

This class extends the read-only :class:`~timenet.registry.BaseRegistry` with one write primitive,
:meth:`WritableRegistry.store`. Backends differ only in where a compiled dataset lands, for example a
local directory, an S3 prefix, or a remote service. So ``store`` is the one abstract method, and
:meth:`exists` is shared.
"""

from abc import ABC, abstractmethod

from timenet.control_plane import DeclarativeDataset
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.registry.base import BaseRegistry


class WritableRegistry(BaseRegistry, ABC):
    """A registry that build can publish datasets into, not just read from."""

    @abstractmethod
    def store(self, dataset: DeclarativeDataset, *, force: bool = False) -> str:
        """Compile a dataset and publish it to this registry.

        If a version is already committed, this method skips it unless ``force`` is set.

        Args:
            dataset: The populated dataset to store. Its metadata names the id and version.
            force: Overwrite an already-committed version instead of skipping it.

        Returns:
            The stored version string.
        """

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
