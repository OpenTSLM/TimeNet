"""Define the writable-registry contract for a backend that curation publishes datasets into.

This class extends the read-only :class:`~timenet.registry.BaseRegistry` with one write primitive,
:meth:`WritableRegistry.store`. Backends differ only in where a compiled dataset lands, for example a
local directory, an S3 prefix, or a remote service. So ``store`` is the one abstract method, and
:meth:`exists` is shared.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from timenet.dataset import TimeFDataset
from timenet.errors import DatasetNotFoundError
from timenet.registry.base import BaseRegistry
from timenet.writer import WriteProgressEvent


class WritableRegistry(BaseRegistry, ABC):
    """A registry that curation can publish datasets into, not just read from."""

    @abstractmethod
    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Compile a dataset and publish it to this registry.

        If the dataset has no schema, this method derives one first. If a version is already
        committed, this method skips it unless ``force`` is set.

        Args:
            dataset: The populated dataset to store.
            force: Overwrite an already-committed version instead of skipping it.
            values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional writer progress callback.

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
        except DatasetNotFoundError:
            return False
        return True
