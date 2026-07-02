"""The writable-registry contract: a backend that curation can publish datasets into.

Extends the read-only :class:`~timenet.registry.BaseRegistry` with a single write primitive,
:meth:`WritableRegistry.store`. Backends differ only in *where* a compiled dataset lands (a local
directory, an S3 prefix, a remote service), so ``store`` is the one abstract method and :meth:`exists`
is shared.
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
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Compile a dataset and publish it to this registry.

        Derives the dataset's schema first if it has none. An already-committed version is skipped
        unless ``force`` is set.

        Args:
            dataset: The populated dataset to store.
            force: Overwrite an already-committed version instead of skipping it.
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
