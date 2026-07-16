"""A registry backed by a remote HTTP(S) service. Deferred; the contract is fixed here.

The intended REST contract (versioned) is: ``GET /v1/datasets``, ``GET /v1/datasets/{id}``,
``GET /v1/datasets/{id}/{version}/manifest``, and ``GET /v1/datasets/{id}/{version}/{relpath}``.
Implementation lands in a later change; for now the methods raise ``NotImplementedError``.
"""

from collections.abc import Callable
from typing import BinaryIO

from timenet.dataset import TimeFDataset
from timenet.manifest import Manifest
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.writer import WriteProgressEvent


class RemoteRegistry(WritableRegistry):
    """Placeholder for an HTTP(S)-backed registry (not yet implemented)."""

    def __init__(self, base_url: str) -> None:
        """Store the base URL of the remote registry.

        Args:
            base_url: The service root, e.g. ``https://registry.timenet.io``.
        """
        self._base_url = base_url

    def list_datasets(self) -> list[DatasetMetadata]:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteRegistry is not yet implemented")

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteRegistry is not yet implemented")

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteRegistry is not yet implemented")

    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("RemoteRegistry is not yet implemented")
