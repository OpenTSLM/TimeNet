"""A registry backed by an S3 (or S3-compatible) bucket. Deferred; the contract is fixed here.

An ``s3://<bucket>/<prefix>`` root holds the same ``<dataset_id>/<version>/`` layout as a
:class:`~timenet.registry.LocalRegistry`. Reads stream objects; :meth:`store` compiles a dataset to a
local staging directory and uploads it. Implementation lands in a later change; for now the methods
raise ``NotImplementedError``.
"""

from collections.abc import Callable
from typing import BinaryIO

from timenet.dataset import TimeFDataset
from timenet.manifest import Manifest
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.writer import WriteProgressEvent


class S3Registry(WritableRegistry):
    """Placeholder for an S3-backed registry (not yet implemented)."""

    def __init__(self, uri: str) -> None:
        """Store the S3 URI root of the registry.

        Args:
            uri: The bucket/prefix root, e.g. ``s3://my-bucket/registry``.
        """
        self._uri = uri

    def list_datasets(self) -> list[DatasetMetadata]:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("S3Registry is not yet implemented")

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("S3Registry is not yet implemented")

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("S3Registry is not yet implemented")

    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("S3Registry is not yet implemented")
