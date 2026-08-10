"""A registry backed by a remote HTTP(S) service. Deferred; the contract is fixed here.

The intended REST contract (versioned) is: ``GET /v1/datasets``, ``GET /v1/datasets/{id}``,
``GET /v1/datasets/{id}/{version}/manifest``, and ``GET /v1/datasets/{id}/{version}/{relpath}``.

The ``{relpath}`` endpoint serves two access shapes. :meth:`open_file` streams it whole for a
sequential download, so plain ``200`` streaming suffices there. :meth:`open_version` reads it out of
order (Parquet footers, value slices), so the endpoint MUST also honor ``Range: bytes=…`` with a
``206`` + ``Content-Range`` and advertise ``Accept-Ranges: bytes`` (a ``HEAD`` returning
``Content-Length`` helps). :meth:`open_version` has no native pyarrow HTTP filesystem to hand back, so
it wraps the service as
``pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fsspec.filesystem("http", ...)))``, which makes
``fsspec`` and ``aiohttp`` real dependencies of that path.

Implementation lands in a later change; for now the methods raise ``NotImplementedError``.
"""

from collections.abc import Callable
from typing import BinaryIO

from timenet.dataset import TimeFDataset
from timenet.manifest import Manifest
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.writer import WriteProgressEvent


class RemoteRegistry(WritableRegistry):
    """Placeholder for an HTTP(S)-backed registry (not yet implemented)."""

    def __init__(self, base_url: str) -> None:
        """Store the base URL of the remote registry.

        Args:
            base_url: The service root, e.g. ``https://registry.timenet.ai``.
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

    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
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
