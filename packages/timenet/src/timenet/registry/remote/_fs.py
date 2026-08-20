"""A pyarrow filesystem over a remote version's presigned URLs, via fsspec.

The reader drives ``DatasetVersion.filesystem``. For an on-demand remote load that filesystem is a
:class:`pyarrow.fs.PyFileSystem` wrapping :class:`_PresignedHTTPFileSystem`, an fsspec ``HTTPFileSystem``
that maps each version-relative path to a presigned object URL (resolved once, on first open) and
range-reads it directly. File sizes and existence come from the manifest, so no HEAD probe is issued: a
presigned GET URL rejects HEAD. A fully cached version is read from local disk instead, so this only
ever serves an uncached on-demand read.
"""

from typing import Any

from fsspec.implementations.http import HTTPFileSystem
from fsspec.spec import AbstractBufferedFile
import pyarrow.fs as pafs

from timenet.registry.remote._http import RegistryHttpClient


def remote_version_root(dataset_id: str, version: str) -> str:
    """Return the sentinel root prefix a remote handle is rooted at.

    Args:
        dataset_id: The ``org/name`` id.
        version: The version string.

    Returns:
        The ``<dataset_id>/<version>`` prefix the handler strips to recover a relpath.
    """
    return f"{dataset_id}/{version}"


class _PresignedHTTPFileSystem(HTTPFileSystem):
    """An fsspec HTTP filesystem mapping a version's relpaths to presigned object URLs.

    Paths are ``<dataset_id>/<version>/<relpath>`` (a ``DatasetVersion`` root plus a relpath). Each is
    resolved to a presigned URL once, on first open, and range-read directly from object storage. Sizes
    and existence come from the manifest, so neither a HEAD (which a presigned GET URL rejects) nor a
    listing round-trip is issued.
    """

    def __init__(
        self, http: RegistryHttpClient, dataset_id: str, version: str, sizes: dict[str, int], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._http = http
        self._dataset_id = dataset_id
        self._version = version
        self._prefix = f"{remote_version_root(dataset_id, version)}/"
        self._file_sizes = sizes
        self._urls: dict[str, str] = {}

    def _relpath(self, path: str) -> str:
        return path[len(self._prefix) :] if path.startswith(self._prefix) else path

    async def _info(self, url: str, **kwargs: Any) -> dict[str, Any]:
        relpath = self._relpath(url)
        if relpath in self._file_sizes:
            return {"name": url, "size": self._file_sizes[relpath], "type": "file"}
        return await super()._info(url, **kwargs)

    async def _exists(self, path: str, strict: bool = False, **kwargs: Any) -> bool:  # noqa: ARG002
        return self._relpath(path) in self._file_sizes

    def _open(  # noqa: PLR0913, PLR0917 - matches fsspec's HTTPFileSystem._open signature
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool | None = None,
        cache_type: str | None = None,
        cache_options: dict[str, Any] | None = None,
        size: int | None = None,  # noqa: ARG002 - the manifest size is authoritative
        **kwargs: Any,
    ) -> AbstractBufferedFile:
        relpath = self._relpath(path)
        url = self._urls.get(relpath)
        if url is None:
            url = self._http.resolve_presigned(self._dataset_id, self._version, relpath)
            self._urls[relpath] = url
        return super()._open(
            url,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options,
            size=self._file_sizes[relpath],
            **kwargs,
        )


def remote_version_filesystem(
    http: RegistryHttpClient, dataset_id: str, version: str, sizes: dict[str, int]
) -> pafs.PyFileSystem:
    """Build the pyarrow filesystem for a remote version's on-demand reads.

    Args:
        http: The registry HTTP client.
        dataset_id: The ``org/name`` id.
        version: The version string.
        sizes: A ``relpath -> size`` map from the manifest.

    Returns:
        A :class:`pyarrow.fs.PyFileSystem` that range-reads each file's presigned URL.
    """
    handler = _PresignedHTTPFileSystem(http, dataset_id, version, sizes)
    return pafs.PyFileSystem(pafs.FSSpecHandler(handler))
