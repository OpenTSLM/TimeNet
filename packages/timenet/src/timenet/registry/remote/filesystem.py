"""Expose manifest files to Arrow through fsspec's cached HTTP range reader."""

from typing import Any

from aiohttp import ClientError, ClientResponseError
from fsspec.implementations.http import HTTPFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
import httpx

from timenet.errors import TimeNetRegistryError
from timenet.registry.remote._http import RegistryHttpClient


_BLOCK_SIZE = 256 * 2**10


class RegistryFileSystem(AbstractFileSystem):
    """A read-only filesystem whose file names and sizes come from a release manifest.

    fsspec handles seeking, ranges, and read-ahead caching. Open files reuse signed URLs and
    refresh them after an authentication failure. Registry credentials never enter the object-store
    HTTP session. Pickling reconstructs clients in the new process.
    """

    protocol = "timenet-http"
    cachable = False

    def __init__(
        self,
        base_url: str,
        files: dict[str, int],
        *,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._files = files
        self._registry = RegistryHttpClient(base_url, token=token, transport=transport)
        self._ranges = HTTPFileSystem(encoded=True, skip_instance_cache=True)

    def info(self, path: str, **_kwargs: Any) -> dict[str, Any]:
        """Return manifest metadata without issuing HTTP requests.

        Returns:
            The file's name, size, and type.

        Raises:
            FileNotFoundError: If the manifest does not declare the file.
        """
        if path not in self._files:
            raise FileNotFoundError(path)
        return {"name": path, "size": self._files[path], "type": "file"}

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict | None = None,
        **_kwargs: Any,
    ) -> AbstractBufferedFile:
        """Open a seekable file with a read-ahead buffer.

        Returns:
            A binary file that downloads requested ranges plus a small read-ahead buffer.

        Raises:
            TimeNetRegistryError: If the caller requests write access.
        """
        if mode != "rb":
            raise TimeNetRegistryError("registry files are read-only")
        return _RegistryFile(
            self,
            path,
            size=self.info(path)["size"],
            block_size=block_size or _BLOCK_SIZE,
            autocommit=autocommit,
            cache_type="readahead",
            cache_options=cache_options,
        )


class _RegistryFile(AbstractBufferedFile):
    """Connect fsspec's buffer to signed, unauthenticated HTTP file reads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signed_url: str | None = None

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Read a block through the HTTP filesystem, using the manifest's known file size.

        Returns:
            The requested bytes.

        Raises:
            TimeNetRegistryError: If HTTP fails or the object server cannot serve byte ranges.
        """  # noqa: DOC501 - the outer handler wraps ClientResponseError
        try:
            try:
                return self._read_signed_range(start, end)
            except ClientResponseError as exc:
                if exc.status not in {401, 403}:
                    raise
                # A long-lived reader can outlast a signed URL. Refresh once, then surface failures.
                self._signed_url = None
                return self._read_signed_range(start, end)
        except (ClientError, OSError, ValueError) as exc:
            raise TimeNetRegistryError(f"cannot read bytes {start}:{end} of {self.path}: {exc}") from exc

    def _read_signed_range(self, start: int, end: int) -> bytes:
        """Read through fsspec using the current signed URL and known file size.

        Returns:
            The requested bytes.
        """
        if self._signed_url is None:
            dataset_org, dataset_name, version, relative_path = self.path.split("/", 3)
            self._signed_url = self.fs._registry.resolve_presigned(
                f"{dataset_org}/{dataset_name}", version, relative_path
            )
        with self.fs._ranges.open(self._signed_url, size=self.size, cache_type="none") as source:
            source.seek(start)
            return source.read(end - start)
