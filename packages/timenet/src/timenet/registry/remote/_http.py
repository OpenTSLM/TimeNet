"""The HTTP transport for :class:`~timenet.registry.RemoteRegistry`.

Owns how the SDK talks to the hosted registry. It prefixes ``/api/v1`` and attaches the bearer token
and a ``timenet/<version>`` User-Agent to API calls. It resolves presigned download URLs without
leaking the token to object storage. It maps HTTP status codes onto TimeNet errors. A sync
:class:`httpx.Client` serves the control plane. :meth:`new_async_client` hands out an
:class:`httpx.AsyncClient` sharing the same configuration for the parallel downloader.
"""

from collections.abc import Mapping
import importlib.metadata
from typing import Any, BinaryIO, cast

import httpx

from timenet.errors import TimeNetDatasetNotFoundError, TimeNetRegistryError


API_PREFIX = "/api/v1"
_STREAM_CHUNK_BYTES = 1 << 20


def timenet_user_agent() -> str:
    """Return the ``timenet/<version>`` User-Agent used for backend tracing.

    Returns:
        The User-Agent string; ``timenet/0+unknown`` if the package metadata is unavailable.
    """
    try:
        version = importlib.metadata.version("timenet")
    except importlib.metadata.PackageNotFoundError:
        version = "0+unknown"
    return f"timenet/{version}"


def _raise_for_status(response: httpx.Response) -> None:
    """Map an error response onto a TimeNet error.

    Args:
        response: The response to check.

    Raises:
        TimeNetDatasetNotFoundError: On 404.
        TimeNetRegistryError: On any other 4xx/5xx, with the status and a body snippet.
    """
    if not response.is_error:
        return
    body = response.text[:200]
    if response.status_code == httpx.codes.NOT_FOUND:
        raise TimeNetDatasetNotFoundError(f"registry 404 for {response.request.url}: {body}")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        retry = response.headers.get("retry-after", "?")
        raise TimeNetRegistryError(f"registry rate-limited (429), retry after {retry}s: {body}")
    raise TimeNetRegistryError(f"registry request to {response.request.url} failed ({response.status_code}): {body}")


def _download_path(dataset_id: str, version: str, relpath: str) -> str:
    """Return the API path for a version-relative file's presigned download.

    Args:
        dataset_id: The ``org/name`` id.
        version: The version string.
        relpath: The version-relative file path.

    Returns:
        The ``/api/v1/datasets/.../download/...`` path.
    """
    return f"{API_PREFIX}/datasets/{dataset_id}/{version}/download/{relpath}"


def _redirect_target(response: httpx.Response, path: str) -> str:
    """Return a download response's redirect ``Location``, or raise if it did not redirect.

    Args:
        response: The download response fetched with ``follow_redirects=False``.
        path: The requested path, for the error message.

    Returns:
        The presigned URL from the ``Location`` header.

    Raises:
        TimeNetRegistryError: If the response is not a redirect.
    """
    if response.is_redirect:
        return response.headers["location"]
    _raise_for_status(response)
    raise TimeNetRegistryError(f"expected a redirect from {path}, got {response.status_code}")


class RegistryHttpClient:
    """A thin httpx wrapper for the registry API and its presigned object URLs."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Open a client against a registry service root.

        Args:
            base_url: The service root, e.g. ``https://registry.dev.timenet.ai``. A trailing slash is
                stripped. The ``/api/v1`` prefix is added per request.
            token: Bearer token, or ``None`` for anonymous.
            timeout: Per-request timeout in seconds.
            transport: An httpx transport for testing (e.g. ``httpx.MockTransport``). ``None`` uses the
                default network transport.
        """
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._transport = transport
        self._user_agent = timenet_user_agent()
        # follow_redirects=True so a presigned object-store GET that 3xx-redirects (an S3 region
        # redirect, a CDN in front of storage) is followed to the bytes; resolve_presigned overrides
        # this per-call with follow_redirects=False to capture the API's own download redirect.
        self._client = httpx.Client(
            base_url=self._base_url,
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
        )

    def api_headers(self) -> dict[str, str]:
        """Return the headers for a registry API call: the User-Agent plus, when set, the token.

        The ``timenet/<version>`` User-Agent identifies the SDK to the backend for tracing and the
        bearer token authenticates the call. Both stay on API calls only; presigned object-store
        requests send neither.

        Returns:
            A dict with ``User-Agent`` and, when a token is set, ``Authorization``.
        """
        headers = {"User-Agent": self._user_agent}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET an API path and return the parsed JSON body.

        Args:
            path: A path under ``/api/v1`` (e.g. ``/datasets``).
            params: Optional query parameters.

        Returns:
            The decoded JSON.
        """
        response = self._client.get(f"{API_PREFIX}{path}", params=params, headers=self.api_headers())
        _raise_for_status(response)
        return response.json()

    def get_text(self, path: str) -> str:
        """GET an API path and return the raw text body (used for ``manifest.json``).

        Args:
            path: A path under ``/api/v1``.

        Returns:
            The response text.
        """
        response = self._client.get(f"{API_PREFIX}{path}", headers=self.api_headers())
        _raise_for_status(response)
        return response.text

    def post(self, path: str, *, content: bytes | None = None, json: Any = None) -> httpx.Response:
        """POST to an API path with a raw body or a JSON body.

        Args:
            path: A path under ``/api/v1``.
            content: Raw request body bytes (e.g. the manifest), mutually exclusive with ``json``.
            json: A JSON-serializable body.

        Returns:
            The raw response (already status-checked).
        """
        response = self._client.post(f"{API_PREFIX}{path}", content=content, json=json, headers=self.api_headers())
        _raise_for_status(response)
        return response

    def resolve_presigned(self, dataset_id: str, version: str, relpath: str) -> str:
        """Resolve a file's presigned URL via the download redirect, without following it.

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            relpath: The version-relative file path.

        Returns:
            The presigned URL from the redirect ``Location``.

        Raises:
            TimeNetRegistryError: If the download endpoint does not redirect.
        """  # noqa: DOC502 - raised by _redirect_target
        path = _download_path(dataset_id, version, relpath)
        response = self._client.get(path, headers=self.api_headers(), follow_redirects=False)
        return _redirect_target(response, path)

    def stream_to(self, url: str, sink: BinaryIO) -> None:
        """Stream a presigned URL's bytes into a binary sink (no auth header).

        Args:
            url: The presigned URL.
            sink: A writable binary file object.
        """
        with self._client.stream("GET", url) as response:
            if response.is_error:
                response.read()  # a streaming response has no body until read; _raise_for_status needs it
                _raise_for_status(response)
            for chunk in response.iter_bytes(_STREAM_CHUNK_BYTES):
                sink.write(chunk)

    def put(self, url: str, *, content: bytes, headers: Mapping[str, str]) -> None:
        """PUT bytes to a presigned upload URL (no auth header).

        Args:
            url: The presigned PUT URL.
            content: The request body.
            headers: Headers the presign requires (e.g. the checksum binding).
        """
        response = self._client.put(url, content=content, headers=dict(headers))
        _raise_for_status(response)

    def new_async_client(self) -> httpx.AsyncClient:
        """Build an async client sharing this client's base URL and transport.

        The downloader only fetches presigned object URLs, which carry neither the token nor the
        User-Agent, so no default headers are attached.

        Returns:
            An :class:`httpx.AsyncClient` for the parallel downloader to manage in its own context.
        """
        # An injected transport (e.g. httpx.MockTransport) implements both the sync and async
        # transport protocols; only the async view is valid for an AsyncClient.
        transport = cast("httpx.AsyncBaseTransport | None", self._transport)
        return httpx.AsyncClient(
            base_url=self._base_url,
            transport=transport,
            timeout=self._timeout,
            follow_redirects=True,  # follow a presigned object-store 3xx redirect to the bytes
        )
