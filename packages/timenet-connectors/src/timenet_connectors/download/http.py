"""Concurrent async HTTP downloads for connectors.

:func:`download_http` fetches one URL. :func:`download_http_many` fetches a list of :class:`Artifact`
with the number of parallel downloads bounded by ``max_concurrency``. Both stream responses to disk with
``aiofiles`` (no whole-file buffering, so multi-GB archives stay off the heap) and write atomically.
The bytes land in a ``.part`` file, and a rename moves it into place only on success. So an interrupted
download never leaves a truncated file that a later ``skip_existing`` check would trust. Each artifact
carries its own headers and cookies, which merge over the batch-level ones, so one call can span hosts
that need different auth.
"""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

import aiofiles
import httpx

from timenet.errors import TimeFFormatError
from timenet_connectors.download.progress import DownloadProgress, report_progress


_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB streamed per write
_DEFAULT_MAX_CONCURRENCY = 8
# No overall cap (archives can take a while), but fail if a connect or a single read stalls.
_TIMEOUT = httpx.Timeout(None, connect=60.0, read=60.0)


@dataclass(frozen=True)
class Artifact:
    """A file to download, with its URL, destination, and optional request settings and SHA-256 hash.

    The headers and cookies apply only to HTTP downloads.
    The SHA-256 hash applies to both HTTP and S3 downloads.
    """

    url: str
    dest: Path
    headers: Mapping[str, str] | None = None
    cookies: Mapping[str, str] | None = None
    sha256: str | None = None


def is_cached(artifact: Artifact) -> bool:
    """Check whether the destination already contains the requested file.

    An existing destination counts as cached if no SHA-256 hash is set.
    If a hash is set, the file must match it.

    Args:
        artifact: The download details to check.

    Returns:
        Whether the destination exists and matches the hash, if one is set.
    """
    if not artifact.dest.exists():
        return False
    if artifact.sha256 is None:
        return True
    with artifact.dest.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == artifact.sha256.lower()


async def download_http(  # ruff: ignore[too-many-arguments]
    url: str,
    dest: str | Path,
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    skip_existing: bool = True,
    sha256: str | None = None,
) -> Path:
    """Download an HTTP URL to ``dest``.

    The function creates missing parent directories and replaces the destination only after success.
    A SHA-256 mismatch raises :class:`~timenet.errors.TimeFFormatError` and preserves any existing destination.

    Args:
        url: The source URL.
        dest: The destination file path.
        headers: Request headers.
        cookies: Request cookies.
        skip_existing: Skip an existing file if it matches ``sha256`` or no expected hash is set.
        sha256: An optional SHA-256 hash in hexadecimal that the downloaded file must match.

    Returns:
        The destination path.
    """
    artifact = Artifact(url, Path(dest), headers=headers, cookies=cookies, sha256=sha256)
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        await _download_one(client, artifact, batch_headers=None, batch_cookies=None, skip_existing=skip_existing)
    return artifact.dest


async def download_http_many(
    artifacts: Iterable[Artifact],
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    skip_existing: bool = True,
) -> list[Path]:
    """Download multiple HTTP files concurrently.

    The first failure stops the batch and raises with the failing URL in its message.

    Args:
        artifacts: The files to download, with their URLs and destination paths.
        headers: Default request headers that each artifact's headers override.
        cookies: Default request cookies that each artifact's cookies override.
        max_concurrency: Maximum number of downloads running at once.
        skip_existing: Skip existing files if they match ``sha256`` or have no expected hash.

    Returns:
        The destination paths, in input order.
    """
    items = list(artifacts)
    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:

        async def bounded(artifact: Artifact) -> None:
            async with semaphore:
                await _download_one(
                    client, artifact, batch_headers=headers, batch_cookies=cookies, skip_existing=skip_existing
                )

        await asyncio.gather(*(bounded(artifact) for artifact in items))
    return [artifact.dest for artifact in items]


async def _download_one(
    client: httpx.AsyncClient,
    artifact: Artifact,
    *,
    batch_headers: Mapping[str, str] | None,
    batch_cookies: Mapping[str, str] | None,
    skip_existing: bool,
) -> None:
    """Download one HTTP file to its destination.

    The function creates missing parent directories and replaces the destination only after success.

    Args:
        client: The HTTP client for the request.
        artifact: The file to download, with its URL and destination path.
        batch_headers: Default request headers that the artifact's headers override.
        batch_cookies: Default request cookies that the artifact's cookies override.
        skip_existing: Skip an existing file if it matches ``sha256`` or no expected hash is set.

    Raises:
        TimeFFormatError: If the downloaded file does not match the supplied SHA-256 hash.
    """
    dest = artifact.dest
    if skip_existing and is_cached(artifact):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.part"
    headers = {**(batch_headers or {}), **(artifact.headers or {})} or None
    cookies = {**(batch_cookies or {}), **(artifact.cookies or {})} or None
    digest = hashlib.sha256() if artifact.sha256 is not None else None
    try:
        async with client.stream("GET", artifact.url, headers=headers, cookies=cookies) as response:
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            total = int(response.headers["content-length"]) if "content-length" in response.headers else None
            async with aiofiles.open(part, "wb") as handle:
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    await handle.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
                    report_progress(DownloadProgress(artifact.url, response.num_bytes_downloaded, total))
        if artifact.sha256 is not None and digest is not None and digest.hexdigest() != artifact.sha256.lower():
            raise TimeFFormatError(
                f"SHA-256 mismatch downloading {artifact.url!r}: "
                f"expected {artifact.sha256.lower()}, got {digest.hexdigest()}"
            )
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)  # a partial or mismatched download must not masquerade as complete
        raise
