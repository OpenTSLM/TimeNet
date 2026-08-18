"""Concurrent async HTTP downloads for connectors.

:func:`download_http` fetches one URL; :func:`download_http_many` fetches a list of :class:`Artifact`
with the number of parallel downloads bounded by ``max_concurrency``. Both stream responses to disk with
``aiofiles`` (no whole-file buffering, so multi-GB archives stay off the heap) and write atomically:
bytes land in a ``.part`` file that is renamed into place only on success, so an interrupted download
never leaves a truncated file a later ``skip_existing`` check would trust. Each artifact carries its own
headers and cookies, which merge over the batch-level ones, so one call can span hosts that need
different auth.
"""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

import aiofiles
import aiohttp

from timenet.errors import TimeFFormatError
from timenet_connectors.download.progress import DownloadProgress, report_progress


_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB streamed per write
_DEFAULT_MAX_CONCURRENCY = 8
# No total cap (archives can take a while), but fail if a connect or a single read stalls.
_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=60)


@dataclass(frozen=True)
class Artifact:
    """One download: a URL, its destination, and optional headers, cookies, and a SHA-256 to verify.

    The ``headers``, ``cookies``, and ``sha256`` apply to an HTTP download and are ignored for an
    ``s3://`` URL.
    """

    url: str
    dest: Path
    headers: Mapping[str, str] | None = None
    cookies: Mapping[str, str] | None = None
    sha256: str | None = None


async def download_http(  # noqa: PLR0913
    url: str,
    dest: str | Path,
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    skip_existing: bool = True,
    sha256: str | None = None,
) -> Path:
    """Download a single URL to ``dest``, streaming to disk and writing atomically.

    Creates ``dest``'s parent directories if they are missing. A set ``sha256`` that does not match the
    downloaded bytes raises :class:`~timenet.errors.TimeFFormatError` and leaves no file behind.

    Args:
        url: The source URL.
        dest: The destination file path.
        headers: Request headers (e.g. auth).
        cookies: Request cookies (e.g. auth).
        skip_existing: Return without downloading if ``dest`` already exists.
        sha256: Optional hex digest the downloaded bytes must match.

    Returns:
        The destination path.
    """
    artifact = Artifact(url, Path(dest), headers=headers, cookies=cookies, sha256=sha256)
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        await _download_one(session, artifact, skip_existing=skip_existing)
    return artifact.dest


async def download_http_many(
    artifacts: Iterable[Artifact],
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    skip_existing: bool = True,
) -> list[Path]:
    """Download many artifacts concurrently, bounded by ``max_concurrency``, failing fast.

    Batch ``headers`` and ``cookies`` apply to every request; each artifact's own headers and cookies
    merge over them. The first failure propagates (its message names the URL) and the destinations that
    were still in flight are left clean.

    Args:
        artifacts: The artifacts to download.
        headers: Headers applied to every request, under each artifact's own headers.
        cookies: Cookies applied to every request, under each artifact's own cookies.
        max_concurrency: Maximum number of downloads running at once.
        skip_existing: Skip any artifact whose destination already exists.

    Returns:
        The destination paths, in input order.
    """
    items = list(artifacts)
    semaphore = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession(
        headers=dict(headers) if headers else None, cookies=dict(cookies) if cookies else None, timeout=_TIMEOUT
    ) as session:

        async def bounded(artifact: Artifact) -> None:
            async with semaphore:
                await _download_one(session, artifact, skip_existing=skip_existing)

        await asyncio.gather(*(bounded(artifact) for artifact in items))
    return [artifact.dest for artifact in items]


async def _download_one(session: aiohttp.ClientSession, artifact: Artifact, *, skip_existing: bool) -> None:
    """Stream one artifact to its destination via a ``.part`` temp file, renamed on success.

    The artifact's headers and cookies are merged over the session's by aiohttp.

    Args:
        session: The shared client session.
        artifact: The artifact to download.
        skip_existing: Return early if the destination already exists.

    Raises:
        TimeFFormatError: If the artifact's ``sha256`` is set and does not match the downloaded bytes.
    """
    dest = artifact.dest
    if skip_existing and dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.part"
    headers = dict(artifact.headers) if artifact.headers else None
    cookies = dict(artifact.cookies) if artifact.cookies else None
    digest = hashlib.sha256() if artifact.sha256 is not None else None
    try:
        async with session.get(artifact.url, headers=headers, cookies=cookies) as response:
            response.raise_for_status()
            total = response.content_length
            downloaded = 0
            async with aiofiles.open(part, "wb") as handle:
                async for chunk in response.content.iter_chunked(_DOWNLOAD_CHUNK_BYTES):
                    await handle.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
                    downloaded += len(chunk)
                    report_progress(DownloadProgress(artifact.url, downloaded, total))
        if artifact.sha256 is not None and digest is not None and digest.hexdigest() != artifact.sha256.lower():
            raise TimeFFormatError(
                f"SHA-256 mismatch downloading {artifact.url!r}: "
                f"expected {artifact.sha256.lower()}, got {digest.hexdigest()}"
            )
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)  # a partial or mismatched download must not masquerade as complete
        raise
