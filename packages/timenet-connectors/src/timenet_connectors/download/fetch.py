"""High-level, scheme-dispatching downloads for connectors.

:func:`fetch_files` downloads a list of :class:`~timenet_connectors.download.http.Artifact`, choosing the
backend from each URL's scheme so a connector never branches on ``s3://`` vs ``http(s)://`` itself. A
single file is just a one-element list. :func:`ensure_archive` builds on it to download a zip and extract
it once. S3 objects go through boto3 (:mod:`~timenet_connectors.download.s3`) and HTTP through httpx
(:mod:`~timenet_connectors.download.http`). Both are async so they compose with a connector's
``download_async``. The S3 branch is a plain blocking call, because boto3 already parallelizes a single
object's transfer. A list mixing schemes runs its S3 entries one at a time and its HTTP entries
concurrently. Progress flows through the ambient :mod:`~timenet_connectors.download.progress` sink, so
neither takes a progress argument.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

from timenet.errors import TimeFValidationError
from timenet_connectors.download.http import Artifact, download_http_many
from timenet_connectors.download.s3 import download_s3_object


_DEFAULT_MAX_CONCURRENCY = 8


__all__ = ["Artifact", "ensure_archive", "fetch_files"]


def _safe_filename(url: str) -> str:
    """Derive a cache filename from a URL's path, hashing the URL when it has no final segment.

    The function uses the last path segment, with the query and fragment stripped. A trailing slash or
    a ``/download`` suffix leaves no usable name. The function then falls back to a short hash of the
    URL. This keeps the cache path deterministic and avoids collisions across URLs.

    Args:
        url: The source URL.

    Returns:
        A filename safe to use inside the cache directory.
    """
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    if name:
        return name
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".download"


def _is_s3(url: str) -> bool:
    """Return whether ``url`` is an ``s3://`` URL."""
    return url.startswith("s3://")


def _is_http(url: str) -> bool:
    """Return whether ``url`` is an ``http://`` or ``https://`` URL."""
    return url.startswith(("http://", "https://"))


async def fetch_files(
    artifacts: Iterable[Artifact],
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    skip_existing: bool = True,
) -> list[Path]:
    """Download a list of artifacts, mixing ``s3://`` and ``http(s)://`` freely.

    HTTP entries download concurrently (bounded by ``max_concurrency``, sharing one connection pool). S3
    entries download one at a time, since boto3 blocks the event loop but parallelizes each transfer
    itself. Batch ``headers`` and ``cookies`` apply to every HTTP request, under each artifact's own
    (ignored for S3). The function validates every scheme up front, so an unsupported one fails before
    any download starts.

    Args:
        artifacts: The artifacts to download. A single file is a one-element list.
        headers: Headers applied to every HTTP request, under each artifact's own.
        cookies: Cookies applied to every HTTP request, under each artifact's own.
        max_concurrency: Maximum number of concurrent HTTP downloads.
        skip_existing: Skip any artifact whose destination already exists.

    Returns:
        The destination paths, in input order.

    Raises:
        TimeFValidationError: If any URL is neither an ``s3://`` nor an ``http(s)://`` URL.
    """
    items = [replace(artifact, dest=Path(artifact.dest)) for artifact in artifacts]
    for artifact in items:
        if not _is_s3(artifact.url) and not _is_http(artifact.url):
            raise TimeFValidationError(f"unsupported download URL scheme: {artifact.url!r}")
    for artifact in items:
        if _is_s3(artifact.url) and not (skip_existing and artifact.dest.exists()):
            download_s3_object(artifact.url, artifact.dest)
    http = [artifact for artifact in items if _is_http(artifact.url)]
    if http:
        await download_http_many(
            http, headers=headers, cookies=cookies, max_concurrency=max_concurrency, skip_existing=skip_existing
        )
    return [artifact.dest for artifact in items]


async def ensure_archive(  # noqa: PLR0913
    url: str,
    target: str | Path,
    *,
    filename: str | None = None,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    sha256: str | None = None,
) -> Path:
    """Download a zip archive (``s3://`` or ``http(s)://``) and extract it into ``target`` once.

    Idempotent: the function writes a marker file under ``target``, keyed by the archive URL, after a
    successful extraction. A re-run then reuses the extracted contents and skips the download.

    Args:
        url: The archive URL, ``s3://`` or ``http(s)://``.
        target: Directory into which the function downloads the archive and extracts its contents.
        filename: Overrides the cached archive name, for URLs whose path has no usable filename (a
            trailing slash or a ``/download`` suffix).
        headers: Request headers for an HTTP download (ignored for S3).
        cookies: Request cookies for an HTTP download (ignored for S3).
        sha256: Optional hex digest the downloaded archive must match (HTTP only).

    Returns:
        ``target`` (the extraction root).
    """
    target = Path(target)
    name = filename or _safe_filename(url)
    # Key the cache path and marker on the URL. This keeps two archives that share a basename from
    # colliding, which would silently skip the second download.
    key = hashlib.sha256(url.encode()).hexdigest()[:8]
    marker = target / f".{key}-{name}.extracted"
    if marker.exists():
        return target
    zip_path = target / f"{key}-{name}"
    await fetch_files([Artifact(url, zip_path, headers=headers, cookies=cookies, sha256=sha256)])
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    marker.touch()
    return target
