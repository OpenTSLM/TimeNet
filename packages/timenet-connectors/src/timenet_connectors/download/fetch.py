"""High-level, scheme-dispatching downloads for connectors.

:func:`download_files` downloads a list of :class:`~timenet_connectors.download.http.Artifact`, choosing the
backend from each URL's scheme so a connector never branches on ``s3://`` vs ``http(s)://`` itself. A
single file is a one-element list. :func:`ensure_archive` builds on it to download a zip archive,
extract it once, and delete the archive. S3 objects go through boto3
(:mod:`~timenet_connectors.download.s3`) and HTTP through httpx
(:mod:`~timenet_connectors.download.http`). Both are async so they compose with a connector's
``download_async``. The S3 branch is a plain blocking call, because boto3 already parallelizes a single
object's transfer. A list mixing schemes runs its S3 entries one at a time and its HTTP entries
concurrently. Progress flows through the ambient :mod:`~timenet_connectors.download.progress` sink, so
neither takes a progress argument. :func:`find_dir_containing` locates a file inside an extracted
tree, whose layout differs from archive to archive.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

from timenet.errors import TimeFValidationError
from timenet_connectors.download.http import Artifact, download_http_many, is_cached
from timenet_connectors.download.s3 import download_s3_object


_DEFAULT_MAX_CONCURRENCY = 8


__all__ = ["Artifact", "download_files", "ensure_archive", "find_dir_containing"]


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


async def download_files(
    artifacts: Iterable[Artifact],
    *,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    skip_existing: bool = True,
) -> list[Path]:
    """Download files from S3 and HTTP URLs.

    S3 downloads block the event loop and run one at a time, while HTTP downloads run concurrently.
    The function checks all URL schemes before starting any download.

    Args:
        artifacts: The files to download, with their URLs and destination paths.
        headers: Default HTTP headers that each artifact's headers override.
        cookies: Default HTTP cookies that each artifact's cookies override.
        max_concurrency: Maximum number of concurrent HTTP downloads.
        skip_existing: Skip existing files if they match ``sha256`` or have no expected hash.

    Returns:
        The destination paths, in input order.

    Raises:
        TimeFValidationError: If a URL uses a scheme other than ``s3://``, ``http://``, or ``https://``.
    """
    items = [replace(artifact, dest=Path(artifact.dest)) for artifact in artifacts]
    for artifact in items:
        if not _is_s3(artifact.url) and not _is_http(artifact.url):
            raise TimeFValidationError(f"unsupported download URL scheme: {artifact.url!r}")
    for artifact in items:
        if _is_s3(artifact.url) and not (skip_existing and is_cached(artifact)):
            download_s3_object(artifact.url, artifact.dest, sha256=artifact.sha256)
    http = [artifact for artifact in items if _is_http(artifact.url)]
    if http:
        await download_http_many(
            http, headers=headers, cookies=cookies, max_concurrency=max_concurrency, skip_existing=skip_existing
        )
    return [artifact.dest for artifact in items]


async def ensure_archive(  # ruff: ignore[too-many-arguments]
    url: str,
    target: str | Path,
    *,
    filename: str | None = None,
    headers: Mapping[str, str] | None = None,
    cookies: Mapping[str, str] | None = None,
    sha256: str | None = None,
) -> Path:
    """Download a zip archive and extract it into ``target``.

    A marker file lets later calls reuse an extraction with the same URL, filename, and SHA-256 hash.
    The function deletes the archive after extraction succeeds and keeps it if extraction fails.

    Args:
        url: The archive URL, ``s3://`` or ``http(s)://``.
        target: The directory for the archive and its extracted contents.
        filename: An optional archive filename to use instead of the name derived from the URL.
        headers: Request headers for HTTP downloads only.
        cookies: Request cookies for HTTP downloads only.
        sha256: An optional SHA-256 hash in hexadecimal that the archive must match.

    Returns:
        The extraction directory, ``target``.
    """
    target = Path(target)
    name = filename or _safe_filename(url)
    # Key the cache path and marker on the URL. This keeps two archives that share a basename from
    # colliding, which would silently skip the second download. A pin joins the key, so a changed
    # pin downloads again instead of reusing an extraction that matched the old one.
    source = url if sha256 is None else f"{url}#sha256={sha256.lower()}"
    key = hashlib.sha256(source.encode()).hexdigest()[:8]
    marker = target / f".{key}-{name}.extracted"
    zip_path = target / f"{key}-{name}"
    if marker.exists():
        # A run that stopped before the unlink below can leave a stale archive. Delete it.
        zip_path.unlink(missing_ok=True)
        return target
    await download_files([Artifact(url, zip_path, headers=headers, cookies=cookies, sha256=sha256)])
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    # Marker first: the reverse order costs the whole download if the run stops between the two.
    marker.touch()
    zip_path.unlink(missing_ok=True)
    return target


def find_dir_containing(root: Path, relative: str) -> Path:
    """Find the directory under ``root`` that contains ``relative`` (archives extract nested).

    Args:
        root: The extraction root to search.
        relative: A file name expected inside the wanted directory.

    Returns:
        The parent directory of the first match.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    for match in root.rglob(relative):
        return match.parent
    raise FileNotFoundError(f"{relative!r} not found under {root}")
