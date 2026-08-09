"""Reusable HTTP download helpers for connectors: atomic file downloads and archive caching.

Downloads stream to a sibling ``.part`` file and rename on success, so an interrupted transfer never
leaves a truncated file that a later run would treat as complete. ``s3://`` URLs are routed to the S3
helper, so a single :func:`ensure_file` or :func:`ensure_archive` call serves both PhysioNet's open
``physionet-open`` bucket and plain HTTP hosts. ``requests`` is imported lazily so base users who only
curate offline datasets don't need it (install the ``http`` extra); a missing library raises an
actionable error.
"""

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import zipfile

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.s3 import download_s3_object


_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB streamed per write when fetching a file
_DEFAULT_TIMEOUT_S = 60


def _requests() -> Any:
    """Import ``requests`` lazily, with an actionable error when the extra is missing.

    Returns:
        The imported ``requests`` module.

    Raises:
        ImportError: If ``requests`` (the ``http`` extra) is not installed.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("downloading over HTTP needs the http extra: pip install 'timenet-connectors[http]'") from exc
    return requests


def _filename_from_url(url: str) -> str:
    """Derive a cache filename from a URL's path, falling back to a stable hash.

    The last path segment is used. A trailing slash (no final segment) falls back to a short hash of
    the URL, so the cache path stays deterministic without colliding across URLs.

    Args:
        url: The source URL.

    Returns:
        A filename safe to use inside the cache directory.
    """
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    if name:
        return name
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".download"


def download_file(
    url: str,
    dest: Path,
    *,
    session: Any = None,
    headers: Mapping[str, str] | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Stream ``url`` to ``dest`` atomically, optionally verifying its SHA-256.

    Writes to a sibling ``.part`` file and renames on success, so an interrupted download never leaves
    a truncated file a later run would treat as complete. When ``expected_sha256`` is given and does
    not match, the partial file is removed and no ``dest`` is produced.

    Args:
        url: The source ``http(s)://`` URL.
        dest: The destination file path; parent directories are created.
        session: An optional ``requests.Session``-like object exposing ``get``; defaults to the
            ``requests`` module.
        headers: Optional request headers.
        expected_sha256: Optional lowercase hex digest the downloaded bytes must match.

    Raises:
        TimeFFormatError: If ``expected_sha256`` is set and does not match the downloaded bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    getter = session.get if session is not None else _requests().get
    digest = hashlib.sha256() if expected_sha256 is not None else None
    with getter(url, stream=True, timeout=_DEFAULT_TIMEOUT_S, headers=headers) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                if digest is not None:
                    digest.update(chunk)
    if digest is not None and expected_sha256 is not None and digest.hexdigest() != expected_sha256.lower():
        part.unlink(missing_ok=True)
        raise TimeFFormatError(
            f"SHA-256 mismatch downloading {url!r}: expected {expected_sha256.lower()}, got {digest.hexdigest()}"
        )
    part.replace(dest)


def ensure_file(
    url: str,
    dest: Path,
    *,
    session: Any = None,
    headers: Mapping[str, str] | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``dest`` once, reusing an existing file.

    Idempotent: if ``dest`` already exists it is returned unchanged. ``s3://`` URLs are routed to the
    S3 helper; any other URL is fetched with :func:`download_file`.

    Args:
        url: An ``s3://bucket/key`` object or an ``http(s)://`` URL.
        dest: The destination file path.
        session: An optional ``requests.Session``-like object, passed through for HTTP downloads.
        headers: Optional request headers, passed through for HTTP downloads.
        expected_sha256: Optional lowercase hex digest, checked for HTTP downloads.

    Returns:
        ``dest``.
    """
    if dest.exists():
        return dest
    if url.startswith("s3://"):
        download_s3_object(url, dest)
    else:
        download_file(url, dest, session=session, headers=headers, expected_sha256=expected_sha256)
    return dest


def ensure_archive(  # noqa: PLR0913
    url: str,
    cache_dir: Path,
    sentinel: str,
    *,
    filename: str | None = None,
    session: Any = None,
    headers: Mapping[str, str] | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download and extract a zip archive into ``cache_dir`` once.

    Idempotent on ``cache_dir / sentinel``: if the sentinel exists the download and extraction are
    skipped, so re-running a build reuses the cache. The sentinel is created after a successful
    extraction.

    Args:
        url: An ``s3://bucket/key`` object or an ``http(s)://`` URL for the zip archive.
        cache_dir: Directory the archive is downloaded and extracted into.
        sentinel: A path relative to ``cache_dir`` marking extraction as complete.
        filename: Overrides the cached archive name, for URLs whose path has no usable filename (a
            trailing slash or a ``/download`` suffix).
        session: An optional ``requests.Session``-like object, passed through for HTTP downloads.
        headers: Optional request headers, passed through for HTTP downloads.
        expected_sha256: Optional lowercase hex digest, checked for HTTP downloads.

    Returns:
        ``cache_dir`` (the extraction root).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / sentinel).exists():
        return cache_dir
    archive_path = cache_dir / (filename or _filename_from_url(url))
    ensure_file(url, archive_path, session=session, headers=headers, expected_sha256=expected_sha256)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(cache_dir)
    (cache_dir / sentinel).touch()
    return cache_dir
