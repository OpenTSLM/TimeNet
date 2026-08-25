"""Parallel, atomic download of a whole dataset version into a local directory.

Resolves each file's presigned URL and streams it concurrently into a staging directory, bounded by a
semaphore. A single rename then swaps the staging directory into place, so an interrupted download
never replaces a good copy. The download is async under the hood (httpx ``AsyncClient``).
:func:`download_version_files` is the synchronous entry point. It runs the coroutine even from inside a
running event loop.
"""

import asyncio
from collections.abc import Coroutine
import hashlib
from pathlib import Path
import shutil
import threading
from typing import Any
import uuid

import httpx

from timenet.errors import TimeFFormatError
from timenet.manifest import Manifest
from timenet.manifest.files import FilePart
from timenet.registry._paths import safe_version_path
from timenet.registry.remote._http import RegistryHttpClient, _download_path, _raise_for_status, _redirect_target


_CHUNK_BYTES = 1 << 20


def download_version_files(  # noqa: PLR0913
    http: RegistryHttpClient,
    manifest: Manifest,
    dataset_id: str,
    version: str,
    dest_dir: Path,
    *,
    force: bool = False,
    max_concurrency: int = 8,
) -> Path:
    """Download every file of a version into ``dest_dir`` atomically.

    Args:
        http: The registry HTTP client.
        manifest: The version's parsed manifest (its file list drives the download).
        dataset_id: The ``org/name`` id.
        version: The version string.
        dest_dir: The target ``<...>/<id>/<version>`` directory.
        force: Re-download even if ``dest_dir/manifest.json`` already exists.
        max_concurrency: Maximum concurrent file downloads.

    Returns:
        ``dest_dir``.
    """
    dest_dir = Path(dest_dir)
    if (dest_dir / "manifest.json").exists() and not force:
        return dest_dir
    files = list(manifest.files.all_files())
    parent = dest_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    # A hard kill (SIGKILL/power loss) skips the finally below, so its staging dir lingers. Sweep any
    # stale <version>.tmp-* sibling before staging a fresh copy (mirrors the local download path).
    for stale in parent.glob(f"{dest_dir.name}.tmp-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    staging = parent / f"{dest_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        _run(_download_all(http, dataset_id, version, files, staging, max_concurrency))
        (staging / "manifest.json").write_text(manifest.to_json())  # already parsed; no extra round trip
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        staging.replace(dest_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest_dir


async def _download_all(  # noqa: PLR0913, PLR0917
    http: RegistryHttpClient, dataset_id: str, version: str, files: list[FilePart], staging: Path, max_concurrency: int
) -> None:
    """Download all files concurrently into the staging directory.

    Args:
        http: The registry HTTP client.
        dataset_id: The ``org/name`` id.
        version: The version string.
        files: The manifest file descriptors (path + checksum) to download.
        staging: The staging directory to write into.
        max_concurrency: Maximum concurrent file downloads.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    headers = http.api_headers()
    async with http.new_async_client() as client:

        async def one(part: FilePart) -> None:
            async with semaphore:
                # Reject a manifest path that would escape the staging dir before writing anything.
                dest = safe_version_path(staging, part.path)
                url = await _resolve(client, headers, dataset_id, version, part.path)
                await _stream_one(client, url, dest, expected_checksum=part.checksum)

        await asyncio.gather(*(one(part) for part in files))


async def _resolve(client: httpx.AsyncClient, headers: dict, dataset_id: str, version: str, relpath: str) -> str:
    """Resolve a file's presigned URL via the download redirect (async).

    Args:
        client: The async HTTP client.
        headers: The API headers (User-Agent plus, when set, the bearer token).
        dataset_id: The ``org/name`` id.
        version: The version string.
        relpath: The version-relative file path.

    Returns:
        The presigned URL from the redirect ``Location``.

    Raises:
        TimeNetRegistryError: If the download endpoint does not redirect.
    """  # noqa: DOC502 - raised by _redirect_target
    path = _download_path(dataset_id, version, relpath)
    response = await client.get(path, headers=headers, follow_redirects=False)
    return _redirect_target(response, path)


async def _stream_one(client: httpx.AsyncClient, url: str, dest: Path, *, expected_checksum: str | None = None) -> None:
    """Stream one presigned URL to ``dest`` via a ``.part`` temp file, renamed on success.

    Args:
        client: The async HTTP client.
        url: The presigned URL.
        dest: The target file path.
        expected_checksum: The manifest's ``sha256:<hex>`` digest to verify the bytes against.

    Raises:
        TimeFFormatError: If ``expected_checksum`` is set and the downloaded bytes do not match it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.part"
    digest = hashlib.sha256() if expected_checksum else None
    try:
        async with client.stream("GET", url) as response:
            if response.is_error:
                await response.aread()
                _raise_for_status(response)
            with part.open("wb") as handle:
                async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                    handle.write(chunk)  # buffered write: a fast memcpy, the OS flushes lazily
                    if digest is not None:
                        digest.update(chunk)
        if digest is not None and f"sha256:{digest.hexdigest()}" != expected_checksum:
            raise TimeFFormatError(f"checksum mismatch for {dest.name!r}: expected {expected_checksum}")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion, even when called from within a running event loop.

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[Any] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # re-raised on the calling thread
            error.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]
