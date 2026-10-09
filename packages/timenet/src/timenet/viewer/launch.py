"""Lifecycle helpers for the local-only viewer server."""

from __future__ import annotations

from pathlib import Path
import socket
import sys
import time
import webbrowser

from timenet.client import TimeNet
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import open_registry
from timenet.registry.version import DatasetVersion
from timenet.viewer.app import create_app
from timenet.viewer.path_registry import PathRegistry
from timenet.viewer.security import new_token


_PROGRESS_INTERVAL_SECONDS = 0.2


def open_reader(dataset_id: str | None, version: str | None, registry: str | None, path: Path | None) -> TimeFReader:
    """Open one viewer root without ever invoking connector code.

    Returns:
        A pinned lazy reader.

    Raises:
        TimeFFormatError: If the direct path is invalid or needs unresolved parents.
    """
    if path is None:
        if dataset_id is None:
            raise TimeFFormatError("a dataset ID or --path is required")
        total = 0
        last_report = 0.0

        def progress(count: int) -> None:
            nonlocal total, last_report
            total += count
            now = time.monotonic()
            if now - last_report >= _PROGRESS_INTERVAL_SECONDS:
                print(f"Preparing pinned dataset: {total:,} bytes downloaded", file=sys.stderr, end="\r", flush=True)
                last_report = now

        opened = TimeNet(registry).open_reader(dataset_id, version, progress_cb=progress)
        if total:
            print(file=sys.stderr)
        return opened
    if version is not None:
        raise TimeFFormatError("--version cannot be used with --path")
    try:
        handle = DatasetVersion.open_local(path)
    except (OSError, ValueError) as exc:
        raise TimeFFormatError(f"cannot open TimeF version at {path}: {exc}") from exc
    if handle.manifest.metadata.parents:
        inferred = path.resolve()
        for _ in (*handle.manifest.dataset_id.split("/"), str(handle.manifest.metadata.dataset_version)):
            inferred = inferred.parent
        resolver = PathRegistry(handle, open_registry(registry or str(inferred)))
        return resolver.open_reader(handle.manifest.dataset_id, str(handle.manifest.metadata.dataset_version))
    return TimeFReader(handle)


def serve(reader: TimeFReader, *, port: int, open_browser: bool, registry: str = "unknown") -> None:
    """Bind an authenticated loopback server and run until interrupted."""
    import uvicorn  # noqa: PLC0415

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    actual_port = sock.getsockname()[1]
    token = new_token()
    url = f"http://127.0.0.1:{actual_port}/#token={token}"
    print(f"TimeNet viewer: {url}")
    if open_browser:
        webbrowser.open(url)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(reader, token=token, port=actual_port, registry=registry), access_log=False, proxy_headers=False
        )
    )
    try:
        server.run(sockets=[sock])
    finally:
        reader.close()
