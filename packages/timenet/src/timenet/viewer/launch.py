"""Lifecycle helpers for the local-only viewer server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import socket
import webbrowser

from fastapi import FastAPI

from timenet.config import settings
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import BaseRegistry, open_registry
from timenet.registry.version import DatasetVersion
from timenet.viewer.app import create_app, create_registry_app
from timenet.viewer.security import new_token


def open_reader(dataset_id: str | None, version: str | None, registry: str | None, path: Path | None) -> TimeFReader:
    """Open one viewer root without ever invoking connector code.

    Returns:
        A pinned lazy reader.

    Raises:
        TimeFFormatError: If the selection is invalid or the direct path cannot be opened.
    """
    if path is None:
        if dataset_id is None:
            raise TimeFFormatError("a dataset ID or --path is required")
        dataset_id, version = _resolve_ref(dataset_id, version)
        cfg = settings(registry=registry)
        backend = open_registry(cfg.registry or "timenet://", cache_dir=cfg.storage_dir)
        return TimeFReader(backend.open_version(dataset_id, version))
    if version is not None:
        raise TimeFFormatError("--version cannot be used with --path")
    try:
        handle = DatasetVersion.open_local(path)
    except (OSError, ValueError) as exc:
        raise TimeFFormatError(f"cannot open TimeF version at {path}: {exc}") from exc
    return TimeFReader(handle)


def _resolve_ref(dataset_id: str, version: str | None) -> tuple[str, str | None]:
    """Combine an ``org/name@version`` reference with an explicit version.

    Returns:
        The bare dataset id and resolved version.

    Raises:
        TimeFFormatError: If both arguments specify a version.
    """
    ref_id, ref_version = split_ref(dataset_id)
    if ref_version is not None and version is not None:
        raise TimeFFormatError(f"version given twice: '@{ref_version}' in the id and version={version!r}")
    return ref_id, ref_version if ref_version is not None else version


def serve(reader: TimeFReader, *, port: int, open_browser: bool, registry: str = "unknown") -> None:
    """Bind an authenticated loopback server and run until interrupted."""
    try:
        _serve(
            lambda token, bound_port: create_app(reader, token=token, port=bound_port, registry=registry),
            port=port,
            open_browser=open_browser,
        )
    finally:
        reader.close()


def open_viewer_registry(registry: str | None) -> tuple[BaseRegistry, str]:
    """Open the configured registry for catalog browsing.

    Returns:
        The registry and its display URI.
    """
    cfg = settings(registry=registry)
    source = cfg.registry or "timenet://"
    return open_registry(source, cache_dir=cfg.storage_dir), source


def serve_registry(registry: BaseRegistry, *, registry_name: str, port: int, open_browser: bool) -> None:
    """Bind an authenticated loopback server for a registry catalog."""
    _serve(
        lambda token, bound_port: create_registry_app(
            registry, token=token, port=bound_port, registry_name=registry_name
        ),
        port=port,
        open_browser=open_browser,
    )


def _serve(create: Callable[[str, int], FastAPI], *, port: int, open_browser: bool) -> None:
    """Bind and serve one authenticated application on a loopback socket."""
    import uvicorn  # noqa: PLC0415

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen()
        actual_port = sock.getsockname()[1]
        token = new_token()
        app = create(token, actual_port)
        url = f"http://127.0.0.1:{actual_port}/#token={token}"
        print(f"TimeNet viewer: {url}")
        if open_browser:
            webbrowser.open(url)
        uvicorn.Server(uvicorn.Config(app, access_log=False, proxy_headers=False)).run(sockets=[sock])
