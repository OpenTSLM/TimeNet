"""Open a registry and serve the viewer browser application."""

import secrets
import socket
from urllib.parse import urlencode
import webbrowser

import typer
import uvicorn

from timenet.config import settings
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.registry import BaseRegistry, open_registry
from timenet.types.references import split_ref
from timenet.viewer.app import create_app


_LOOPBACK = "127.0.0.1"
# Bytes of randomness in the per-launch access token.
_TOKEN_BYTES = 48


def open_viewer(reference: str | None = None, registry: str | None = None) -> tuple[BaseRegistry, Manifest | None]:
    """Open a registry and optionally resolve one pinned manifest.

    Args:
        reference: Dataset ID, optionally pinned with ``@version``, or ``None`` for the catalog.
        registry: Registry URI or local directory, or ``None`` for configuration.

    Returns:
        The registry and an optional committed manifest.
    """
    cfg = settings(registry=registry)
    backend = open_registry(cfg.registry or "timenet://", cache_dir=cfg.storage_dir)
    if reference is None:
        return backend, None
    dataset_id, version = split_ref(reference)
    return backend, backend.get_manifest(dataset_id, version)


def serve(
    registry: BaseRegistry,
    *,
    initial_manifest: Manifest | None = None,
    port: int = 0,
    open_browser: bool = True,
    quiet: bool = False,
) -> None:
    """Bind the authenticated viewer to loopback until the process is interrupted.

    Args:
        registry: Registry to browse.
        initial_manifest: Pinned manifest for a direct dataset launch, if any.
        port: TCP port on 127.0.0.1, with zero choosing an available port.
        open_browser: Whether to open the printed URL locally.
        quiet: Hide the server's status logs, keeping warnings and errors. The URL still prints.

    Raises:
        TimeFValidationError: If the port cannot be bound.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((_LOOPBACK, port))
        except OSError as exc:
            raise TimeFValidationError(f"cannot bind the viewer to {_LOOPBACK}:{port}: {exc}") from exc
        sock.listen()
        actual_port = sock.getsockname()[1]
        origin = f"http://{_LOOPBACK}:{actual_port}"
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        app = create_app(registry, token=token, origin=origin, initial_manifest=initial_manifest)
        query = ""
        if initial_manifest is not None:
            query = "?" + urlencode(
                {
                    "dataset": initial_manifest.dataset_id,
                    "version": str(initial_manifest.metadata.dataset_version),
                }
            )
        url = f"{origin}/{query}#token={token}"
        typer.echo(url)
        if open_browser:
            webbrowser.open(url)
        config = uvicorn.Config(app, access_log=False, proxy_headers=False, log_level="warning" if quiet else "info")
        uvicorn.Server(config).run(sockets=[sock])
