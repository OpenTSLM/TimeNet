"""Open a registry and serve the viewer browser application."""

import secrets
import socket
from urllib.parse import urlencode
import webbrowser

import uvicorn

from timenet.config import settings
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.refs import split_ref
from timenet.registry import BaseRegistry, open_registry
from timenet.viewer.app import create_app
from timenet.viewer.security import normalize_origin


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


def serve(  # noqa: PLR0913 - launch options map to CLI flags
    registry: BaseRegistry,
    *,
    initial_manifest: Manifest | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    public_origin: str | None = None,
    open_browser: bool = True,
) -> None:
    """Bind the authenticated viewer until the process is interrupted.

    Args:
        registry: Registry to browse.
        initial_manifest: Pinned manifest for a direct dataset launch, if any.
        host: IPv4 address to bind, defaulting to loopback.
        port: TCP port, with zero choosing an available port.
        public_origin: HTTPS origin of a remote reverse proxy, when used.
        open_browser: Whether to open the printed URL locally.

    Raises:
        TimeFValidationError: If a non-loopback bind lacks a public HTTPS origin.
    """
    if host != "127.0.0.1" and public_origin is None:
        raise TimeFValidationError("a non-loopback viewer bind requires --public-origin")
    if public_origin is not None:
        public_origin = normalize_origin(public_origin)
        if not public_origin.startswith("https://"):
            raise TimeFValidationError("--public-origin must be an HTTPS origin")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen()
        actual_port = sock.getsockname()[1]
        origin = public_origin or f"http://127.0.0.1:{actual_port}"
        token = secrets.token_urlsafe(48)
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
        print(url, flush=True)
        if open_browser:
            webbrowser.open(url)
        config = uvicorn.Config(app, access_log=False, proxy_headers=False)
        uvicorn.Server(config).run(sockets=[sock])
