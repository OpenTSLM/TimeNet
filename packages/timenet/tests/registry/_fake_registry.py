"""An httpx.MockTransport fake of the registry, backed by a real on-disk TimeF version."""

from collections.abc import Callable, Generator
import contextlib
from dataclasses import dataclass, field
from functools import partial
import http.server
from itertools import count
import json
from pathlib import Path
import threading
from typing import Any

import httpx

from timenet.errors import TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.registry import LocalRegistry


def _summary(manifest: Manifest) -> dict:
    md = manifest.metadata
    return {
        "dataset_id": md.dataset_id,
        "version": str(md.dataset_version),
        "name": md.name,
        "description": md.description,
        "license": md.license.value,
        "domains": [d.value for d in md.domains],
        "tags": list(md.tags),
    }


def _detail(manifest: Manifest) -> dict:
    schema = manifest.dataset_schema
    return {
        **_summary(manifest),
        "task_types": [t.task_type for t in schema.tasks],
        "spec_types": [s.spec_type for s in schema.time_series_specs],
    }


def build_fake(version_dir: Path, *, token: str | None = None, blob_base: str = "http://blob.local"):
    """Serve one on-disk version through a MockTransport; return (transport, recorded_requests).

    The version_dir is ``<...>/<org>/<name>/<version>``; its ``manifest.json`` drives every response.
    Presigned downloads redirect to ``<blob_base>/<relpath>``. The default ``http://blob.local`` host is
    served by this same MockTransport with Range support (the httpx full-download path); point
    ``blob_base`` at a :func:`range_server` for the fsspec on-demand path, which reads over aiohttp and
    cannot see a MockTransport.

    The default blob host never contacts the network. It reads existing files from the temporary
    fixture release; this download handler does not create files.
    """
    version_dir = Path(version_dir)
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    dataset_id = manifest.metadata.dataset_id
    version = str(manifest.metadata.dataset_version)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        requests.append(request)
        if request.url.host == "blob.local":
            return read_file(version_dir, request)

        if token is not None and request.headers.get("authorization") != f"Bearer {token}":
            return httpx.Response(401, json={"detail": "invalid API token"})

        path = request.url.path
        base = f"/api/v1/datasets/{dataset_id}"
        if path == "/api/v1/datasets":
            return httpx.Response(200, json={"datasets": [_summary(manifest)]})
        if path == base:
            return httpx.Response(200, json=_detail(manifest))
        if path == f"{base}/{version}/manifest":
            return httpx.Response(
                200,
                content=(version_dir / "manifest.json").read_bytes(),
                headers={"content-type": "application/json"},
            )
        if path == f"{base}/{version}/files":
            files = [{"path": p.path, "size": p.size} for p in manifest.files.all_files()]
            return httpx.Response(200, json={"dataset_id": dataset_id, "version": version, "files": files})
        if path.startswith(f"{base}/{version}/download/"):
            relpath = path.removeprefix(f"{base}/{version}/download/")
            return httpx.Response(307, headers={"location": f"{blob_base}/{relpath}"})
        return httpx.Response(404, json={"detail": f"no route: {path}"})

    return httpx.MockTransport(handler), requests


@dataclass
class HttpServer:
    """A running test server and the responses it has sent."""

    url: str
    responses: list[httpx.Response] = field(default_factory=list)


@contextlib.contextmanager
def _http_server(serve: Callable[[httpx.Request], httpx.Response]) -> Generator[HttpServer]:
    """Run a request handler over a real socket and record its responses."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = httpx.Request("GET", running.url + self.path, headers=dict(self.headers))
            response = serve(request)
            response.request = request
            running.responses.append(response)
            self.send_response(response.status_code)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, format: str, *args: Any) -> None:  # silence per-request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    running = HttpServer(f"http://127.0.0.1:{server.server_address[1]}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield running
    finally:
        server.shutdown()
        server.server_close()


def read_file(root: Path, request: httpx.Request) -> httpx.Response:
    """Read one file, honoring its HTTP Range header."""
    try:
        data = (root / request.url.path.lstrip("/")).read_bytes()
    except FileNotFoundError:
        return httpx.Response(404)
    rng = request.headers.get("range")
    if not rng:
        return httpx.Response(200, content=data)
    start, _, end = rng.removeprefix("bytes=").partition("-")
    first = int(start)
    last = min(int(end) if end else len(data) - 1, len(data) - 1)
    return httpx.Response(
        206,
        content=data[first : last + 1],
        headers={"Content-Range": f"bytes {first}-{last}/{len(data)}", "Accept-Ranges": "bytes"},
    )


@contextlib.contextmanager
def range_server(version_dir: Path) -> Generator[str]:
    """Serve files over real HTTP with byte ranges, without registry API routes."""
    with _http_server(partial(read_file, version_dir)) as server:
        yield server.url


@contextlib.contextmanager
def registry_server(
    version_dir: Path,
    *,
    serve_file: Callable[[httpx.Request], httpx.Response] | None = None,
) -> Generator[HttpServer]:
    """Serve the registry API and files, with an optional custom download handler."""
    download = serve_file or partial(read_file, version_dir)
    grants = count()

    def serve(request: httpx.Request) -> httpx.Response:
        if not request.url.path.startswith("/api/v1/"):
            return download(request)
        response = transport.handle_request(request)
        if response.is_redirect:
            response.headers["location"] += f"?grant={next(grants)}"
        return response

    with _http_server(serve) as server:
        transport, _ = build_fake(version_dir, blob_base=server.url)
        yield server


def build_publish_fake(*, token: str = "tok_rw"):  # noqa: S107
    """A MockTransport that accepts a full publish flow into an in-memory store.

    Returns (transport, state) where state has: ``store`` (relpath -> bytes), ``published`` (list of
    files from /publish), and ``finalized`` (bool).
    """
    state: dict[str, Any] = {"store": {}, "published": [], "finalized": False, "manifest": None}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "blob.local":  # a PUT upload
            relpath = request.url.path.removeprefix("/upload/")
            state["store"][relpath] = request.content
            return httpx.Response(200)
        if request.headers.get("authorization") != f"Bearer {token}":
            return httpx.Response(403, json={"detail": "write scope required"})
        if path.endswith("/publish"):
            state["manifest"] = json.loads(request.content)
            files = [p["path"] for group in state["manifest"]["files"].values() for p in group]
            state["published"] = files
            return httpx.Response(200, json={"files": sorted(files)})
        if path.endswith("/publish/upload-url"):
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "url": f"http://blob.local/upload/{body['path']}",
                    "headers": {"x-amz-checksum-sha256": "AAAA"},
                    "expires_in": 900,
                },
            )
        if path.endswith("/finalize"):
            state["finalized"] = True
            md = state["manifest"]["metadata"]
            return httpx.Response(
                200,
                json={
                    "dataset_id": state["manifest"]["dataset_id"],
                    "version": md["dataset_version"],
                    "name": md["name"],
                    "description": md["description"],
                    "license": md["license"],
                    "domains": md.get("domains", []),
                    "tags": md.get("tags", []),
                    "task_types": [],
                    "spec_types": [],
                },
            )
        return httpx.Response(404, json={"detail": path})

    return httpx.MockTransport(handler), state


def _blob(request: httpx.Request, root: Path) -> httpx.Response:
    """Serve the presigned blob host: PUT writes an upload into ``root``, GET reads a file with Range."""
    relpath = request.url.path.lstrip("/")
    if request.method == "PUT":
        target = root / relpath.removeprefix("upload/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(request.content)
        return httpx.Response(200)
    return read_file(root, request)


def _service_read(local: LocalRegistry, root: Path, parts: list[str]) -> httpx.Response:  # noqa: PLR0911
    """Serve the anonymous read endpoints from a LocalRegistry over ``root``."""
    if not parts:  # GET /api/v1/datasets
        summaries = [_summary(local.get_manifest(md.dataset_id)) for md in local.list_datasets()]
        return httpx.Response(200, json={"datasets": summaries})
    dataset_id = "/".join(parts[:2])
    if len(parts) == 2:  # GET /api/v1/datasets/<org>/<name>
        try:
            return httpx.Response(200, json=_detail(local.get_manifest(dataset_id)))
        except TimeNetDatasetNotFoundError as exc:
            return httpx.Response(404, json={"detail": str(exc)})
    version, action = parts[2], parts[3] if len(parts) > 3 else ""
    if action == "manifest":
        path = root / dataset_id / version / "manifest.json"
        if not path.exists():
            return httpx.Response(404, json={"detail": f"no manifest for {dataset_id}@{version}"})
        return httpx.Response(200, content=path.read_bytes(), headers={"content-type": "application/json"})
    if action == "download":
        relpath = "/".join(parts[4:])
        return httpx.Response(307, headers={"location": f"http://blob.local/{dataset_id}/{version}/{relpath}"})
    return httpx.Response(404, json={"detail": "/".join(parts)})


def _service_write(root: Path, manifests: dict, parts: list[str], request: httpx.Request) -> httpx.Response:
    """Serve the token-gated publish endpoints, landing new versions back under ``root``."""
    dataset_id, version, action = "/".join(parts[:2]), parts[2], "/".join(parts[3:])
    if action == "publish":
        manifests[dataset_id, version] = request.content
        files = json.loads(request.content)["files"]
        return httpx.Response(200, json={"files": [p["path"] for group in files.values() for p in group]})
    if action == "publish/upload-url":
        relpath = json.loads(request.content)["path"]
        url = f"http://blob.local/upload/{dataset_id}/{version}/{relpath}"
        return httpx.Response(200, json={"url": url, "headers": {}})
    if action == "finalize":
        target = root / dataset_id / version / "manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(manifests[dataset_id, version])
        return httpx.Response(200, json={"ok": True})
    return httpx.Response(404, json={"detail": action})


def build_service_fake(root: Path, *, token: str | None = None):
    """Emulate the whole registry API over a real on-disk tree at ``root``; return (transport, requests).

    Reads (list, detail, manifest, presigned download) are anonymous and served by a
    :class:`~timenet.registry.LocalRegistry` rooted at ``root``. Writes (publish, upload-url, finalize)
    require ``token`` and land new ``<org>/<name>/<version>/`` versions back under ``root``. Presigned
    downloads redirect to a ``blob.local`` host this same transport serves (GET with Range), and uploads
    PUT their bytes to that host. This is enough to drive a :class:`RemoteRegistry` end to end, publish
    included, without a network.
    """
    root = Path(root)
    local = LocalRegistry(root)
    manifests: dict[tuple[str, str], bytes] = {}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "blob.local":
            return _blob(request, root)
        parts = [p for p in request.url.path.removeprefix("/api/v1/datasets").split("/") if p]
        if request.method == "GET":
            return _service_read(local, root, parts)
        if token is not None and request.headers.get("authorization") != f"Bearer {token}":
            return httpx.Response(403, json={"detail": "write scope required"})
        return _service_write(root, manifests, parts, request)

    return httpx.MockTransport(handler), requests
