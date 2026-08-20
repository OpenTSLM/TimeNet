"""An httpx.MockTransport fake of the registry, backed by a real on-disk TimeF version."""

from collections.abc import Iterator
import contextlib
import http.server
import json
from pathlib import Path
import threading
from typing import Any

import httpx

from timenet.manifest import Manifest


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
    schema = manifest.schema
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
    """
    version_dir = Path(version_dir)
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    dataset_id = manifest.metadata.dataset_id
    version = str(manifest.metadata.dataset_version)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        requests.append(request)
        if request.url.host == "blob.local":
            relpath = request.url.path.lstrip("/")
            data = (version_dir / relpath).read_bytes()
            rng = request.headers.get("range")
            if rng:
                start, _, end = rng.removeprefix("bytes=").partition("-")
                s = int(start)
                e = int(end) if end else len(data) - 1
                e = min(e, len(data) - 1)
                return httpx.Response(206, content=data[s : e + 1], headers={"Accept-Ranges": "bytes"})
            return httpx.Response(200, content=data, headers={"Accept-Ranges": "bytes"})

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


@contextlib.contextmanager
def range_server(version_dir: Path) -> Iterator[str]:
    """Serve ``version_dir`` over real HTTP with Range support; yield its base URL.

    The fsspec on-demand reader fetches presigned URLs over aiohttp, which a ``MockTransport`` cannot
    intercept, so its object bytes must come from a real socket. Point ``build_fake(blob_base=...)`` here.
    """
    directory = str(version_dir)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                body = (Path(directory) / self.path.lstrip("/")).read_bytes()
            except OSError:
                self.send_error(404)
                return
            total = len(body)
            rng = self.headers.get("Range")
            if rng:
                start, _, end = rng.removeprefix("bytes=").partition("-")
                s = int(start)
                e = min(int(end) if end else total - 1, total - 1)
                chunk = body[s : e + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {s}-{e}/{total}")
            else:
                chunk = body
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, format: str, *args: Any) -> None:  # silence per-request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


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
