"""Tests for the async HTTP download helpers, driven against a real local aiohttp server."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import hashlib
from pathlib import Path
import socket

import aiohttp
from aiohttp import web
import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.download.http import Artifact, download_http, download_http_many


_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def serving(*, routes: dict[str, _Handler] | None = None, catch_all: _Handler | None = None):
    """Serve GET routes on an ephemeral localhost port; yields the base URL."""
    app = web.Application()
    for path, handler in (routes or {}).items():
        app.router.add_get(path, handler)
    if catch_all is not None:
        app.router.add_get("/{tail:.*}", catch_all)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _bytes(payload: bytes) -> _Handler:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=payload)

    return handler


async def _not_found(request: web.Request) -> web.Response:
    raise web.HTTPNotFound


def test_downloads_a_single_file(tmp_path):
    async def scenario() -> None:
        async with serving(routes={"/data.bin": _bytes(b"hello world")}) as base:
            await download_http(f"{base}/data.bin", tmp_path / "data.bin")

    asyncio.run(scenario())
    assert (tmp_path / "data.bin").read_bytes() == b"hello world"


def test_leaves_no_part_file_after_success(tmp_path):
    async def scenario() -> None:
        async with serving(routes={"/data.bin": _bytes(b"payload")}) as base:
            await download_http(f"{base}/data.bin", tmp_path / "data.bin")

    asyncio.run(scenario())
    assert not (tmp_path / "data.bin.part").exists()  # renamed into place, temp cleaned up


def test_skips_existing_destination(tmp_path):
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"cached")
    hits: list[str] = []

    async def scenario() -> None:
        async def handler(request: web.Request) -> web.Response:
            hits.append(request.path)
            return web.Response(body=b"fresh")

        async with serving(routes={"/data.bin": handler}) as base:
            await download_http(f"{base}/data.bin", dest, skip_existing=True)

    asyncio.run(scenario())
    assert dest.read_bytes() == b"cached"  # untouched
    assert hits == []  # server never contacted


def test_sends_per_artifact_headers(tmp_path):
    seen: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(request.headers.get("Authorization"))
        return web.Response(body=b"ok")

    async def scenario() -> None:
        async with serving(routes={"/a": handler, "/b": handler}) as base:
            await download_http_many(
                [
                    Artifact(f"{base}/a", tmp_path / "a", headers={"Authorization": "token-a"}),
                    Artifact(f"{base}/b", tmp_path / "b", headers={"Authorization": "token-b"}),
                ]
            )

    asyncio.run(scenario())
    assert set(seen) == {"token-a", "token-b"}


def test_sends_per_artifact_cookies(tmp_path):
    seen: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(request.cookies.get("session"))
        return web.Response(body=b"ok")

    async def scenario() -> None:
        async with serving(routes={"/a": handler, "/b": handler}) as base:
            await download_http_many(
                [
                    Artifact(f"{base}/a", tmp_path / "a", cookies={"session": "cookie-a"}),
                    Artifact(f"{base}/b", tmp_path / "b", cookies={"session": "cookie-b"}),
                ]
            )

    asyncio.run(scenario())
    assert set(seen) == {"cookie-a", "cookie-b"}


def test_batch_headers_apply_to_all(tmp_path):
    seen: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(request.headers.get("X-Api-Key"))
        return web.Response(body=b"ok")

    async def scenario() -> None:
        async with serving(catch_all=handler) as base:
            await download_http_many(
                [Artifact(f"{base}/a", tmp_path / "a"), Artifact(f"{base}/b", tmp_path / "b")],
                headers={"X-Api-Key": "shared"},
            )

    asyncio.run(scenario())
    assert seen == ["shared", "shared"]


def test_downloads_many_concurrently(tmp_path):
    async def scenario() -> list[Path]:
        async with serving(catch_all=_bytes(b"x")) as base:
            artifacts = [Artifact(f"{base}/f{i}", tmp_path / f"f{i}") for i in range(5)]
            return await download_http_many(artifacts, max_concurrency=8)

    paths = asyncio.run(scenario())
    assert len(paths) == 5
    assert all(p.read_bytes() == b"x" for p in paths)


def test_respects_max_concurrency(tmp_path):
    active = 0
    peak = 0

    async def slow(request: web.Request) -> web.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return web.Response(body=b"x")

    async def scenario() -> None:
        async with serving(catch_all=slow) as base:
            artifacts = [Artifact(f"{base}/f{i}", tmp_path / f"f{i}") for i in range(6)]
            await download_http_many(artifacts, max_concurrency=2)

    asyncio.run(scenario())
    assert peak == 2  # never exceeded the bound, and actually reached it


def test_fails_fast_and_names_the_url(tmp_path):
    async def scenario() -> None:
        async with serving(routes={"/ok": _bytes(b"ok"), "/bad": _not_found}) as base:
            await download_http_many(
                [
                    Artifact(f"{base}/ok", tmp_path / "ok"),
                    Artifact(f"{base}/bad", tmp_path / "bad"),
                ],
                max_concurrency=4,
            )

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        asyncio.run(scenario())
    assert "/bad" in str(exc_info.value)
    assert not (tmp_path / "bad").exists()  # the failed download left nothing behind


def test_download_http_accepts_a_matching_sha256(tmp_path):
    payload = b"integrity matters"
    digest = hashlib.sha256(payload).hexdigest()

    async def scenario() -> None:
        async with serving(routes={"/f": _bytes(payload)}) as base:
            await download_http(f"{base}/f", tmp_path / "f", sha256=digest)

    asyncio.run(scenario())
    assert (tmp_path / "f").read_bytes() == payload


def test_download_http_rejects_a_sha256_mismatch(tmp_path):
    async def scenario() -> None:
        async with serving(routes={"/f": _bytes(b"actual bytes")}) as base:
            await download_http(f"{base}/f", tmp_path / "f", sha256="00" * 32)

    with pytest.raises(TimeFFormatError, match="SHA-256"):
        asyncio.run(scenario())
    assert not (tmp_path / "f").exists()  # no file produced on mismatch
    assert not (tmp_path / "f.part").exists()  # partial removed


def test_cleans_up_part_file_on_mid_stream_failure(tmp_path):
    async def drop(request: web.Request) -> web.StreamResponse:
        # Advertise more bytes than we send, then abort: the client raises mid-stream.
        response = web.StreamResponse(headers={"Content-Length": "1000"})
        await response.prepare(request)
        await response.write(b"partial")
        assert request.transport is not None
        request.transport.close()
        return response

    async def scenario() -> None:
        async with serving(routes={"/broken.bin": drop}) as base:
            await download_http(f"{base}/broken.bin", tmp_path / "broken.bin")

    with pytest.raises(aiohttp.ClientError):
        asyncio.run(scenario())
    assert not (tmp_path / "broken.bin").exists()
    assert not (tmp_path / "broken.bin.part").exists()  # partial temp file removed
