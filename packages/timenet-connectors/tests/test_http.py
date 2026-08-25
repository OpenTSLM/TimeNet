"""Tests for the async HTTP download helpers, driven against httpx.MockTransport."""

import asyncio
import hashlib

import httpx
import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.download.http import Artifact, download_http, download_http_many
from timenet_connectors.download.progress import progress_sink


def _patch_transport(monkeypatch, transport):
    """Force download_http/_many to build clients bound to a MockTransport."""
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("timenet_connectors.download.http.httpx.AsyncClient", factory)


def _transport(handler):
    return httpx.MockTransport(handler)


def _serve_bytes(payload: bytes):
    def handler(request):
        return httpx.Response(200, content=payload)

    return _transport(handler)


def test_downloads_a_single_file(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, _serve_bytes(b"hello world"))
    asyncio.run(download_http("http://host/data.bin", tmp_path / "data.bin"))
    assert (tmp_path / "data.bin").read_bytes() == b"hello world"


def test_leaves_no_part_file_after_success(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, _serve_bytes(b"payload"))
    asyncio.run(download_http("http://host/data.bin", tmp_path / "data.bin"))
    assert not (tmp_path / "data.bin.part").exists()


def test_skips_existing_destination(tmp_path, monkeypatch):
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"cached")
    hits = []

    def handler(request):
        hits.append(request.url.path)
        return httpx.Response(200, content=b"fresh")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(download_http("http://host/data.bin", dest, skip_existing=True))
    assert dest.read_bytes() == b"cached"
    assert hits == []


def test_sends_per_artifact_headers(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, content=b"ok")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(
        download_http_many(
            [
                Artifact("http://host/a", tmp_path / "a", headers={"Authorization": "token-a"}),
                Artifact("http://host/b", tmp_path / "b", headers={"Authorization": "token-b"}),
            ]
        )
    )
    assert set(seen) == {"token-a", "token-b"}


def test_sends_per_artifact_cookies(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, content=b"ok")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(
        download_http_many(
            [
                Artifact("http://host/a", tmp_path / "a", cookies={"session": "cookie-a"}),
                Artifact("http://host/b", tmp_path / "b", cookies={"session": "cookie-b"}),
            ]
        )
    )
    assert {"session=cookie-a", "session=cookie-b"} == set(seen)


def test_batch_headers_apply_to_all(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("x-api-key"))
        return httpx.Response(200, content=b"ok")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(
        download_http_many(
            [Artifact("http://host/a", tmp_path / "a"), Artifact("http://host/b", tmp_path / "b")],
            headers={"X-Api-Key": "shared"},
        )
    )
    assert seen == ["shared", "shared"]


def test_downloads_many_concurrently(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, _serve_bytes(b"x"))
    paths = asyncio.run(
        download_http_many([Artifact(f"http://host/f{i}", tmp_path / f"f{i}") for i in range(5)], max_concurrency=8)
    )
    assert len(paths) == 5
    assert all(p.read_bytes() == b"x" for p in paths)


def test_respects_max_concurrency(tmp_path, monkeypatch):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return httpx.Response(200, content=b"x")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(
        download_http_many([Artifact(f"http://host/f{i}", tmp_path / f"f{i}") for i in range(6)], max_concurrency=2)
    )
    assert peak == 2


def test_fails_fast_and_names_the_url(tmp_path, monkeypatch):
    def handler(request):
        if request.url.path == "/bad":
            return httpx.Response(404)
        return httpx.Response(200, content=b"ok")

    _patch_transport(monkeypatch, _transport(handler))
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(
            download_http_many(
                [Artifact("http://host/ok", tmp_path / "ok"), Artifact("http://host/bad", tmp_path / "bad")],
                max_concurrency=4,
            )
        )
    assert "/bad" in str(exc_info.value.request.url)
    assert not (tmp_path / "bad").exists()


def test_follows_redirects(tmp_path, monkeypatch):
    def handler(request):
        if request.url.path == "/download":
            return httpx.Response(302, headers={"Location": "http://host/final"})
        return httpx.Response(200, content=b"final payload")

    _patch_transport(monkeypatch, _transport(handler))
    asyncio.run(download_http("http://host/download", tmp_path / "data.bin"))
    assert (tmp_path / "data.bin").read_bytes() == b"final payload"


def test_download_http_reports_progress(tmp_path, monkeypatch):
    payload = b"x" * 3_000_000
    _patch_transport(monkeypatch, _serve_bytes(payload))
    events = []
    with progress_sink(events.append):
        asyncio.run(download_http("http://host/data.bin", tmp_path / "data.bin"))
    assert events
    assert all(event.url.endswith("/data.bin") for event in events)
    assert [event.downloaded for event in events] == sorted(event.downloaded for event in events)
    assert events[-1].downloaded == 3_000_000
    assert events[-1].total == 3_000_000


def test_download_http_accepts_a_matching_sha256(tmp_path, monkeypatch):
    payload = b"integrity matters"
    _patch_transport(monkeypatch, _serve_bytes(payload))
    asyncio.run(download_http("http://host/f", tmp_path / "f", sha256=hashlib.sha256(payload).hexdigest()))
    assert (tmp_path / "f").read_bytes() == payload


def test_download_http_rejects_a_sha256_mismatch(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, _serve_bytes(b"actual bytes"))
    with pytest.raises(TimeFFormatError, match="SHA-256"):
        asyncio.run(download_http("http://host/f", tmp_path / "f", sha256="00" * 32))
    assert not (tmp_path / "f").exists()
    assert not (tmp_path / "f.part").exists()


def test_cleans_up_part_file_on_mid_stream_failure(tmp_path, monkeypatch):
    def handler(request):
        async def broken():
            yield b"partial"
            raise httpx.RemoteProtocolError("peer closed", request=request)

        return httpx.Response(200, headers={"Content-Length": "1000"}, content=broken())

    _patch_transport(monkeypatch, _transport(handler))
    with pytest.raises(httpx.HTTPError):
        asyncio.run(download_http("http://host/broken.bin", tmp_path / "broken.bin"))
    assert not (tmp_path / "broken.bin").exists()
    assert not (tmp_path / "broken.bin.part").exists()
