"""Tests for the high-level, scheme-dispatching download helpers."""

import asyncio
import io
import zipfile

import pytest

from timenet_connectors.download import fetch
from timenet_connectors.download.fetch import Artifact, ensure_archive, fetch_files


def _zip_bytes(name: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


# ---- fetch_files ------------------------------------------------------------------------------


def test_fetch_files_routes_s3_sequential_and_http_batched(monkeypatch, tmp_path):
    s3_urls = []
    http_urls = []

    def _fake_s3(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"s3")
        s3_urls.append(url)

    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(b"http")
            http_urls.append(artifact.url)
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_s3_object", _fake_s3)
    monkeypatch.setattr(fetch, "download_http_many", _fake_many)

    items = [
        Artifact("s3://b/a", tmp_path / "a"),
        Artifact("https://h/b", tmp_path / "b"),
        Artifact("s3://b/c", tmp_path / "c"),
    ]
    result = asyncio.run(fetch_files(items))
    assert result == [tmp_path / "a", tmp_path / "b", tmp_path / "c"]  # input order preserved
    assert s3_urls == ["s3://b/a", "s3://b/c"]  # s3 handled one at a time
    assert http_urls == ["https://h/b"]  # http handled as one batch


def test_fetch_files_passes_batch_and_per_artifact_auth(monkeypatch, tmp_path):
    captured = {}

    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        captured["artifacts"] = list(artifacts)
        captured["headers"] = headers
        captured["cookies"] = cookies
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_http_many", _fake_many)
    items = [Artifact("https://h/a", tmp_path / "a", headers={"Authorization": "tok"}, cookies={"s": "1"})]
    asyncio.run(fetch_files(items, headers={"X-App": "t"}, cookies={"batch": "c"}))

    assert captured["headers"] == {"X-App": "t"}  # batch-level forwarded to the session
    assert captured["cookies"] == {"batch": "c"}
    assert captured["artifacts"][0].headers == {"Authorization": "tok"}  # per-artifact preserved
    assert captured["artifacts"][0].cookies == {"s": "1"}


def test_fetch_files_skips_existing_s3(monkeypatch, tmp_path):
    dest = tmp_path / "a"
    dest.write_bytes(b"cached")

    def _boom(url, dest):
        raise AssertionError("should not download when the destination exists")

    monkeypatch.setattr(fetch, "download_s3_object", _boom)
    asyncio.run(fetch_files([Artifact("s3://b/a", dest)]))
    assert dest.read_bytes() == b"cached"


def test_fetch_files_rejects_unknown_scheme_before_downloading(monkeypatch, tmp_path):
    def _boom(url, dest):
        raise AssertionError("must validate every scheme before any download runs")

    monkeypatch.setattr(fetch, "download_s3_object", _boom)
    with pytest.raises(ValueError, match="scheme"):
        asyncio.run(fetch_files([Artifact("s3://b/a", tmp_path / "a"), Artifact("ftp://h/b", tmp_path / "b")]))


# ---- ensure_archive ---------------------------------------------------------------------------


def test_ensure_archive_downloads_and_extracts_http(monkeypatch, tmp_path):
    payload = _zip_bytes("hello.txt", "hi")

    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(payload)
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_http_many", _fake_many)
    target = tmp_path / "out"
    result = asyncio.run(ensure_archive("https://h/data.zip", target))
    assert result == target
    assert (target / "hello.txt").read_text() == "hi"


def test_ensure_archive_is_idempotent(monkeypatch, tmp_path):
    payload = _zip_bytes("hello.txt", "hi")
    calls = {"n": 0}

    def _fake_s3(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        calls["n"] += 1

    monkeypatch.setattr(fetch, "download_s3_object", _fake_s3)
    target = tmp_path / "out"
    asyncio.run(ensure_archive("s3://b/data.zip", target))
    asyncio.run(ensure_archive("s3://b/data.zip", target))  # marker short-circuits
    assert calls["n"] == 1
    assert (target / "hello.txt").read_text() == "hi"


def test_ensure_archive_disambiguates_same_basename_urls(monkeypatch, tmp_path):
    # Two different archives that share the basename "data.zip" must both extract, not silently skip.
    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        for artifact in artifacts:
            member = "from_a.txt" if "hostA" in artifact.url else "from_b.txt"
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(_zip_bytes(member, "x"))
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_http_many", _fake_many)
    target = tmp_path / "out"
    asyncio.run(ensure_archive("https://hostA/data.zip", target))
    asyncio.run(ensure_archive("https://hostB/data.zip", target))
    assert (target / "from_a.txt").exists()
    assert (target / "from_b.txt").exists()  # second archive not skipped despite the shared basename


def test_ensure_archive_rejects_unknown_scheme(tmp_path):
    with pytest.raises(ValueError, match="scheme"):
        asyncio.run(ensure_archive("ftp://h/data.zip", tmp_path / "out"))


# ---- filename safety + sha256 -----------------------------------------------------------------


def test_safe_filename_uses_the_last_path_segment():
    assert fetch._safe_filename("https://host/a/b/records.zip") == "records.zip"


def test_safe_filename_falls_back_when_the_path_has_no_final_segment():
    name = fetch._safe_filename("https://host/dataset/download/")
    assert name.endswith(".download")  # deterministic hash-based fallback, never empty


def test_ensure_archive_uses_a_filename_override(monkeypatch, tmp_path):
    payload = _zip_bytes("hello.txt", "hi")
    seen = {}

    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(payload)
            seen["name"] = artifact.dest.name
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_http_many", _fake_many)
    target = tmp_path / "out"
    asyncio.run(ensure_archive("https://host/s/xyz/download/", target, filename="records.zip"))
    assert seen["name"].endswith("-records.zip")  # override used (URL-hash prefixed), not the .download fallback
    assert (target / "hello.txt").read_text() == "hi"


def test_fetch_files_forwards_sha256_to_the_download(monkeypatch, tmp_path):
    seen = {}

    async def _fake_many(artifacts, *, headers=None, cookies=None, max_concurrency=8, skip_existing=True):
        seen["sha256"] = artifacts[0].sha256
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(b"x")
        return [artifact.dest for artifact in artifacts]

    monkeypatch.setattr(fetch, "download_http_many", _fake_many)
    asyncio.run(fetch_files([Artifact("https://h/a", tmp_path / "a", sha256="abc123")]))
    assert seen["sha256"] == "abc123"
