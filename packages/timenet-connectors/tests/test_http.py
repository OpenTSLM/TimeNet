import hashlib
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.bases import http


def _zip_bytes(name: str, content: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return self._chunks


class _FakeSession:
    """A stand-in for requests: serves ``payload`` split across two chunks."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls = 0

    def get(self, url: str, *, stream: bool, timeout: int, headers: object = None) -> _FakeResponse:
        self.calls += 1
        mid = len(self._payload) // 2
        return _FakeResponse([self._payload[:mid], self._payload[mid:]])


class _BoomSession:
    def get(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("must not download")


def test_download_file_streams_and_leaves_no_part(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "f.bin"
    http.download_file("https://example.com/f.bin", dest, session=_FakeSession(b"hello world"))
    assert dest.read_bytes() == b"hello world"
    assert not dest.with_name("f.bin.part").exists()


def test_download_file_accepts_matching_sha256(tmp_path: Path) -> None:
    payload = b"payload-content"
    dest = tmp_path / "f.bin"
    http.download_file(
        "https://example.com/f.bin",
        dest,
        session=_FakeSession(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert dest.read_bytes() == payload


def test_download_file_rejects_sha256_mismatch(tmp_path: Path) -> None:
    dest = tmp_path / "f.bin"
    with pytest.raises(TimeFFormatError, match="SHA-256"):
        http.download_file(
            "https://example.com/f.bin", dest, session=_FakeSession(b"payload"), expected_sha256="00" * 32
        )
    assert not dest.exists()
    assert not dest.with_name("f.bin.part").exists()


def test_ensure_file_reuses_existing(tmp_path: Path) -> None:
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"cached")
    result = http.ensure_file("https://example.com/f.bin", dest, session=_BoomSession())
    assert result == dest
    assert dest.read_bytes() == b"cached"


def test_ensure_archive_extracts_once_and_marks_sentinel(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    http.ensure_archive(
        "https://example.com/data.zip", cache, "marker", session=_FakeSession(_zip_bytes("hello.txt", "hi"))
    )
    assert (cache / "hello.txt").read_text() == "hi"
    assert (cache / "marker").exists()

    # Idempotent: the sentinel short-circuits, so no second download is attempted.
    http.ensure_archive("https://example.com/data.zip", cache, "marker", session=_BoomSession())


def test_ensure_archive_filename_override_for_slash_url(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    http.ensure_archive(
        "https://example.com/records/download/",
        cache,
        "marker",
        filename="data.zip",
        session=_FakeSession(_zip_bytes("a.txt", "x")),
    )
    assert (cache / "data.zip").exists()
    assert (cache / "a.txt").read_text() == "x"


def test_ensure_archive_routes_s3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _zip_bytes("h.txt", "hi")
    monkeypatch.setattr(http, "download_s3_object", lambda url, dest: dest.write_bytes(payload))
    cache = tmp_path / "cache"
    http.ensure_archive("s3://bucket/data.zip", cache, "marker")
    assert (cache / "h.txt").read_text() == "hi"


def test_missing_requests_raises_http_extra_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "requests", None)  # `import requests` -> ImportError
    with pytest.raises(ImportError, match="http"):
        http.download_file("https://example.com/f.bin", tmp_path / "f.bin")


def test_filename_from_url_falls_back_on_trailing_slash() -> None:
    assert http._filename_from_url("https://example.com/a/b.zip") == "b.zip"
    assert http._filename_from_url("https://example.com/dir/").endswith(".download")
