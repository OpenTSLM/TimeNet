from pathlib import Path
import sys
import zipfile

import boto3
from botocore import UNSIGNED
import pytest

from timenet_connectors.bases import physionet
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.bases.s3 import _s3_client, download_s3_object


class _Conn(BasePhysioNetConnector[str]):
    def download(self, cache_dir: Path) -> list[str]:
        return []

    def convert(self, raw_refs: list[str]):
        raise NotImplementedError


def _zip_bytes(name: str, content: str) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


# ---- the reusable S3 helper -------------------------------------------------------------------


def _spy_client(monkeypatch) -> dict:
    # Capture what _s3_client passes to boto3.client, without building a real (credential-resolving)
    # client that would touch the machine's AWS config.
    captured: dict = {}

    def fake_client(service, **kwargs):
        captured["service"] = service
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    return captured


def test_s3_client_unsigned_without_env_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    captured = _spy_client(monkeypatch)
    _s3_client()
    assert captured["service"] == "s3"
    assert captured["kwargs"]["config"].signature_version is UNSIGNED


def test_s3_client_signed_with_env_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    captured = _spy_client(monkeypatch)
    _s3_client()
    assert captured["service"] == "s3"
    assert "config" not in captured["kwargs"]  # signed: env creds, no forced UNSIGNED config


def test_download_s3_object_rejects_non_s3_url():
    with pytest.raises(ValueError, match="s3://"):
        download_s3_object("https://example.com/x.zip", Path("dest"))


def test_missing_boto3_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)  # `import boto3` -> ImportError
    with pytest.raises(ImportError, match="physionet"):
        download_s3_object("s3://bucket/key.zip", Path("dest"))


# ---- _ensure_archive scheme routing ----------------------------------------------------------


def test_ensure_archive_routes_s3_and_marks_sentinel(monkeypatch, tmp_path):
    payload = _zip_bytes("hello.txt", "hi")
    monkeypatch.setattr(physionet, "download_s3_object", lambda url, dest: dest.write_bytes(payload))

    cache = tmp_path / "cache"
    _Conn()._ensure_archive("s3://bucket/data.zip", cache, "marker")
    assert (cache / "hello.txt").read_text() == "hi"
    assert (cache / "marker").exists()  # sentinel created after extraction

    # Idempotent: a second call short-circuits on the sentinel and never re-downloads.
    def _boom(url, dest):
        raise AssertionError("should not re-download when the sentinel exists")

    monkeypatch.setattr(physionet, "download_s3_object", _boom)
    _Conn()._ensure_archive("s3://bucket/data.zip", cache, "marker")


def test_ensure_archive_routes_http(monkeypatch, tmp_path):
    payload = _zip_bytes("hello.txt", "hi")
    monkeypatch.setattr(
        BasePhysioNetConnector, "_stream_download", staticmethod(lambda url, dest: dest.write_bytes(payload))
    )
    monkeypatch.setattr(
        physionet, "download_s3_object", lambda url, dest: (_ for _ in ()).throw(AssertionError("http must not use s3"))
    )

    cache = tmp_path / "cache"
    _Conn()._ensure_archive("https://example.com/data.zip", cache, "marker")
    assert (cache / "hello.txt").read_text() == "hi"
