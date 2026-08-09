from pathlib import Path
import sys

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


# ---- archive caching delegates to bases/http.py ----------------------------------------------
# The download/extract/sentinel behaviour lives in bases/http.py and is covered by test_http.py;
# here we only pin that the PhysioNet base forwards to it (so subclasses keep calling self._*).


def test_ensure_archive_delegates_to_http(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_ensure_archive(url, cache_dir, sentinel, *, filename=None):
        captured.update(url=url, cache_dir=cache_dir, sentinel=sentinel, filename=filename)
        return cache_dir

    monkeypatch.setattr(physionet.http, "ensure_archive", fake_ensure_archive)
    cache = tmp_path / "cache"
    result = _Conn()._ensure_archive("https://example.com/data.zip", cache, "marker", filename="d.zip")
    assert result == cache
    assert captured == {
        "url": "https://example.com/data.zip",
        "cache_dir": cache,
        "sentinel": "marker",
        "filename": "d.zip",
    }


def test_stream_download_delegates_to_http(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr(physionet.http, "download_file", lambda url, dest: captured.update(url=url, dest=dest))
    dest = tmp_path / "f.zip"
    BasePhysioNetConnector._stream_download("https://example.com/f.zip", dest)
    assert captured == {"url": "https://example.com/f.zip", "dest": dest}
