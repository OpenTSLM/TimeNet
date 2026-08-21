from pathlib import Path
import sys
import threading

import boto3
from botocore import UNSIGNED
from botocore.exceptions import BotoCoreError
import pytest

from timenet.errors import TimeNetBuildError
from timenet_connectors.download import s3
from timenet_connectors.download.progress import progress_sink
from timenet_connectors.download.s3 import _s3_client, download_s3_object


def _spy_session(monkeypatch, *, credentials, fail: bool = False) -> dict:
    # Capture what _s3_client passes to Session.client, and control credential resolution, without
    # touching the machine's AWS config.
    captured: dict = {}

    class _FakeSession:
        def get_credentials(self):
            if fail:
                raise BotoCoreError()
            return credentials

        def client(self, service, **kwargs):
            captured["service"] = service
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setattr(boto3.session, "Session", _FakeSession)
    return captured


def test_s3_client_reads_unsigned_without_credentials(monkeypatch):
    # No credentials resolve: read public data anonymously with a forced UNSIGNED signature.
    captured = _spy_session(monkeypatch, credentials=None)
    _s3_client()
    assert captured["service"] == "s3"
    assert captured["kwargs"]["config"].signature_version is UNSIGNED


def test_s3_client_uses_credentials_when_available(monkeypatch):
    # Credentials resolve (env / profile / SSO / instance role): use them, with no forced UNSIGNED.
    captured = _spy_session(monkeypatch, credentials=object())
    _s3_client()
    assert captured["service"] == "s3"
    assert captured["kwargs"] == {}


def test_s3_client_falls_back_to_unsigned_on_credential_error(monkeypatch):
    # A broken credential provider (for example an SSO profile missing an optional dependency) must not
    # stop an anonymous read of a public bucket.
    captured = _spy_session(monkeypatch, credentials=None, fail=True)
    _s3_client()
    assert captured["kwargs"]["config"].signature_version is UNSIGNED


def test_download_s3_object_rejects_non_s3_url():
    with pytest.raises(ValueError, match="s3://"):
        download_s3_object("https://example.com/x.zip", Path("dest"))


def test_missing_boto3_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)  # `import boto3` -> ImportError
    with pytest.raises(TimeNetBuildError, match=r"requirements\.txt"):
        download_s3_object("s3://bucket/key.zip", Path("dest"))


def test_download_s3_object_writes_atomically(monkeypatch, tmp_path):
    class _FakeClient:
        def download_file(self, Bucket, Key, Filename, Callback=None):  # noqa: N803 (boto3's kwarg names)
            assert Filename.endswith(".part")  # boto3 writes to the temp file, not the final dest
            Path(Filename).write_bytes(b"payload")

    monkeypatch.setattr(s3, "_s3_client", _FakeClient)
    dest = tmp_path / "obj.bin"
    download_s3_object("s3://bucket/key.bin", dest)
    assert dest.read_bytes() == b"payload"
    assert not (tmp_path / "obj.bin.part").exists()  # renamed into place, temp cleaned up


def test_download_s3_object_cleans_up_part_on_failure(monkeypatch, tmp_path):
    class _FailingClient:
        def download_file(self, Bucket, Key, Filename, Callback=None):  # noqa: N803
            Path(Filename).write_bytes(b"partial")  # a partial landed before the failure
            raise RuntimeError("connection reset")

    monkeypatch.setattr(s3, "_s3_client", _FailingClient)
    dest = tmp_path / "obj.bin"
    with pytest.raises(RuntimeError):
        download_s3_object("s3://bucket/key.bin", dest)
    assert not dest.exists()  # no file produced on failure
    assert not (tmp_path / "obj.bin.part").exists()  # partial removed


def test_download_s3_object_reports_progress(monkeypatch, tmp_path):
    class _FakeClient:
        def head_object(self, Bucket, Key):  # noqa: N803 (boto3's kwarg names)
            return {"ContentLength": 100}

        def download_file(self, Bucket, Key, Filename, Callback=None):  # noqa: N803
            Path(Filename).parent.mkdir(parents=True, exist_ok=True)
            Path(Filename).write_bytes(b"x" * 100)
            if Callback is None:
                return

            def _report() -> None:
                Callback(60)  # boto3 reports bytes transferred incrementally
                Callback(40)

            # boto3 fires the Callback from a transfer worker thread, which does not inherit the ambient
            # ContextVar sink; running it in a thread here fails the test unless the sink was captured.
            worker = threading.Thread(target=_report)
            worker.start()
            worker.join()

    monkeypatch.setattr(s3, "_s3_client", _FakeClient)
    events = []
    with progress_sink(events.append):
        download_s3_object("s3://bucket/key.bin", tmp_path / "key.bin")

    assert [event.downloaded for event in events] == [60, 100]  # accumulated across the worker thread
    assert all(event.total == 100 for event in events)
