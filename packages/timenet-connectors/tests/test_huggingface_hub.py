"""Tests for the pinned Hub base and its download helper, driven against a fake huggingface_hub."""

import asyncio
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.testing.builders import make_dataset
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.download.progress import DownloadProgress, progress_sink
from timenet_connectors.sources.huggingface_hub import _DEFAULT_MAX_CONCURRENCY, download_hub_files


_SHA = "0" * 40


class _Pinned(BaseHuggingFaceConnector):
    HF_REPO = "org/repo"
    SOURCE_REVISION = _SHA
    SOURCE_FILES: ClassVar = {"train": "data/train.jsonl", "test": "data/test.jsonl"}

    def convert(self, raw_refs):
        raise NotImplementedError


def _hf_hub_url(repo, filename, *, repo_type, revision):
    return f"https://hub/{repo_type}s/{repo}/resolve/{revision}/{filename}"


def _fake_hub(monkeypatch, tmp_path, download=None):
    """Install a fake huggingface_hub whose hf_hub_download writes the file into ``tmp_path``."""
    calls = []

    def hf_hub_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        if download is not None:
            return download(repo, filename, **kwargs)
        path = tmp_path / "snapshot" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
        return str(path)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf_hub_download, hf_hub_url=_hf_hub_url)
    )
    return calls


def test_download_returns_one_ref_per_source_file_in_declared_order(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path)

    refs = asyncio.run(_Pinned().download_async(tmp_path / "cache"))

    assert refs == [
        HubFileRef(tmp_path / "snapshot" / "data/train.jsonl", "train"),
        HubFileRef(tmp_path / "snapshot" / "data/test.jsonl", "test"),
    ]
    kwargs = {"repo_type": "dataset", "revision": _SHA, "cache_dir": str(tmp_path / "cache"), "tqdm_class": None}
    assert sorted(calls) == [("org/repo", "data/test.jsonl", kwargs), ("org/repo", "data/train.jsonl", kwargs)]


@pytest.mark.parametrize(("limit", "expected"), [({}, _DEFAULT_MAX_CONCURRENCY), ({"max_concurrency": 3}, 3)])
def test_downloads_run_concurrently_up_to_the_limit(monkeypatch, tmp_path, limit, expected):
    lock = threading.Lock()
    running = 0
    peak = 0

    def download(repo, filename, **kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        path = tmp_path / filename
        path.write_text("x")
        return str(path)

    _fake_hub(monkeypatch, tmp_path, download)
    filenames = [f"part-{index}.parquet" for index in range(_DEFAULT_MAX_CONCURRENCY + 4)]
    asyncio.run(download_hub_files("org/repo", filenames, revision=_SHA, cache_dir=tmp_path, **limit))
    assert peak == expected


def test_downloads_report_progress_to_the_sink_without_printing(monkeypatch, tmp_path, capsys):
    def download(repo, filename, *, tqdm_class, **kwargs):
        # Mirror how huggingface_hub drives a custom tqdm_class: Hub-only keywords, then byte updates.
        with tqdm_class(total=10, initial=2, desc=filename, name="huggingface_hub.http_get", unit="B") as bar:
            bar.update(3)
            bar.update(5)
            bar.update_transfer(8)
            bar.set_transfer_postfix_str("fast", refresh=False)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return str(path)

    _fake_hub(monkeypatch, tmp_path, download)
    events = []
    with progress_sink(events.append):
        asyncio.run(download_hub_files("org/repo", ["data/a.jsonl"], revision=_SHA, cache_dir=tmp_path))

    url = f"https://hub/datasets/org/repo/resolve/{_SHA}/data/a.jsonl"
    assert events == [DownloadProgress(url, 5, 10), DownloadProgress(url, 10, 10)]
    assert capsys.readouterr().err == ""


def test_download_failure_names_file_and_pinned_revision(monkeypatch, tmp_path):
    def download(repo, filename, **kwargs):
        if filename.endswith("test.jsonl"):
            raise OSError("404 entry not found")
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return str(path)

    _fake_hub(monkeypatch, tmp_path, download)
    with pytest.raises(TimeNetDownloadError, match=rf"could not download data/test\.jsonl from org/repo@{_SHA}: 404"):
        asyncio.run(_Pinned().download_async(tmp_path / "cache"))


def test_a_file_missing_after_download_fails(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, lambda repo, filename, **kwargs: str(tmp_path / "nowhere" / filename))
    # Both files are missing; whichever download finishes first is reported.
    with pytest.raises(TimeNetDownloadError, match=rf"data/t\w+\.jsonl from org/repo@{_SHA} is missing"):
        asyncio.run(_Pinned().download_async(tmp_path / "cache"))


def test_missing_hub_library_states_the_fix(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match=r"needs huggingface_hub.*requirements\.txt"):
        asyncio.run(_Pinned().download_async(tmp_path))


@pytest.mark.parametrize("revision", ["main", "refs/convert/parquet", "A" * 40, "0" * 39])
def test_subclass_rejects_mutable_or_malformed_revisions(revision):
    with pytest.raises(TimeFValidationError, match=r"SOURCE_REVISION must be a 40-character commit"):
        type("_Branch", (_Pinned,), {"SOURCE_REVISION": revision})


@pytest.mark.parametrize(
    ("attr", "value"),
    [("SOURCE_REVISION", None), ("SOURCE_FILES", None), ("SOURCE_FILES", {})],
)
def test_subclass_rejects_a_missing_or_empty_declaration(attr, value):
    attrs = {"HF_REPO": "org/repo", "SOURCE_REVISION": _SHA, "SOURCE_FILES": {"a": "a.jsonl"}}
    if value is None:
        del attrs[attr]
    else:
        attrs[attr] = value
    with pytest.raises(TimeFValidationError, match=attr):
        type("_Invalid", (BaseHuggingFaceConnector,), attrs)


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ((), r"requires nonempty, distinct part references"),
        (("train", "train"), r"requires nonempty, distinct part references"),
        (("train", "valid", "dev"), r"has no source part dev, valid$"),
    ],
)
def test_check_refs_rejects_empty_repeated_or_undeclared_parts(parts, message):
    refs = [HubFileRef(Path(f"{part}.jsonl"), part) for part in parts]
    with pytest.raises(TimeFValidationError, match=message):
        _Pinned().check_refs(refs)


def test_new_dataset_checks_refs_and_records_provenance(monkeypatch):
    monkeypatch.setattr(_Pinned, "metadata", lambda self: make_dataset().metadata)
    with pytest.raises(TimeFValidationError, match=r"has no source part dev$"):
        _Pinned().new_dataset([HubFileRef(Path("dev.jsonl"), "dev")], "pin")
    dataset = _Pinned().new_dataset([HubFileRef(Path("test.jsonl"), "test")], "pin")
    assert {(a.id, a.key, a.value) for a in dataset.annotations} == {
        ("pin-source_repository", "source_repository", "org/repo"),
        ("pin-source_revision", "source_revision", _SHA),
    }
