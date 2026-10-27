"""Tests for the pinned Hub base and its download helper, driven against a fake huggingface_hub."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.download.progress import DownloadProgress, progress_sink
from timenet_connectors.sources.huggingface_hub import download_hub_files


_SHA = "0" * 40


class _Pinned(BaseHuggingFaceConnector):
    HF_REPO = "org/repo"
    SOURCE_REVISION = _SHA
    SOURCE_FILES: ClassVar = {"train": "data/train.jsonl", "test": "data/test.jsonl"}

    def convert(self, raw_refs):
        return raw_refs


class _HubTqdm:
    """Stand in for ``huggingface_hub.utils.tqdm``, which accepts the Hub-only ``name`` keyword."""

    def __init__(self, *args, name=None, disable=False, total=None, **kwargs):
        assert disable
        self.total = total

    def update(self, n=1):
        return None


def _fake_hub(monkeypatch, tmp_path, present=None, error=None, bars=None):
    """Install a fake huggingface_hub whose snapshot_download writes the allowed files into a snapshot."""
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        if bars is not None:
            bars(kwargs["tqdm_class"])
        if error is not None:
            raise error
        snapshot = tmp_path / "snapshot"
        for name in kwargs["allow_patterns"]:
            if present is None or name in present:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", SimpleNamespace(tqdm=_HubTqdm))
    return calls


def _snapshot_bars(tqdm_class):
    # Mirror the bars huggingface_hub 1.28 opens: file bytes, network bytes, and a count of finished files.
    file_bytes = tqdm_class(name="huggingface_hub.snapshot_download", total=10, unit="B")
    network = tqdm_class(name="huggingface_hub.snapshot_download.transfer", total=10, unit="B")
    files = tqdm_class(desc="Fetching 2 files", total=2)
    file_bytes.update(4)
    network.update(3)
    files.update(1)
    file_bytes.update(6)


def test_download_returns_one_ref_per_source_file_in_declared_order(monkeypatch, tmp_path):
    # No sink is installed, so the progress class must stay silent without one.
    calls = _fake_hub(monkeypatch, tmp_path, bars=_snapshot_bars)

    refs = asyncio.run(_Pinned().download_async(tmp_path / "cache"))

    assert refs == [
        HubFileRef(tmp_path / "snapshot" / "data/train.jsonl", "train"),
        HubFileRef(tmp_path / "snapshot" / "data/test.jsonl", "test"),
    ]
    ((repo, kwargs),) = calls
    assert issubclass(kwargs.pop("tqdm_class"), _HubTqdm)
    assert (repo, kwargs) == (
        "org/repo",
        {
            "repo_type": "dataset",
            "revision": _SHA,
            "cache_dir": tmp_path / "cache",
            "allow_patterns": ["data/train.jsonl", "data/test.jsonl"],
        },
    )


def test_download_reports_file_bytes_to_the_sink_as_one_row(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, bars=_snapshot_bars)
    events = []
    with progress_sink(events.append):
        asyncio.run(_Pinned().download_async(tmp_path / "cache"))

    url = "org/repo@0000000"
    assert events == [DownloadProgress(url, 4, 10), DownloadProgress(url, 10, 10)]


def test_download_failure_names_the_pinned_revision(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, error=OSError("404 entry not found"))
    with pytest.raises(TimeNetDownloadError, match=rf"Could not download org/repo@{_SHA}: 404"):
        asyncio.run(_Pinned().download_async(tmp_path / "cache"))


def test_a_file_absent_from_the_revision_fails(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, present={"data/train.jsonl"})
    with pytest.raises(TimeNetDownloadError, match=rf"org/repo@{_SHA} has no data/test\.jsonl$"):
        asyncio.run(
            download_hub_files("org/repo", ["data/train.jsonl", "data/test.jsonl"], revision=_SHA, cache_dir=tmp_path)
        )


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


def test_convert_checks_refs_before_the_subclass_runs():
    with pytest.raises(TimeFValidationError, match=r"has no source part dev$"):
        _Pinned().convert([HubFileRef(Path("dev.jsonl"), "dev")])
    refs = [HubFileRef(Path("test.jsonl"), "test")]
    assert _Pinned().convert(refs) is refs
    assert _Pinned.convert.__name__ == "convert"
