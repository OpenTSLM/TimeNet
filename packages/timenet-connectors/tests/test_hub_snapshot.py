from pathlib import Path
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from timenet.errors import TimeNetDownloadError
from timenet_connectors.bases.huggingface import BaseHubSnapshotConnector, HubFileRef
from timenet_connectors.sources.huggingface_hub import hub_file, hub_files, hub_snapshot


class _Snapshot(BaseHubSnapshotConnector):
    HF_REPO = "org/repo"
    SOURCE_REVISION = "0" * 40
    SOURCE_FILES: ClassVar = {"train": "data/train.jsonl", "test": "data/test.jsonl"}
    SOURCE_ROWS: ClassVar = {"train": 3}

    def convert(self, raw_refs):
        raise NotImplementedError


def _fake_hub(monkeypatch, **functions):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(**functions))


def test_download_returns_one_ref_per_source_file(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path / "snapshot")

    _fake_hub(monkeypatch, snapshot_download=snapshot_download)
    refs = _Snapshot().download(tmp_path / "cache")

    assert refs == [
        HubFileRef(tmp_path / "snapshot" / "data/train.jsonl", "train", "data/train.jsonl", 3),
        HubFileRef(tmp_path / "snapshot" / "data/test.jsonl", "test", "data/test.jsonl", None),
    ]
    assert calls == [
        (
            "org/repo",
            {
                "repo_type": "dataset",
                "revision": "0" * 40,
                "cache_dir": str(tmp_path / "cache"),
                "allow_patterns": ["data/train.jsonl", "data/test.jsonl"],
            },
        )
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda: hub_files("org/repo", "main"),
        lambda: hub_file("org/repo", "main", Path(), "data.parquet"),
        lambda: hub_snapshot("org/repo", "main", Path(), ()),
    ],
)
def test_missing_hub_library_states_the_fix(monkeypatch, call):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match=r"needs huggingface_hub.*requirements\.txt"):
        call()


def test_hub_failures_name_the_pinned_revision(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    _fake_hub(monkeypatch, list_repo_files=fail, hf_hub_download=fail, snapshot_download=fail)
    with pytest.raises(TimeNetDownloadError, match=r"could not list org/repo@abc: offline"):
        hub_files("org/repo", "abc")
    with pytest.raises(TimeNetDownloadError, match=r"could not download a\.parquet from org/repo@abc: offline"):
        hub_file("org/repo", "abc", Path(), "a.parquet")
    with pytest.raises(TimeNetDownloadError, match=r"could not download org/repo@abc: offline"):
        hub_snapshot("org/repo", "abc", Path(), ())
