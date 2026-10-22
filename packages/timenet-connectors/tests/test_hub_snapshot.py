from pathlib import Path
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from timenet.errors import TimeFValidationError, TimeNetDownloadError
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

    def hf_hub_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / "snapshot" / filename)

    _fake_hub(monkeypatch, hf_hub_download=hf_hub_download)
    refs = _Snapshot().download(tmp_path / "cache")

    assert refs == [
        HubFileRef(tmp_path / "snapshot" / "data/train.jsonl", "train", 3),
        HubFileRef(tmp_path / "snapshot" / "data/test.jsonl", "test", None),
    ]
    kwargs = {"repo_type": "dataset", "revision": "0" * 40, "cache_dir": str(tmp_path / "cache")}
    assert calls == [
        ("org/repo", "data/train.jsonl", kwargs),
        ("org/repo", "data/test.jsonl", kwargs),
    ]


def test_download_names_declared_files_missing_from_the_revision(monkeypatch, tmp_path):
    def hf_hub_download(repo, filename, **kwargs):
        if filename == "data/test.jsonl":
            raise FileNotFoundError("404 entry not found")
        return str(tmp_path / filename)

    _fake_hub(monkeypatch, hf_hub_download=hf_hub_download)
    with pytest.raises(TimeNetDownloadError, match=rf"could not download data/test\.jsonl from org/repo@{'0' * 40}"):
        _Snapshot().download(tmp_path / "cache")


def test_subclass_rejects_row_counts_for_undeclared_parts():
    with pytest.raises(TimeFValidationError, match=r"SOURCE_ROWS names parts absent from SOURCE_FILES: trian$"):

        class _Misspelled(_Snapshot):
            SOURCE_ROWS: ClassVar = {"trian": 3}


@pytest.mark.parametrize("revision", ["main", "refs/convert/parquet", "ABC" * 13 + "D", "0" * 39])
def test_subclass_rejects_mutable_or_malformed_revisions(revision):
    with pytest.raises(TimeFValidationError, match=r"SOURCE_REVISION must be a 40-character commit"):
        type("_Branch", (_Snapshot,), {"SOURCE_REVISION": revision})


@pytest.mark.parametrize(
    ("expected_rows", "count", "message"),
    [
        (None, 0, r"\(train\): no rows$"),
        (3, 0, r"\(train\): no rows$"),
        (3, 2, r"\(train\): expected 3 rows, found 2$"),
    ],
)
def test_check_rows_rejects_empty_or_unexpected_counts(expected_rows, count, message):
    ref = HubFileRef(Path("train.jsonl"), "train", expected_rows)
    with pytest.raises(TimeFValidationError, match=message):
        ref.check_rows(count)


@pytest.mark.parametrize(("expected_rows", "count"), [(None, 5), (3, 3)])
def test_check_rows_accepts_matching_or_unpinned_counts(expected_rows, count):
    HubFileRef(Path("train.jsonl"), "train", expected_rows).check_rows(count)


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
