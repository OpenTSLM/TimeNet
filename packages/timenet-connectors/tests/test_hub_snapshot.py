from pathlib import Path
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.testing.builders import make_dataset
from timenet_connectors.bases.huggingface import BaseHubSnapshotConnector, HubFileRef
from timenet_connectors.sources.huggingface_hub import hub_file, hub_files


class _Snapshot(BaseHubSnapshotConnector):
    HF_REPO = "org/repo"
    SOURCE_REVISION = "0" * 40
    SOURCE_FILES: ClassVar = {"train": "data/train.jsonl", "test": "data/test.jsonl"}

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
        HubFileRef(tmp_path / "snapshot" / "data/train.jsonl", "train"),
        HubFileRef(tmp_path / "snapshot" / "data/test.jsonl", "test"),
    ]
    kwargs = {"repo_type": "dataset", "revision": "0" * 40, "cache_dir": str(tmp_path / "cache")}
    assert calls == [
        ("org/repo", "data/train.jsonl", kwargs),
        ("org/repo", "data/test.jsonl", kwargs),
    ]


def test_download_failure_names_file_and_pinned_revision(monkeypatch, tmp_path):
    def hf_hub_download(repo, filename, **kwargs):
        raise OSError("404 entry not found")

    _fake_hub(monkeypatch, hf_hub_download=hf_hub_download)
    with pytest.raises(TimeNetDownloadError, match=rf"could not download data/train\.jsonl from org/repo@{'0' * 40}"):
        _Snapshot().download(tmp_path / "cache")


@pytest.mark.parametrize("revision", ["main", "A" * 40, "0" * 39])
def test_subclass_rejects_mutable_or_malformed_revisions(revision):
    with pytest.raises(TimeFValidationError, match=r"SOURCE_REVISION must be a 40-character commit"):
        type("_Branch", (_Snapshot,), {"SOURCE_REVISION": revision})


@pytest.mark.parametrize(
    ("attr", "value"),
    [("SOURCE_REVISION", None), ("SOURCE_FILES", None), ("SOURCE_FILES", {})],
)
def test_subclass_rejects_a_missing_or_empty_declaration(attr, value):
    attrs = {"HF_REPO": "org/repo", "SOURCE_REVISION": "0" * 40, "SOURCE_FILES": {"a": "a.jsonl"}}
    if value is None:
        del attrs[attr]
    else:
        attrs[attr] = value
    with pytest.raises(TimeFValidationError, match=attr):
        type("_Invalid", (BaseHubSnapshotConnector,), attrs)


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
        _Snapshot().check_refs(refs)


def test_new_dataset_checks_refs_and_records_provenance(monkeypatch):
    monkeypatch.setattr(_Snapshot, "metadata", lambda self: make_dataset().metadata)
    with pytest.raises(TimeFValidationError, match=r"has no source part dev$"):
        _Snapshot().new_dataset([HubFileRef(Path("dev.jsonl"), "dev")], "snap")
    dataset = _Snapshot().new_dataset([HubFileRef(Path("test.jsonl"), "test")], "snap")
    assert {(a.id, a.key, a.value) for a in dataset.annotations} == {
        ("snap-source_repository", "source_repository", "org/repo"),
        ("snap-source_revision", "source_revision", "0" * 40),
    }


@pytest.mark.parametrize(
    "call",
    [
        lambda: hub_files("org/repo", "main"),
        lambda: hub_file("org/repo", "data.parquet", revision="main", cache_dir=Path()),
    ],
)
def test_missing_hub_library_states_the_fix(monkeypatch, call):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match=r"needs huggingface_hub.*requirements\.txt"):
        call()


def test_listing_failures_name_the_pinned_revision(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    _fake_hub(monkeypatch, list_repo_files=fail)
    with pytest.raises(TimeNetDownloadError, match=r"could not list org/repo@abc: offline"):
        hub_files("org/repo", "abc")
