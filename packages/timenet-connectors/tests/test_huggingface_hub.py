"""Tests for the pinned Hub base and its download helper, driven against a fake huggingface_hub."""

import asyncio
from fnmatch import fnmatchcase
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.download.progress import DownloadProgress, progress_sink
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_SHA = "0" * 40


class _Pinned(
    BaseHuggingFaceConnector,
    hf_repo="org/repo",
    source_revision=_SHA,
    source_files={"train": "data/train.jsonl", "test": "data/test.jsonl"},
):
    def convert(self, files):
        return files


class _HubTqdm:
    """Stand in for ``huggingface_hub.utils.tqdm``, which accepts the Hub-only ``name`` keyword."""

    def __init__(self, *args, name=None, disable=False, total=None, **kwargs):
        assert disable
        self.total = total

    def update(self, n=1):
        return None


def _fake_hub(monkeypatch, tmp_path, files=("data/train.jsonl", "data/test.jsonl"), error=None, bars=None):
    """Install a fake huggingface_hub. Its snapshot_download writes the revision's ``files`` that match the
    allowed patterns into a snapshot, as the Hub does."""
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        if bars is not None:
            bars(kwargs["tqdm_class"])
        if error is not None:
            raise error
        snapshot = tmp_path / "snapshot"
        for name in files:
            if any(fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
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


def test_download_returns_the_path_of_each_part_in_declared_order(monkeypatch, tmp_path):
    # No sink is installed, so the progress class must stay silent without one.
    calls = _fake_hub(monkeypatch, tmp_path, bars=_snapshot_bars)

    files = asyncio.run(_Pinned().download_async(tmp_path / "cache"))

    assert list(files.items()) == [
        ("train", tmp_path / "snapshot" / "data/train.jsonl"),
        ("test", tmp_path / "snapshot" / "data/test.jsonl"),
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
    _fake_hub(monkeypatch, tmp_path, files=("data/train.jsonl",))
    with pytest.raises(TimeNetDownloadError, match=rf"org/repo@{_SHA} has no data/test\.jsonl$"):
        asyncio.run(_Pinned().download_async(tmp_path / "cache"))


def test_missing_hub_library_states_the_fix(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match=r"needs huggingface_hub.*requirements\.txt"):
        asyncio.run(_Pinned().download_async(tmp_path))


@pytest.mark.parametrize("revision", ["main", "refs/convert/parquet", "A" * 40, "0" * 39])
def test_subclass_rejects_mutable_or_malformed_revisions(revision):
    with pytest.raises(TimeFValidationError, match=r"source_revision must be a 40-character commit"):
        types.new_class("_Branch", (_Pinned,), {"source_revision": revision})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("hf_repo", None), ("source_revision", None), ("source_files", None), ("source_files", {})],
)
def test_subclass_rejects_a_missing_or_empty_keyword(keyword, value):
    keywords = {"hf_repo": "org/repo", "source_revision": _SHA, "source_files": {"a": "a.jsonl"}}
    if value is None:
        del keywords[keyword]
    else:
        keywords[keyword] = value
    with pytest.raises(TimeFValidationError, match=keyword):
        types.new_class("_Invalid", (BaseHuggingFaceConnector,), keywords)


def test_subclass_stores_the_keywords_and_inherits_them():
    assert _Pinned.HF_REPO == "org/repo"
    assert _Pinned.SOURCE_REVISION == _SHA
    assert dict(_Pinned.SOURCE_FILES) == {"train": "data/train.jsonl", "test": "data/test.jsonl"}
    with pytest.raises(TypeError):
        _Pinned.SOURCE_FILES["dev"] = "data/dev.jsonl"  # ty: ignore[invalid-assignment]
    child = type("_Child", (_Pinned,), {})
    assert (child.HF_REPO, child.SOURCE_REVISION, child.SOURCE_FILES) == (
        _Pinned.HF_REPO,
        _Pinned.SOURCE_REVISION,
        _Pinned.SOURCE_FILES,
    )


def test_snapshot_downloads_the_matching_files_and_returns_the_snapshot_root(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path, files=("meta.csv", "data/train.parquet", "README.md"))

    # asyncio.run gives the helper a running event loop, as download_async does in a build.
    root = asyncio.run(hub_snapshot("org/repo", _SHA, tmp_path, ("data/*.parquet",)))
    assert root == tmp_path / "snapshot"
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()) == [
        "data/train.parquet"
    ]
    assert calls[0][1]["allow_patterns"] == ["data/*.parquet"]


class _Patterned(
    BaseHuggingFaceConnector,
    hf_repo="org/repo",
    source_revision=_SHA,
    source_files={"questions": "qa.csv", "series": "data/*.parquet"},
):
    def convert(self, files):
        return files


def test_a_pattern_part_resolves_to_the_directory_of_its_files(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path, files=("qa.csv", "data/a.parquet", "data/b.parquet", "README.md"))

    files = asyncio.run(_Patterned().download_async(tmp_path / "cache"))

    assert files == {"questions": tmp_path / "snapshot" / "qa.csv", "series": tmp_path / "snapshot" / "data"}
    assert calls[0][1]["allow_patterns"] == ["qa.csv", "data/*.parquet"]


def test_a_declared_file_or_pattern_without_a_match_fails(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, files=("qa.csv", "README.md"))
    with pytest.raises(TimeNetDownloadError, match=rf"org/repo@{_SHA} has no data/\*\.parquet$"):
        asyncio.run(_Patterned().download_async(tmp_path / "cache"))


_TSQA_CARD = Path(__file__).parents[1] / "src/timenet_connectors/datasets/chengsenwang/tsqa/dataset.yaml"


def _card(tmp_path: Path, **fields: str) -> Path:
    lines = [line for line in _TSQA_CARD.read_text().splitlines() if not line.startswith("source_")]
    path = tmp_path / "dataset.yaml"
    path.write_text("\n".join([*lines, *(f'{key}: "{value}"' for key, value in fields.items())]) + "\n")
    return path


def test_metadata_takes_the_source_from_the_connector(tmp_path):
    connector = type("_Carded", (_Pinned,), {"CARD": _card(tmp_path)})()
    metadata = connector.metadata()
    assert metadata.source_url == "https://huggingface.co/datasets/org/repo"
    assert metadata.source_revision == _SHA


def test_metadata_accepts_a_card_that_repeats_the_source(tmp_path):
    card = _card(tmp_path, source_url="https://huggingface.co/datasets/org/repo", source_revision=_SHA)
    assert type("_Carded", (_Pinned,), {"CARD": card})().metadata().source_revision == _SHA


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_url", "https://huggingface.co/datasets/org/other"), ("source_revision", "1" * 40)],
)
def test_metadata_rejects_a_card_source_that_differs(tmp_path, field, value):
    connector = type("_Carded", (_Pinned,), {"CARD": _card(tmp_path, **{field: value})})()
    with pytest.raises(TimeFValidationError, match=f"card sets {field}"):
        connector.metadata()
