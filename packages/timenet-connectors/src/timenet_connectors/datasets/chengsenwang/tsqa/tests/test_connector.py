import json
from pathlib import Path
import sys

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import QATask
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.chengsenwang.tsqa import TSQAConnector


FIXTURE = Path(__file__).parent / "fixtures" / "tsqa_sample.json"


def _fixture_rows():
    return json.loads(FIXTURE.read_text())


def _convert() -> TimeFDataset:
    # The Hub rows are plain dicts; the fixture is exactly that shape, so convert() takes it directly.
    return TSQAConnector().convert(_fixture_rows())


def test_is_a_connector():
    assert isinstance(TSQAConnector(), BaseConnector)
    assert isinstance(TSQAConnector(), BaseHuggingFaceConnector)


def test_metadata():
    assert TSQAConnector().metadata().dataset_id == "chengsenwang/tsqa"
    assert str(TSQAConnector().metadata().license) == "Apache-2.0"


def test_convert_builds_one_sample_per_row():
    dataset = _convert()
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.samples) == len(_fixture_rows())


def test_qa_tasks_match_fixture():
    dataset = _convert()
    rows = _fixture_rows()
    qa = [t for t in dataset.tasks if isinstance(t, QATask)]
    assert len(qa) == len(rows)
    assert {t.question for t in qa} == {r["Question"] for r in rows}
    assert {t.target for t in qa} == {r["Answer"] for r in rows}


def test_series_values_parsed_from_fixture():
    dataset = _convert()
    expected = json.loads(_fixture_rows()[0]["Series"])
    got = dataset.samples[0].time_series[0].to_numpy()
    assert len(got) == len(expected)
    assert float(got[0]) == pytest.approx(expected[0], rel=1e-5)


def test_task_annotation_present():
    keys = {ann.key for sample in _convert().samples for ann in sample.annotations}
    assert "task" in keys


def test_missing_hf_library_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match="huggingface"):
        TSQAConnector().download(Path("cache"))


def test_real_download_reads_auto_converted_parquet(monkeypatch, tmp_path):
    """The real path reads HF's auto-converted parquet branch, not the (CSV-only) main revision."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"Series": ["[1.0, 2.0]"], "Question": ["q"], "Answer": ["a"], "Task": ["t"], "Label": [""]})
    parquet_path = tmp_path / "default" / "train" / "0000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(table, parquet_path)

    seen: dict[str, object] = {}

    def fake_list(repo_id, repo_type=None, revision=None):
        seen["list_revision"] = revision
        return ["README.md", "default/train/0000.parquet"]

    def fake_download(repo_id, filename, repo_type=None, revision=None, cache_dir=None):
        seen["download_revision"] = revision
        return str(parquet_path)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    rows = TSQAConnector().download(tmp_path / "cache")

    assert seen["list_revision"] == "refs/convert/parquet"
    assert seen["download_revision"] == "refs/convert/parquet"
    assert rows == [{"Series": "[1.0, 2.0]", "Question": "q", "Answer": "a", "Task": "t", "Label": ""}]
