from collections import Counter
import json
from pathlib import Path
import sys

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask
from timenet.types.splits import Split
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.chengsenwang.tsqa import TSQAConnector


# Synthetic rows shaped exactly like the HuggingFace Hub rows this connector converts. They are
# hand-written for the tests and not derived from the real TSQA dataset, so the repo ships no dataset
# bytes. Questions and answers are distinct per row; each Series is a JSON string of floats.
_FIXTURE_ROWS = [
    {
        "Task": "Trend",
        "Size": 8,
        "Question": "Does the first synthetic series trend upward?",
        "Answer": "(a) yes",
        "Label": "increasing",
        "Series": json.dumps([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    },
    {
        "Task": "Trend",
        "Size": 8,
        "Question": "Does the second synthetic series trend downward?",
        "Answer": "(b) no",
        "Label": "decreasing",
        "Series": json.dumps([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
    },
    {
        "Task": "Volatility",
        "Size": 8,
        "Question": "Is the third synthetic series flat?",
        "Answer": "(c) constant",
        "Label": "constant",
        "Series": json.dumps([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
]


def _fixture_rows():
    return _FIXTURE_ROWS


def _convert() -> TimeFDataset:
    # The Hub rows are plain dicts; the fixture is exactly that shape, so convert() takes it directly.
    return TSQAConnector().convert(_fixture_rows())


def test_is_a_connector():
    assert isinstance(TSQAConnector(), BaseConnector)
    assert isinstance(TSQAConnector(), BaseHuggingFaceConnector)


def test_metadata():
    assert TSQAConnector().metadata().dataset_id == "chengsenwang/tsqa"
    assert str(TSQAConnector().metadata().license) == "Apache-2.0"
    assert str(TSQAConnector().metadata().dataset_version) == "1.1.0"


def test_convert_builds_one_record_per_row():
    dataset = _convert()
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.records) == len(_fixture_rows())


def test_qa_tasks_match_fixture():
    dataset = _convert()
    rows = _fixture_rows()
    qa = [t for t in dataset.tasks if isinstance(t, AnswerTask)]
    assert len(qa) == len(rows)
    assert {t.prompt for t in qa} == {r["Question"] for r in rows}
    assert {t.targets[0] for t in qa if t.targets is not None} == {r["Answer"] for r in rows}


def test_series_values_parsed_from_fixture():
    dataset = _convert()
    expected = json.loads(_fixture_rows()[0]["Series"])
    record = next(record for record in dataset.records if record.record_id == "row-0")
    got = record.signals[0].to_numpy()
    assert len(got) == len(expected)
    assert float(got[0]) == pytest.approx(expected[0], rel=1e-5)


def test_task_annotation_present():
    keys = {ann.key for record in _convert().records for ann in record.annotations}
    assert "task" in keys


def _balanced_rows():
    rows = []
    for signal in range(20):
        for category in ("Trend", "Volatility"):
            rows.append(
                {
                    "Task": category,
                    "Question": f"{category} question about signal {signal}",
                    "Answer": "(a)",
                    "Series": json.dumps([signal, signal + 1]),
                }
            )
    return rows


def test_generated_splits_balance_categories_and_keep_signals_together():
    rows = _balanced_rows()
    dataset = TSQAConnector().convert(rows)
    partitions = {
        Split.TRAIN: dataset.get_train(),
        Split.VALIDATION: dataset.get_validation(),
        Split.TEST: dataset.get_test(),
    }

    assert {split: len(tasks) for split, tasks in partitions.items()} == {
        Split.TRAIN: 32,
        Split.VALIDATION: 4,
        Split.TEST: 4,
    }
    seen_signals = set()
    for split, tasks in partitions.items():
        signals = set()
        categories = Counter()
        for task in tasks:
            index = int(task.id.removeprefix("qa-"))
            signals.add(rows[index]["Series"])
            categories[rows[index]["Task"]] += 1
            assert task.split == split
        assert seen_signals.isdisjoint(signals)
        seen_signals.update(signals)
        assert categories["Trend"] == categories["Volatility"]


def test_split_assignments_are_repeatable_and_preserve_source_ids():
    rows = _balanced_rows()
    first = TSQAConnector().convert(rows)
    second = TSQAConnector().convert(rows)
    assert [(task.id, task.split) for task in first.tasks] == [(task.id, task.split) for task in second.tasks]
    assert {task.id for task in first.tasks} == {f"qa-{index}" for index in range(len(rows))}
    assert {record.record_id for record in first.records} == {f"row-{index}" for index in range(len(rows))}
    for task in first.tasks:
        index = int(task.id.removeprefix("qa-"))
        assert task.prompt == rows[index]["Question"]
        assert task.inputs[0].record_id == f"row-{index}"


@pytest.mark.parametrize(
    "series",
    [
        ["[1,2]", "[ 1.0, 2.0 ]", "[[1, 2]]"],
        ["[[1,2],[3,4]]", "[[1.0, 2.0], [3.0, 4.0]]"],
    ],
)
def test_equivalent_series_encodings_share_a_split(series):
    rows = _balanced_rows()
    for encoded in series:
        rows.append({"Series": encoded, "Task": "Trend", "Question": "duplicate", "Answer": "a"})
    dataset = TSQAConnector().convert(rows)
    duplicates = [task for task in dataset.tasks if task.prompt == "duplicate"]
    assert len(duplicates) == len(series)
    assert len({task.split for task in duplicates}) == 1
    assert duplicates[0].split is not None


def test_splits_survive_storage_round_trip(tmp_path):
    original = TSQAConnector().convert(_balanced_rows())
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("chengsenwang/tsqa", "1.1.0")) as reader:
        restored = reader.read()
    assert {task.id: task.split for task in restored.tasks} == {task.id: task.split for task in original.tasks}


def test_missing_hf_library_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # `import huggingface_hub` -> ImportError
    with pytest.raises(ImportError, match="huggingface"):
        TSQAConnector().download(Path("cache"))


def test_real_download_reads_auto_converted_parquet(monkeypatch, tmp_path):
    """The real path reads HF's auto-converted parquet branch, not the (CSV-only) main revision."""
    import huggingface_hub  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

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
