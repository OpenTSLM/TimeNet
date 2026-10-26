from collections import Counter
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask
from timenet.types.splits import Split
from timenet_connectors.bases.huggingface import HubFileRef
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
    # The ten-fold protocol needs enough distinct signals and category examples.
    rows = list(_FIXTURE_ROWS)
    for index in range(17):
        rows.append(
            {
                "Task": "Trend" if index < 8 else "Volatility",
                "Question": f"Question about synthetic series {index + 3}",
                "Answer": "(a) yes",
                "Series": json.dumps([index + 10.0, index + 11.0]),
            }
        )
    return rows


def _refs(tmp_path, rows):
    # Write the rows to a parquet file shaped like the Hub's converted copy; convert() reads it back.
    path = tmp_path / f"rows-{len(list(tmp_path.iterdir()))}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return [HubFileRef(path, "train")]


def _convert(tmp_path) -> TimeFDataset:
    return TSQAConnector().convert(_refs(tmp_path, _fixture_rows()))


def test_metadata():
    assert TSQAConnector().metadata().dataset_id == "chengsenwang/tsqa"
    assert str(TSQAConnector().metadata().license) == "Apache-2.0"
    assert str(TSQAConnector().metadata().dataset_version) == "1.2.0"


def test_conversion_rejects_too_few_signal_groups(tmp_path):
    with pytest.raises(TimeFValidationError, match=r"10 folds.*found 3"):
        TSQAConnector().convert(_refs(tmp_path, _FIXTURE_ROWS))


def test_convert_builds_one_record_per_row(tmp_path):
    dataset = _convert(tmp_path)
    assert isinstance(dataset, TimeFDataset)
    assert len(dataset.records) == len(_fixture_rows())


def test_qa_tasks_match_fixture(tmp_path):
    dataset = _convert(tmp_path)
    rows = _fixture_rows()
    qa = [t for t in dataset.tasks if isinstance(t, AnswerTask)]
    assert len(qa) == len(rows)
    assert {t.prompt for t in qa} == {r["Question"] for r in rows}
    assert {t.targets[0] for t in qa if t.targets is not None} == {r["Answer"] for r in rows}


def test_series_values_parsed_from_fixture(tmp_path):
    dataset = _convert(tmp_path)
    expected = json.loads(_fixture_rows()[0]["Series"])
    task = next(task for task in dataset.tasks if task.prompt == _fixture_rows()[0]["Question"])
    (record,) = task.inputs
    got = record.signals[0].to_numpy()
    assert len(got) == len(expected)
    assert float(got[0]) == pytest.approx(expected[0], rel=1e-5)


def test_task_annotation_present(tmp_path):
    keys = {ann.key for record in _convert(tmp_path).records for ann in record.annotations}
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


def test_generated_splits_balance_categories_and_keep_signals_together(tmp_path):
    rows = _balanced_rows()
    rows_by_question = {row["Question"]: row for row in rows}
    dataset = TSQAConnector().convert(_refs(tmp_path, rows))
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
            row = rows_by_question[task.prompt]
            signals.add(row["Series"])
            categories[row["Task"]] += 1
            assert task.split == split
        assert seen_signals.isdisjoint(signals)
        seen_signals.update(signals)
        assert categories["Trend"] == categories["Volatility"]


def test_split_assignments_are_repeatable(tmp_path):
    rows = _balanced_rows()
    first = TSQAConnector().convert(_refs(tmp_path, rows))
    second = TSQAConnector().convert(_refs(tmp_path, rows))
    assert [(task.prompt, task.split) for task in first.tasks] == [(task.prompt, task.split) for task in second.tasks]
    assert {task.prompt for task in first.tasks} == {row["Question"] for row in rows}


@pytest.mark.parametrize(
    "series",
    [
        ["[1,2]", "[ 1.0, 2.0 ]", "[[1, 2]]"],
        ["[[1,2],[3,4]]", "[[1.0, 2.0], [3.0, 4.0]]"],
    ],
)
def test_equivalent_series_encodings_share_a_split(series, tmp_path):
    rows = _balanced_rows()
    for encoded in series:
        rows.append({"Series": encoded, "Task": "Trend", "Question": "duplicate", "Answer": "a"})
    dataset = TSQAConnector().convert(_refs(tmp_path, rows))
    duplicates = [task for task in dataset.tasks if task.prompt == "duplicate"]
    assert len(duplicates) == len(series)
    assert len({task.split for task in duplicates}) == 1
    assert duplicates[0].split is not None


def test_splits_survive_storage_round_trip(tmp_path):
    original = TSQAConnector().convert(_refs(tmp_path, _balanced_rows()))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("chengsenwang/tsqa", "1.2.0")) as reader:
        restored = reader.read()
    assert {task.prompt: task.split for task in restored.tasks} == {task.prompt: task.split for task in original.tasks}


def test_source_is_pinned():
    # Literal pins, so moving to another repo, commit, or file is a deliberate test edit.
    assert TSQAConnector.HF_REPO == "ChengsenWang/TSQA"
    assert TSQAConnector.SOURCE_REVISION == "22518209e3ee6302be4b4957f2692b3e8de849c3"
    assert dict(TSQAConnector.SOURCE_FILES) == {"train": "default/train/0000.parquet"}
