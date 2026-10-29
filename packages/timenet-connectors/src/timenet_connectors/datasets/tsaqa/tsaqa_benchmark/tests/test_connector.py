from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import re
import shutil
from typing import Any, ClassVar

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import TSAQAConnector, records
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import SourceIssue, without_source_issues
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.records import record_metadata
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import PINNED_INVENTORY, InventoryKey, SourceSplit
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.request import authors_question
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import (
    QuestionFormat as F,
    QuestionType as Q,
    SourceTask as T,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_SPLITS = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}
# The fixture files' rows, counted as release.check_inventory counts them (tests/release_fixtures.py).
_FIXTURE_INVENTORY: dict[SourceSplit, dict[InventoryKey, int]] = {
    SourceSplit.TRAIN: {
        (T.ANOMALY_DETECTION, Q.TRUE_OR_FALSE, F.TRUE_OR_FALSE): 1,
        (T.CLASSIFICATION, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 2,
        (T.CHARACTERIZATION, Q.TRUE_OR_FALSE, F.TRUE_OR_FALSE): 1,
        (T.CHARACTERIZATION, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 1,
        (T.COMPARISON, Q.TRUE_OR_FALSE, F.TRUE_OR_FALSE): 1,
        (T.COMPARISON, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 1,
        (T.DATA_TRANSFORMATION, Q.TRUE_OR_FALSE, F.TRUE_OR_FALSE): 3,
        (T.DATA_TRANSFORMATION, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 1,
        (T.TEMPORAL_RELATIONSHIP, Q.TRUE_OR_FALSE, F.TRUE_OR_FALSE): 1,
        (T.TEMPORAL_RELATIONSHIP, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 1,
        (T.TEMPORAL_RELATIONSHIP, Q.ORDERING, F.TRUE_OR_FALSE): 1,
        (T.TEMPORAL_RELATIONSHIP, Q.ORDERING, F.MULTIPLE_CHOICE): 1,
        (T.TEMPORAL_RELATIONSHIP, Q.ORDERING, F.PUZZLING): 2,
    },
    SourceSplit.VAL: {(T.CLASSIFICATION, Q.MULTIPLE_CHOICES, F.MULTIPLE_CHOICE): 1},
    SourceSplit.TEST: {(T.TEMPORAL_RELATIONSHIP, Q.ORDERING, F.PUZZLING): 1},
}
# Each fixture row's issues, by file and row, written out from tests/release_fixtures.py.
_EXPECTED_ISSUES = {
    ("train.parquet", 12): ["question_type_mismatch"],
    ("train.parquet", 13): ["question_type_mismatch"],
    ("train.parquet", 15): ["repeated_across_splits", "repeated_question"],
    ("train.parquet", 16): ["conflicting_answer", "repeated_across_splits"],
    ("val.parquet", 0): ["repeated_across_splits", "repeated_question"],
    ("test.parquet", 0): ["conflicting_answer", "repeated_across_splits"],
}
_PRINTED_SERIES = re.compile(r"^(?:[A-D]: )?(\[[-0-9., e]*\])$", re.MULTILINE)


class _FixtureConnector(TSAQAConnector):
    """The connector, pinned to the fixtures' inventory instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_INVENTORY: ClassVar[Mapping[SourceSplit, Mapping[InventoryKey, int]]] = _FIXTURE_INVENTORY


def _refs(directory: Path = _FIXTURES) -> list[HubFileRef]:
    return [HubFileRef(directory / f"{split}.parquet", split) for split in _SPLITS]


def _published() -> dict[tuple[str, int], dict[str, Any]]:
    """Every fixture row as the release publishes it, read with pyarrow alone, by file and row."""
    return {
        (f"{split}.parquet", position): row
        for split in _SPLITS
        for position, row in enumerate(pq.read_table(_FIXTURES / f"{split}.parquet").to_pylist())
    }


def _row_of(task) -> tuple[str, int]:
    return task.metadata["file"], task.metadata["row"]


def _published_series(row: dict[str, Any]) -> list[list[float]]:
    """A row's series in the order a model reads them: ``input_ts``, then each list the question prints."""
    inputs = json.loads(row["input_ts"])
    series = inputs if row["task"] == "comparison" else [inputs]
    if row["task"] in {"data_transformation", "temporal_relationship"}:
        series += [json.loads(text) for text in _PRINTED_SERIES.findall(row["question"])]
    return [[float(value) for value in values] for values in series]


def _check_task(task, row: dict[str, Any]) -> None:
    """Require a task to hold its published row: question, answer, series, context and provenance."""
    true_or_false = row["answer"] in {"T", "F"} and "T: True.\nF: False.\n" in row["question"]
    assert isinstance(task, ClassificationTask if true_or_false else AnswerTask)
    assert task.targets == (row["answer"],)
    assert task.split is _SPLITS[task.metadata["file"].removesuffix(".parquet")]
    assert authors_question(task) == row["question"]
    series = _published_series(row)
    assert [record.signals[0].to_numpy().tolist() for record in task.inputs] == series
    prompt = task.prompt.read()
    first_printed = 3 if row["task"] == "comparison" else 2
    assert re.findall(r"<series (\d+)>", prompt) == [str(n) for n in range(first_printed, len(series) + 1)]
    assert "[" not in prompt or row["task"] not in {"data_transformation", "temporal_relationship"}
    for record in task.inputs:
        assert (record_metadata(record).file, record_metadata(record).row) == _row_of(task)
    context = [(a.key, a.value) for a in task.input_annotations]
    if row["task"] == "comparison":
        assert context == [
            (key, json.loads(row[key])[position]) for position in range(2) for key in ("meta_info", "domain", "dataset")
        ]
    else:
        assert context == [(key, row[key]) for key in ("meta_info", "domain", "dataset")]
    assert task.metadata["task"] == row["task"]
    assert task.metadata["question_type"] == row["question_type"]
    assert task.metadata["source_issues"] == _EXPECTED_ISSUES.get(_row_of(task), [])
    if isinstance(task, ClassificationTask):
        assert task.target_schema == "true_or_false_vocabulary"


def test_every_task_holds_its_published_row():
    dataset = _FixtureConnector().convert(_refs())
    published = _published()
    tasks = list(dataset.iter_tasks())
    assert sorted(_row_of(task) for task in tasks) == sorted(published)
    for task in tasks:
        _check_task(task, published[_row_of(task)])
    assert len(dataset.records) == sum(len(_published_series(row)) for row in published.values())


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert(_refs())
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    published = _published()
    with TimeFReader(registry.open_version("tsaqa/tsaqa-benchmark", "1.0.0")) as reader:
        restored = reader.read()
        assert {a.key: a.value for a in restored.annotations} == {
            "source_repository": "TSAQA/TSAQA-Benchmark",
            "source_revision": "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd",
            "true_or_false_vocabulary": ["T", "F"],
        }
        tasks = list(restored.iter_tasks())
        assert len(tasks) == len(published)
        for task in tasks:
            _check_task(task, published[_row_of(task)])
        specs = Counter(record.signals[0].spec.spec_type for record in restored.records)
    # 19 inputs and the comparisons' second ones, and the temporal relationship series; then the
    # sequences of one Fourier, one wavelet and one first-order difference question, and the four
    # choices of the first-order difference multiple-choice question.
    assert specs == {
        "tsaqa_series": 43,
        "tsaqa_fourier_transform": 1,
        "tsaqa_wavelet_transform": 1,
        "tsaqa_first_order_difference": 5,
    }


def test_task_kinds_and_issue_filter():
    dataset = _FixtureConnector().convert(_refs())
    kinds = Counter((type(task).__name__, task.metadata["question_format"]) for task in dataset.iter_tasks())
    assert kinds == {
        ("ClassificationTask", "true_or_false"): 8,
        ("AnswerTask", "multiple_choice"): 8,
        ("AnswerTask", "puzzling"): 3,
    }
    clean = list(without_source_issues(dataset.iter_tasks(), issues=[SourceIssue.REPEATED_ACROSS_SPLITS]))
    assert len(clean) == 19 - 4


def test_each_split_file_is_read_once_for_all_its_series(monkeypatch):
    reads = Counter()
    read = records.read_series_columns

    def counted(path):
        reads[path.name] += 1
        return read(path)

    monkeypatch.setattr(records, "read_series_columns", counted)
    dataset = _FixtureConnector().convert(_refs())
    for record in dataset.records:
        record.signals[0].to_numpy()
    assert reads == {"train.parquet": 1, "val.parquet": 1, "test.parquet": 1}


def test_declares_the_pinned_release():
    assert TSAQAConnector.HF_REPO == "TSAQA/TSAQA-Benchmark"
    assert TSAQAConnector.SOURCE_REVISION == "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd"
    assert dict(TSAQAConnector.SOURCE_FILES) == {
        "train": "train.parquet",
        "val": "val.parquet",
        "test": "test.parquet",
    }
    # The paper's 70/10/20 split of 210,000 rows, and its question type counts (Section 3.4).
    assert {split: sum(counts.values()) for split, counts in PINNED_INVENTORY.items()} == {
        SourceSplit.TRAIN: 147000,
        SourceSplit.VAL: 21000,
        SourceSplit.TEST: 42000,
    }
    by_type = Counter()
    for counts in PINNED_INVENTORY.values():
        for (_, question_type, _), count in counts.items():
            by_type[question_type] += count
    assert by_type == {Q.TRUE_OR_FALSE: 84632, Q.MULTIPLE_CHOICES: 85479, Q.ORDERING: 39889}


def test_pinned_inventory_mismatch_fails():
    with pytest.raises(TimeFValidationError, match="differ from the pinned release"):
        TSAQAConnector().convert(_refs())


def _rewrite(path: Path, position: int, **columns: str) -> None:
    """Rewrite one row of a fixture file in place, keeping its schema."""
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[position].update(columns)
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), path)


def test_a_reread_that_no_longer_matches_its_row_fails(tmp_path):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    dataset: TimeFDataset = _FixtureConnector().convert(_refs(tmp_path))
    _rewrite(tmp_path / "train.parquet", 0, input_ts="[1.0, 2.0]")
    first = next(record for record in dataset.records if record_metadata(record).file == "train.parquet")
    with pytest.raises(TimeFFormatError, match=r"train.parquet row 0: the series differ"):
        first.signals[0].to_numpy()


@pytest.mark.parametrize(
    ("position", "columns", "message"),
    [
        (0, {"answer": "Z"}, "not one of the question's letters"),
        (0, {"task": "forecasting"}, "is not one of"),
        (0, {"input_ts": "[1.0, NaN]"}, "NaN"),
        (4, {"meta_info": '["only one entry"]'}, "JSON list of 2 entries"),
        (14, {"answer": "A,B,C"}, "a puzzle's answer orders"),
        (6, {"question": "Is it?\n"}, "lists its choices once"),
    ],
)
def test_malformed_rows_name_their_file_and_row(tmp_path, position, columns, message):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    _rewrite(tmp_path / "train.parquet", position, **columns)
    with pytest.raises(TimeFValidationError, match=rf"train.parquet row {position}: .*{message}"):
        _FixtureConnector().convert(_refs(tmp_path))
