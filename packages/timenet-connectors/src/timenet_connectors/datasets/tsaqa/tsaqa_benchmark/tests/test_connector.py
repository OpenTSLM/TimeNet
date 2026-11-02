from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
from typing import Any, ClassVar

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import TSAQAConnector, records
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import SourceIssue, without_source_issues
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.loader import TSAQADatasetLoader
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
    # Four equal flat patches, so every order of them is right.
    ("train.parquet", 14): ["ambiguous_answer"],
    ("train.parquet", 15): ["repeated_across_splits", "repeated_question"],
    ("train.parquet", 16): ["ambiguous_answer", "conflicting_answer", "repeated_across_splits"],
    ("val.parquet", 0): ["repeated_across_splits", "repeated_question"],
    ("test.parquet", 0): ["ambiguous_answer", "conflicting_answer", "repeated_across_splits"],
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


def test_each_question_format_has_its_task_type():
    dataset = _FixtureConnector().convert(_refs())
    kinds = Counter((type(task).__name__, task.metadata["question_format"]) for task in dataset.iter_tasks())
    assert kinds == {
        ("ClassificationTask", "true_or_false"): 8,
        ("AnswerTask", "multiple_choice"): 8,
        ("AnswerTask", "puzzling"): 3,
    }


def test_issue_filter_leaves_out_the_tasks_with_the_named_issues():
    dataset = _FixtureConnector().convert(_refs())
    clean = list(without_source_issues(dataset.iter_tasks(), issues=[SourceIssue.REPEATED_ACROSS_SPLITS]))
    assert len(clean) == 15


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "has no 'source_issues' list"),
        ({"source_issues": "repeated_question"}, "must list issue names, got 'repeated_question'"),
        ({"source_issues": ["unknown"]}, r"must list issue names, got \['unknown'\]"),
    ],
)
def test_issue_filter_rejects_a_task_whose_issues_are_unknown(metadata, message):
    task = next(iter(_FixtureConnector().convert(_refs()).iter_tasks()))
    clean = without_source_issues([replace(task, metadata=metadata)])
    with pytest.raises(TimeFValidationError, match=message):
        list(clean)


@pytest.mark.parametrize(
    ("issues", "message"),
    [
        ("repeated_question", "got the text 'repeated_question'"),
        # An empty text would otherwise exclude nothing.
        ("", "got the text ''"),
        # An empty YAML value, or a number, is not a collection at all.
        (None, "issues is a collection of issue names, got None"),
        (5, "issues is a collection of issue names, got 5"),
        (["unknown"], "unknown source issue 'unknown'"),
        # One level of nesting too many, as from a config file: not a name, and not hashable.
        ([["repeated_question"]], r"unknown source issue \['repeated_question'\]"),
    ],
)
def test_issue_filter_rejects_issues_that_are_not_issue_names_at_the_call(issues, message):
    with pytest.raises(TimeFValidationError, match=message):
        without_source_issues([], issues=issues)


def _issues_after_rewrite(tmp_path: Path, position: int, printed: str, replacement: str) -> JsonValue:
    """Replace one text in a train row's question, convert the fixtures, and return that row's issues."""
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    question = pq.read_table(tmp_path / "train.parquet").column("question")[position].as_py()
    assert printed in question
    _rewrite(tmp_path / "train.parquet", position, question=question.replace(printed, replacement))
    tasks = _FixtureConnector().convert(_refs(tmp_path)).iter_tasks()
    return next(task for task in tasks if _row_of(task) == ("train.parquet", position)).metadata["source_issues"]


@pytest.mark.parametrize(
    ("printed", "replacement", "issues"),
    [
        # D prints the right choice B's series, so D is as right as B.
        ("D: [1.0069, 0.1787, 1.3382, 0.6093, 1.3713, 0.2449]", "D: [0, 0, 0, 0, 0, 0]", ["ambiguous_answer"]),
        # A prints C's series, but both are wrong, so only B is right.
        (
            "A: [0.4973, 0.1671, -0.2276, -0.7165, -1.2239, -1.7123]",
            "A: [-0.873, -0.4457, 0.8337, 2.7393, 0.8221, -0.3572]",
            [],
        ),
    ],
)
def test_a_series_choice_makes_the_answer_ambiguous_only_when_it_equals_the_right_one(
    tmp_path, printed, replacement, issues
):
    # Train row 11 is a multiple-choice temporal relationship question whose answer B is a flat series.
    assert _issues_after_rewrite(tmp_path, 11, printed, replacement) == issues


def test_a_text_choice_equal_to_the_right_one_makes_the_answer_ambiguous(tmp_path):
    # Train row 5 is a multiple-choice comparison question whose answer is A.
    printed = "B: Lag persistence is high and consistent."
    assert _issues_after_rewrite(tmp_path, 5, printed, "B: There is no persistence in lags.") == ["ambiguous_answer"]


_FLAT_PATCHES = "\n".join(f"{letter}: [0, 0, 0, 0, 0, 0]" for letter in "ABCD")


@pytest.mark.parametrize(
    ("patches", "issues"),
    [
        # Four different patches: only the published order rebuilds the sequence.
        (("1.0", "2.0", "3.0", "4.0"), []),
        # Only A and C are equal: swapping them is a second right order.
        (("1.0", "2.0", "1.0", "4.0"), ["ambiguous_answer"]),
    ],
)
def test_a_puzzle_is_ambiguous_only_when_two_patches_are_equal(tmp_path, patches, issues):
    # Train row 14 is a puzzle whose four patches are all the flat series.
    replacement = "\n".join(
        f"{letter}: [{value}, 0, 0, 0, 0, 0]" for letter, value in zip("ABCD", patches, strict=True)
    )
    assert _issues_after_rewrite(tmp_path, 14, _FLAT_PATCHES, replacement) == issues


def test_each_format_has_its_own_method():
    loader = TSAQADatasetLoader.parse_from(_refs(), inventory=_FIXTURE_INVENTORY)
    assert len(list(loader.true_or_false_tasks())) == 8
    assert len([task for task in loader.multiple_choice_tasks() if task.split is Split.VALIDATION]) == 1
    assert [task.split for task in loader.puzzling_tasks()] == [Split.TRAIN, Split.TRAIN, Split.TEST]
    # tasks() keeps the file order, and holds exactly the tasks of the three methods.
    by_format = [*loader.true_or_false_tasks(), *loader.multiple_choice_tasks(), *loader.puzzling_tasks()]
    assert sorted(map(_row_of, loader.tasks())) == sorted(map(_row_of, by_format))
    assert [_row_of(task) for task in loader.tasks()] == [
        (f"{split}.parquet", row)
        for split in _SPLITS
        for row in range(len(pq.read_table(_FIXTURES / f"{split}.parquet")))
    ]
    assert (ClassificationTask, AnswerTask) == loader.TASK_TYPES


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


def test_a_file_that_names_no_split_fails():
    with pytest.raises(TimeFValidationError, match="'dev' is not a TSAQA split"):
        TSAQADatasetLoader.parse_from([HubFileRef(_FIXTURES / "val.parquet", "dev")], inventory=_FIXTURE_INVENTORY)


def test_a_split_without_inventory_counts_fails():
    inventory = {split: counts for split, counts in _FIXTURE_INVENTORY.items() if split is not SourceSplit.VAL}
    with pytest.raises(TimeFValidationError, match=r"no row counts for the split files \['val'\]"):
        TSAQADatasetLoader.parse_from(_refs(), inventory=inventory)


def test_no_split_file_fails():
    with pytest.raises(TimeFValidationError, match="needs at least one of its split files"):
        TSAQADatasetLoader.parse_from([], inventory=_FIXTURE_INVENTORY)


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
