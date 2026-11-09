from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
from typing import Any

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetConnectorError
from timenet.json import JsonValue
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import connector
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.connector import (
    SourceIssue,
    TSAQAConnector,
    without_source_issues,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_SPLITS = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}
# Each fixture row's issues, by file and row, written out from tests/README.md.
_EXPECTED_ISSUES = {
    # Rows 10 and 12 ask the same question about the same series, and only row 12 has another question_type.
    ("train.parquet", 10): ["repeated_question"],
    ("train.parquet", 12): ["question_type_mismatch", "repeated_question"],
    ("train.parquet", 13): ["question_type_mismatch"],
    # Four equal flat patches, so every order of them is right.
    ("train.parquet", 14): ["ambiguous_answer"],
    ("train.parquet", 15): ["repeated_across_splits", "repeated_question"],
    ("train.parquet", 16): ["ambiguous_answer", "conflicting_answer", "repeated_across_splits"],
    ("val.parquet", 0): ["repeated_across_splits", "repeated_question"],
    ("test.parquet", 0): ["ambiguous_answer", "conflicting_answer", "repeated_across_splits"],
    # The input and the sequence are all zeros, so the sequence is the first-order difference: not "F".
    ("train.parquet", 17): ["wrong_transform_target"],
}
_PRINTED_SERIES = re.compile(r"^(?:[A-D]: )?(\[[-0-9., e]*\])$", re.MULTILINE)


class _FixtureConnector(TSAQAConnector):
    """The connector with the card of this dataset, which the base reads."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"


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
    series = _published_series(row)
    assert [record.signals[0].to_numpy().tolist() for record in task.inputs] == series
    prompt = task.prompt.read()
    first_printed = 3 if row["task"] == "comparison" else 2
    assert re.findall(r"<series (\d+)>", prompt) == [str(n) for n in range(first_printed, len(series) + 1)]
    assert "[" not in prompt or row["task"] not in {"data_transformation", "temporal_relationship"}
    for record in task.inputs:
        assert (record.metadata["file"], record.metadata["row"]) == _row_of(task)
    keys = ("meta_info", "domain", "dataset")
    published = [("task", row["task"]), ("question_type", row["question_type"])]
    # The model reads the context, then the published task and question_type.
    context = [(a.key, a.value) for a in task.input_annotations]
    if row["task"] in {"comparison", "data_transformation", "temporal_relationship"}:
        # No source says what each entry describes, so the lists stay whole, under published_* keys.
        context_values = [(f"published_{key}", json.loads(row[key])) for key in keys]
    else:
        context_values = [(key, row[key]) for key in keys]
    assert context == context_values + published
    # The context is on the first input Record only, and nothing is left on the task.
    first, *others = task.inputs
    assert [(a.key, a.value) for a in first.annotations] == context_values
    assert all(not record.annotations for record in others)
    assert not task.annotations
    assert task.metadata["task"] == row["task"]
    assert task.metadata["question_type"] == row["question_type"]
    assert task.metadata["source_issues"] == _EXPECTED_ISSUES.get(_row_of(task), [])
    if isinstance(task, ClassificationTask):
        assert (task.label_set.name, task.label_set.names) == ("true_or_false", ("T", "F"))


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
        assert {a.key: a.value for a in restored.annotations if a.key not in {"task", "question_type"}} == {
            "source_repository": "TSAQA/TSAQA-Benchmark",
            "source_revision": "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd",
        }
        labels = [(a.key, a.value) for a in restored.annotations if a.key in {"task", "question_type"}]
        expected = {("task", row["task"]) for row in published.values()}
        expected |= {("question_type", row["question_type"]) for row in published.values()}
        assert len(labels) == len(expected)
        assert all(label in labels for label in expected)
        tasks = list(restored.iter_tasks())
        assert len(tasks) == len(published)
        for task in tasks:
            _check_task(task, published[_row_of(task)])
        specs = Counter(record.signals[0].spec.spec_type for record in restored.records)
    # 20 inputs and the comparisons' second ones, and the temporal relationship series; then the
    # sequences of one Fourier, one wavelet and one first-order difference question, and the four
    # choices of the first-order difference multiple-choice question.
    assert specs == {
        "tsaqa_series": 44,
        "tsaqa_fourier_transform": 1,
        "tsaqa_wavelet_transform": 1,
        "tsaqa_first_order_difference": 6,
    }


def test_each_question_format_has_its_task_type():
    dataset = _FixtureConnector().convert(_refs())
    kinds = Counter((type(task).__name__, task.metadata["question_format"]) for task in dataset.iter_tasks())
    assert kinds == {
        ("ClassificationTask", "true_or_false"): 9,
        ("AnswerTask", "multiple_choice"): 8,
        ("AnswerTask", "puzzling"): 3,
    }


def test_issue_filter_leaves_out_the_tasks_with_the_named_issues():
    dataset = _FixtureConnector().convert(_refs())
    clean = list(without_source_issues(dataset.iter_tasks(), issues=[SourceIssue.REPEATED_ACROSS_SPLITS]))
    assert len(clean) == 16


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


def test_reading_the_records_in_order_reads_each_split_file_once(monkeypatch):
    reads = Counter()
    read = connector.pq.read_table

    def counted(path, *args, **kwargs):
        reads[path.name] += 1
        return read(path, *args, **kwargs)

    monkeypatch.setattr(connector.pq, "read_table", counted)
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


def test_a_file_that_names_no_split_fails():
    with pytest.raises(TimeFValidationError, match="has no source part dev"):
        _FixtureConnector().convert([HubFileRef(_FIXTURES / "val.parquet", "dev")])


def test_no_split_file_fails():
    with pytest.raises(TimeFValidationError, match="nonempty"):
        _FixtureConnector().convert([])


def _rewrite(path: Path, position: int, **columns: str) -> None:
    """Rewrite one row of a fixture file in place, keeping its schema."""
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[position].update(columns)
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), path)


def _bump_first_value(series: str) -> str:
    """Return a series with its first value changed and its length kept."""
    values = json.loads(series)
    values[0] += 1.0
    return json.dumps(values)


def _bump_first_printed_value(question: str) -> str:
    """Return a question whose first printed series has its first value changed and its length kept."""
    match = _PRINTED_SERIES.search(question)
    assert match is not None
    return question[: match.start(1)] + _bump_first_value(match[1]) + question[match.end(1) :]


@pytest.mark.parametrize(
    ("position", "rewrite"),
    [
        (0, lambda row: {"input_ts": "[1.0, 2.0]"}),
        # The lengths stay, so only the values tell the file changed.
        (0, lambda row: {"input_ts": _bump_first_value(row["input_ts"])}),
        (6, lambda row: {"question": _bump_first_printed_value(row["question"])}),
    ],
    ids=["shorter-series", "changed-input-value", "changed-printed-value"],
)
def test_a_reread_that_no_longer_matches_its_row_fails(tmp_path, position, rewrite):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    dataset: TimeFDataset = _FixtureConnector().convert(_refs(tmp_path))
    _rewrite(tmp_path / "train.parquet", position, **rewrite(_published()["train.parquet", position]))
    first = next(
        record
        for record in dataset.records
        if (record.metadata["file"], record.metadata["row"]) == ("train.parquet", position)
    )
    with pytest.raises(TimeFFormatError, match=rf"train.parquet row {position}: the series differ"):
        first.signals[0].to_numpy()


@pytest.mark.parametrize(
    ("position", "columns", "message"),
    [
        (0, {"answer": "Z"}, "does not fit the question's letters"),
        (0, {"input_ts": "[1.0, NaN]"}, "finite numbers"),
        (4, {"meta_info": '["only one entry"]'}, "JSON list of 2 texts"),
        (9, {"domain": '["energy"]'}, "context lists differ in length"),
        (14, {"answer": "A,B,C"}, "does not fit"),
        (6, {"question": "Is it?\n"}, "offers A to D in order"),
    ],
)
def test_malformed_rows_name_their_file_and_row(tmp_path, position, columns, message):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    _rewrite(tmp_path / "train.parquet", position, **columns)
    with pytest.raises(TimeFValidationError, match=rf"train.parquet row {position}: .*{message}"):
        _FixtureConnector().convert(_refs(tmp_path))


def test_a_text_choice_that_looks_like_a_series_stays_in_the_prompt(tmp_path):
    # Four characterization questions of the release have choices such as "[13, 17]": intervals, not series.
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    rows = pq.read_table(tmp_path / "train.parquet").to_pylist()
    position = next(i for i, row in enumerate(rows) if row["task"] == "characterization" and "\nB: " in row["question"])
    question = re.sub(r"\nB: .*", "\nB: [13, 17]", rows[position]["question"])
    _rewrite(tmp_path / "train.parquet", position, question=question)
    tasks = _FixtureConnector().convert(_refs(tmp_path)).iter_tasks()
    task = next(task for task in tasks if _row_of(task) == ("train.parquet", position))
    assert task.prompt is not None
    assert task.prompt.read() == question
    assert len(task.inputs) == 1


def _position(tmp_path: Path, wanted) -> tuple[int, dict[str, Any]]:
    """Copy the fixtures and return the position and row of the first train row that ``wanted`` accepts."""
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    rows = pq.read_table(tmp_path / "train.parquet").to_pylist()
    return next((i, row) for i, row in enumerate(rows) if wanted(row))


def test_a_true_or_false_question_without_its_final_newline_fails(tmp_path):
    position, row = _position(tmp_path, lambda row: row["task"] == "characterization" and "T: True." in row["question"])
    _rewrite(tmp_path / "train.parquet", position, question=row["question"].removesuffix("\n"))
    with pytest.raises(TimeFValidationError, match=rf"train.parquet row {position}: .*offers A to D in order"):
        _FixtureConnector().convert(_refs(tmp_path))


def test_a_printed_series_that_loses_its_line_fails_naming_the_labels(tmp_path):
    position, row = _position(
        tmp_path, lambda row: row["task"] == "temporal_relationship" and "\nB: [" in row["question"]
    )
    # A trailing space makes the line something else than a printed series.
    question = re.sub(r"(\nB: \[[^\n]*\])", r"\1 ", row["question"])
    _rewrite(tmp_path / "train.parquet", position, question=question)
    with pytest.raises(TimeFValidationError, match=rf"train.parquet row {position}: .*prints series labelled"):
        _FixtureConnector().convert(_refs(tmp_path))


def test_a_conversion_needs_every_split_file():
    refs = [ref for ref in _refs() if ref.part == "test"]
    with pytest.raises(TimeFValidationError, match=r"missing \['train', 'val'\]"):
        _FixtureConnector().convert(refs)


def test_a_file_that_is_not_parquet_fails_as_a_connector_error(tmp_path):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    (tmp_path / "test.parquet").write_text("not parquet")
    with pytest.raises(TimeNetConnectorError, match=r"test\.parquet"):
        _FixtureConnector().convert(_refs(tmp_path))


def test_a_zero_input_whose_zero_sequence_is_called_the_transform_is_not_flagged(tmp_path):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    _rewrite(tmp_path / "train.parquet", 17, answer="T")
    tasks = _FixtureConnector().convert(_refs(tmp_path)).iter_tasks()
    assert next(task for task in tasks if _row_of(task) == ("train.parquet", 17)).metadata["source_issues"] == []


def test_unknown_published_task_names_its_source_field(tmp_path):
    shutil.copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    _rewrite(tmp_path / "train.parquet", 0, task="forecasting")
    with pytest.raises(TimeNetConnectorError, match=r"train.parquet row 0: task:"):
        _FixtureConnector().convert(_refs(tmp_path))
