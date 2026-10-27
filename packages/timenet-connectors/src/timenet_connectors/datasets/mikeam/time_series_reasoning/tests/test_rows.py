import json
import shutil

import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonMapping, JsonValue
from timenet.types import AnswerTask
from timenet_connectors.datasets.mikeam.time_series_reasoning.issues import (
    SourceIssue,
    without_source_issues,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import TaskKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import (
    SourceFile,
    check_blank_lines,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import (
    CounterfactualRow,
    DescriptionRow,
    QuestionRow,
    Row,
    read_rows,
    reread_values,
)
from timenet_connectors.datasets.mikeam.time_series_reasoning.tests.release_fixtures import fixture_path, published_rows


_ROW_TYPES = {
    SourceFile.SERIES: Row,
    SourceFile.DESCRIPTION_TRAIN: DescriptionRow,
    SourceFile.DESCRIPTION_VAL: DescriptionRow,
    SourceFile.DESCRIPTION_TEST: DescriptionRow,
    SourceFile.QUESTIONS: QuestionRow,
    SourceFile.COUNTERFACTUALS: CounterfactualRow,
}


def _rows(file: SourceFile) -> list[Row]:
    return list(read_rows(fixture_path(file), file, _ROW_TYPES[file]))


# The three description files share one row model, so one of them stands for all.
@pytest.mark.parametrize(
    "file", [SourceFile.SERIES, SourceFile.DESCRIPTION_TRAIN, SourceFile.QUESTIONS, SourceFile.COUNTERFACTUALS]
)
def test_every_row_keeps_its_published_fields(file):
    rows, published = _rows(file), published_rows(file)
    assert len(rows) == len(published)
    for line, (row, source) in enumerate(zip(rows, published, strict=True), start=1):
        assert type(row) is _ROW_TYPES[file]
        assert (row.ref.file, row.ref.line) == (file, line)
        # Texts keep their edge whitespace, and metadata its key order: the authors' prompt prints both.
        for field in ("uuid", "description", "description_short", "description_tiny", "characteristics", "generator"):
            assert getattr(row, field) == source[field]
        assert list(row.metadata.items()) == list(source["metadata"].items())
        assert row.series_shape.length == len(source["series"])
        if isinstance(row, (DescriptionRow, QuestionRow)):
            assert row.options == source["options"]
        if isinstance(row, QuestionRow):
            assert (row.question, row.answer_index) == (source["question"], source["answer_index"])
        if isinstance(row, CounterfactualRow):
            assert (row.category, row.counterfactual) == (source["category"], source["counterfactual"])
            assert row.new_series_shape.length == len(source["new_series"])


def test_a_reread_fails_when_the_file_changed(tmp_path):
    path = tmp_path / SourceFile.SERIES.value
    shutil.copy(fixture_path(SourceFile.SERIES), path)
    row = next(read_rows(path, SourceFile.SERIES, Row))
    lines = path.read_bytes().splitlines(keepends=True)
    changed = json.loads(lines[0])
    changed["series"][0] += 1
    path.write_bytes(json.dumps(changed).encode() + b"\n" + b"".join(lines[1:]))
    with pytest.raises(TimeFFormatError, match="line 1"):
        reread_values(row.ref, "series", row.series_shape)


def test_blank_texts_must_be_on_the_pinned_lines():
    found = frozenset(
        row.ref.line for row in _rows(SourceFile.COUNTERFACTUALS) if isinstance(row, QuestionRow) and row.has_blank_text
    )
    assert found == {3}
    # The fixture's blank row sits on line 3, not on the pinned release's line 45584.
    with pytest.raises(TimeFValidationError, match=r"found \[3\]"):
        check_blank_lines(SourceFile.COUNTERFACTUALS, fixture_path(SourceFile.COUNTERFACTUALS), found)


def _first_published(file: SourceFile) -> dict:
    return published_rows(file)[0]


def _with(source: dict, **changes) -> dict:
    return {**source, **changes}


def _without(source: dict, field: str) -> dict:
    return {key: value for key, value in source.items() if key != field}


_QUESTION = _first_published(SourceFile.QUESTIONS)
_DESCRIPTION = _first_published(SourceFile.DESCRIPTION_TRAIN)
_COUNTERFACTUAL = _first_published(SourceFile.COUNTERFACTUALS)


@pytest.mark.parametrize(
    ("file", "line", "message"),
    [
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, series=[1.0, True])), "finite number"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, series=[1.0, [2.0]])), "finite number"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, series=[])), "nonempty list"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, series=[1.0, 2**53 + 1])), "float64 can't hold"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, series=[1, 2**31])), "int32 can't hold"),
        (SourceFile.QUESTIONS, json.dumps(_QUESTION).replace('"series": [', '"series": [NaN, ', 1), "NaN"),
        (SourceFile.QUESTIONS, json.dumps(_QUESTION).replace('"series": [', '"series": [1e999, ', 1), "finite number"),
        (SourceFile.QUESTIONS, json.dumps(_QUESTION).replace("{", '{"uuid": "x", ', 1), "duplicate key"),
        (SourceFile.QUESTIONS, json.dumps(_without(_QUESTION, "question")), "question"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, extra=1)), "extra"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, ref="x")), "'ref'"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, options=_QUESTION["options"][:3])), "options"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, answer_index=4)), "answer_index"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, answer_index=True)), "answer_index"),
        (SourceFile.COUNTERFACTUALS, json.dumps(_with(_COUNTERFACTUAL, category="contrast")), "category"),
        (
            SourceFile.DESCRIPTION_TRAIN,
            json.dumps(_with(_DESCRIPTION, scenario_category="Health and Medical")),
            "scenario_category",
        ),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, uuid="not-a-uuid")), "uuid"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, description=" ")), "description"),
        (SourceFile.QUESTIONS, json.dumps(_with(_QUESTION, metadata={"units": ""})), "metadata"),
        (SourceFile.QUESTIONS, "[]", "row object"),
        (SourceFile.QUESTIONS, "", "Expecting value"),
        (
            SourceFile.DESCRIPTION_TRAIN,
            json.dumps(_with(_DESCRIPTION, options=[*_DESCRIPTION["options"][1:], _DESCRIPTION["options"][0]])),
            "first option",
        ),
    ],
)
def test_a_malformed_row_is_rejected_with_its_line(tmp_path, file, line, message):
    path = tmp_path / "rows.jsonl"
    path.write_text(f"{line}\n", encoding="utf-8")
    with pytest.raises(TimeFValidationError, match="line 1") as raised:
        list(read_rows(path, file, _ROW_TYPES[file]))
    assert message in str(raised.value)


def _task(issues: list[JsonValue] | None) -> AnswerTask:
    metadata: JsonMapping = {} if issues is None else {TaskKey.SOURCE_ISSUES: issues}
    return AnswerTask(prompt="q", targets=("a",), metadata=metadata)


def test_without_source_issues_leaves_out_the_tasks_with_the_named_issues():
    clean, blank, repeated, description = (
        _task([]),
        _task(["blank_option"]),
        _task(["repeated_option", "answer_text_repeated"]),
        _task(None),
    )
    tasks = [clean, blank, repeated, description]
    # Compare identities: tasks compare equal by value, not by object.
    assert [id(task) for task in without_source_issues(tasks)] == [id(clean), id(description)]
    kept = without_source_issues(tasks, issues=[SourceIssue.ANSWER_TEXT_REPEATED])
    assert [id(task) for task in kept] == [id(clean), id(blank), id(description)]


@pytest.mark.parametrize("found", ["blank_option", ["unknown"]])
def test_without_source_issues_rejects_malformed_metadata(found):
    task = AnswerTask(prompt="q", targets=("a",), metadata={TaskKey.SOURCE_ISSUES: found})
    with pytest.raises(TimeFValidationError, match=TaskKey.SOURCE_ISSUES.value):
        list(without_source_issues([task]))
