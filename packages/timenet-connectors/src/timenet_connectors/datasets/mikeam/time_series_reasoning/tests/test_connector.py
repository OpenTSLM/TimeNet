"""Tests on a small fixture release: real lines of the pinned release, copied byte for byte, in order.

Each case comes from the row with the shortest series that has it:

- ``MCQ_1_TS.jsonl``: lines 14221 (the answer's text is also a wrong option's) and 80150.
- ``MCQ_2_TS.jsonl``: lines 20351 and 20441, two counterfactuals of ``04e3c8fe``, one of whose changed
  series is longer than the series; and 45584, whose question and options are all blank.
- The description files: the rows of ``1b23301e``, ``37ff629a`` (another metadata key order),
  ``fe2aff54`` (a series in integers) and ``8d269711`` (twice, with the same values) in train,
  ``198e1858`` in val and ``10fbcbdc`` in test.
- ``TS_Dataset.jsonl``: the rows of the 9 series the other files name.

The ``TS_Dataset_MCQ/*.json`` files are JSON Lines. Their fixtures end in ``.jsonl``, so the check-json
hook does not parse them as one JSON document.
"""

from collections.abc import Callable
import json
from pathlib import Path
import random
import shutil
from typing import Any

import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import License, Split, Task
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.mikeam.time_series_reasoning import CONNECTOR, TimeSeriesReasoningConnector
from timenet_connectors.datasets.mikeam.time_series_reasoning.connector import (
    _File,
    _Question,
    _Ref,
    _Series,
    _Shape,
    _ValueType,
)


_FIXTURES = Path(__file__).parent / "fixtures"
# Each source file's part name and its fixture, and the file's name in the Hub repository.
_FILES = {
    "series": ("TS_Dataset.jsonl", "TS_Dataset.jsonl"),
    "description_train": ("TS_Dataset_MCQ/train.jsonl", "TS_Dataset_MCQ/train.json"),
    "description_val": ("TS_Dataset_MCQ/val.jsonl", "TS_Dataset_MCQ/val.json"),
    "description_test": ("TS_Dataset_MCQ/test.jsonl", "TS_Dataset_MCQ/test.json"),
    "questions": ("MCQ_1_TS.jsonl", "MCQ_1_TS.jsonl"),
    "counterfactuals": ("MCQ_2_TS.jsonl", "MCQ_2_TS.jsonl"),
}
_UUID = "00000000-0000-0000-0000-000000000000"


class _FixtureConnector(TimeSeriesReasoningConnector):
    """The connector with the card of this dataset, which the base reads."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"


def _refs(root: Path = _FIXTURES) -> list[HubFileRef]:
    return [HubFileRef(root / fixture, part) for part, (fixture, _) in _FILES.items()]


def _convert(root: Path = _FIXTURES) -> TimeFDataset:
    return _FixtureConnector().convert(_refs(root))


def _published(part: str, root: Path = _FIXTURES) -> list[dict[str, Any]]:
    """Return a fixture's rows as published, decoded without the connector."""
    return [json.loads(line) for line in (root / _FILES[part][0]).read_bytes().splitlines()]


def _copy_fixtures(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    shutil.copytree(_FIXTURES, root)
    return root


def _rewrite(root: Path, part: str, line: int, change: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite one fixture line, counted from 1, with ``change`` applied to its decoded row."""
    rows = _published(part, root)
    change(rows[line - 1])
    (root / _FILES[part][0]).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _replace_line(root: Path, part: str, line: int, text: str) -> None:
    path = root / _FILES[part][0]
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[line - 1] = text + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def _series_of(task: Task) -> list[list[int | float]]:
    """Return the values of each of a task's input Records, read from the files."""
    return [signal.to_numpy().tolist() for record in task.inputs for signal in record.signals]


def _letter(text: str, options: list[str]) -> str:
    return "ABCD"[options.index(text)]


def test_declares_the_pinned_source_files():
    assert CONNECTOR is TimeSeriesReasoningConnector
    assert _FixtureConnector().metadata().license is License.MIT
    assert TimeSeriesReasoningConnector.HF_REPO == "mikeam/time-series-reasoning"
    assert TimeSeriesReasoningConnector.SOURCE_REVISION == "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0"
    assert dict(TimeSeriesReasoningConnector.SOURCE_FILES) == {part: hub for part, (_, hub) in _FILES.items()}


def test_the_dataset_holds_the_provenance_and_the_expected_counts():
    dataset = _convert()
    assert {annotation.key: annotation.value for annotation in dataset.annotations} == {
        "source_repository": "mikeam/time-series-reasoning",
        "source_revision": "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0",
    }
    # 9 series, 4 distinct rounded copies in the description files, and 3 changed series.
    assert len(dataset.records) == 16
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == 12
    # The description tasks keep the authors' split; the question tasks have none.
    assert [task.split for task in tasks] == [Split.TRAIN] * 5 + [Split.VALIDATION, Split.TEST] + [None] * 5
    # A prompt holds all the authors show a model, so no annotation is a task's input.
    assert all(task.input_annotations == () for task in tasks)


def test_a_series_record_holds_the_published_values_texts_and_place():
    records = list(_convert().records)[:9]
    for line, (record, source) in enumerate(zip(records, _published("series"), strict=True), start=1):
        (signal,) = record.signals
        assert signal.to_numpy().tolist() == source["series"]
        assert signal.n_values == len(source["series"])
        assert signal.spec.dtype == ("int32" if all(type(value) is int for value in source["series"]) else "float64")
        assert record.metadata == {"uuid": source["uuid"], "file": "TS_Dataset.jsonl", "line": line}
        assert {annotation.key: annotation.value for annotation in record.annotations} == {
            "description": source["description"],
            "description_short": source["description_short"],
            "description_tiny": source["description_tiny"],
            "characteristics": source["characteristics"],
            "generator": source["generator"],
            "metadata": str(source["metadata"]),
        }


def test_a_changed_series_record_holds_its_values_change_and_generator():
    records = list(_convert().records)[13:]
    published = _published("counterfactuals")
    # 04e3c8fe has two changed series, one for each counterfactual; line 2's is longer than the series.
    for record, line in zip(records, (1, 2, 3), strict=True):
        source = published[line - 1]
        (signal,) = record.signals
        assert signal.to_numpy().tolist() == source["new_series"]
        assert record.metadata == {"uuid": source["uuid"], "file": "MCQ_2_TS.jsonl", "line": line}
        assert {annotation.key: annotation.value for annotation in record.annotations} == {
            "counterfactual": source["counterfactual"],
            "generator": source["new_generator"],
        }


def test_a_description_task_reads_its_rounded_copy_or_else_its_series_record():
    tasks = [task for task in _convert().iter_tasks() if task.split is not None]
    published = [
        row for part in ("description_train", "description_val", "description_test") for row in _published(part)
    ]
    for task, row in zip(tasks, published, strict=True):
        assert _series_of(task) == [row["series"]]
        assert task.metadata["scenario_category"] == row["scenario_category"]
    records = {id(task.inputs[0]) for task in tasks}
    # Train lines 1 and 5 repeat 8d269711 with the same values, so they share one copy. Lines 3 and 4 hold
    # their series' values, so they use its own Record. Train line 2, val and test each have a copy of their own.
    assert len(records) == 6
    assert tasks[0].inputs[0] is tasks[4].inputs[0]
    assert tasks[2].inputs[0].metadata["file"] == "TS_Dataset.jsonl"
    assert tasks[3].inputs[0].metadata["file"] == "TS_Dataset.jsonl"
    assert tasks[1].inputs[0].metadata["file"] == "TS_Dataset_MCQ/train.json"


def _authors_item(data: dict[str, Any], config: dict[str, Any], shuffle: Callable[[list[str]], None]) -> dict[str, Any]:
    """``MultimodalMCQDataset.__getitem__`` (src/models/tasks/multimodal.py, lines 79-125), line for line.

    ``shuffle`` stands in for the module-level ``random.shuffle`` the authors call unseeded, so a test
    can give it the draw the connector made. The defaults are the class's.
    """
    options_column = config.get("options_column", "options")
    label_column = config.get("label_column", "source")
    contrastive_column = config.get("contrastive_column")
    options = data[options_column]

    context = "\n".join([str(data[col]) for col in config["context_columns"]])
    context = config["context_prefix"] + context

    if config["shuffle_labels"]:
        shuffle(options)

    if "answer_index" in data:  # ruff: ignore[if-else-block-instead-of-if-exp] - the authors' code, line for line
        label_index = data["answer_index"]
    else:
        label_index = options.index(data[label_column])

    if config["format_abc_mcq"]:
        full_options = [f"{chr(ord('A') + i)}) {x}" for i, x in enumerate(options)]
        options = [f"{chr(ord('A') + i)}" for i, x in enumerate(options)]
        context = context + "\n" + "\n".join(full_options)
        answer = options[label_index]
    else:
        answer = chr(ord("A") + label_index)

    if contrastive_column:  # ruff: ignore[if-else-block-instead-of-if-exp] - the authors' code, line for line
        ts = data[config["ts_column"]] + data[contrastive_column]
    else:
        ts = data[config["ts_column"]]

    return {"context": context, "ts": ts, "label": answer}


# The authors' configs, configs/tasks/llms_and_ts/*.yaml at d4e748cb.
_QUESTION_CONFIG = {
    "ts_column": "series",
    "context_prefix": "Answer the following question about the time series",
    "context_columns": ["description_tiny", "metadata", "question"],
    "shuffle_labels": False,
    "format_abc_mcq": True,
}
_CONFIGS = {
    "description": {
        "ts_column": "series",
        "label_column": "description",
        "context_columns": [],
        "context_prefix": "What description best relates to this time series?",
        "shuffle_labels": True,
        "format_abc_mcq": True,
    },
    "questions": _QUESTION_CONFIG,
    "counterfactuals": _QUESTION_CONFIG | {"contrastive_column": "new_series"},
}


def test_every_task_is_the_item_the_authors_loader_builds():
    tasks = list(_convert().iter_tasks())
    rows = [
        (kind, row)
        for kind, parts in (
            ("description", ("description_train", "description_val", "description_test")),
            ("questions", ("questions",)),
            ("counterfactuals", ("counterfactuals",)),
        )
        for part in parts
        for row in _published(part)
    ]
    for task, (kind, row) in zip(tasks, rows, strict=True):
        # The connector draws the description order from this seed. Any draw is one the authors' loader can make.
        seed = f"{task.metadata['file']}:{task.metadata['line']}"
        draw = random.Random(seed).shuffle  # ruff: ignore[suspicious-non-cryptographic-random-usage] - a shuffle, not cryptography
        item = _authors_item(row, _CONFIGS[kind], draw)
        assert read_value(task.prompt) == item["context"]
        assert tuple(read_value(target) for target in task.targets or ()) == (item["label"],)
        # A two-series question passes the series and its changed version as one list.
        assert [value for series in _series_of(task) for value in series] == item["ts"]


def test_a_description_target_is_the_letter_of_the_description_in_the_seeded_order():
    descriptions = [task for task in _convert().iter_tasks() if task.split is not None]
    # Published, every answer is the first option. The seeded draw is pinned here: a change to it, or to
    # Python's shuffle, shows as a changed target.
    assert [task.targets for task in descriptions] == [(letter,) for letter in "DBDCABA"]


def test_question_tasks_keep_the_answer_and_the_source_issues_in_metadata():
    tasks = [task for task in _convert().iter_tasks() if task.split is None]
    questions, counterfactuals = _published("questions"), _published("counterfactuals")
    # MCQ_1_TS.jsonl line 14221: the answer's text is also a wrong option's. The letter keeps the authors' choice.
    assert tasks[0].metadata == {
        "file": "MCQ_1_TS.jsonl",
        "line": 1,
        "answer_index": 0,
        "answer_text": questions[0]["options"][0],
        "source_issues": ["repeated_option", "answer_text_repeated"],
    }
    assert tasks[1].metadata["source_issues"] == []
    assert [task.metadata["category"] for task in tasks[2:]] == ["difference"] * 3
    # MCQ_2_TS.jsonl line 45584: the question and every option are blank.
    assert tasks[4].metadata == {
        "file": "MCQ_2_TS.jsonl",
        "line": 3,
        "answer_index": 0,
        "answer_text": "",
        "source_issues": ["blank_question", "blank_option", "repeated_option", "answer_text_repeated"],
        "category": "difference",
    }
    assert counterfactuals[2]["question"] == ""


def _output(dataset: TimeFDataset) -> dict[str, Any]:
    """Return what a dataset holds, comparable before and after storage."""

    def values(record: Record) -> list[Any]:
        return [signal.to_numpy().tolist() for signal in record.signals]

    return {
        "annotations": {annotation.key: annotation.value for annotation in dataset.annotations},
        "records": [
            (
                record.metadata,
                values(record),
                {annotation.key: read_value(annotation.value) for annotation in record.annotations},
            )
            for record in dataset.records
        ],
        "tasks": [
            (read_value(task.prompt), [read_value(t) for t in task.targets or ()], task.split, task.metadata)
            for task in dataset.iter_tasks()
        ],
    }


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _convert()
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("mikeam/time-series-reasoning", "1.0.0")) as reader:
        restored = reader.read()
    # The writer read every lazy signal and the task stream already, so this pass reads them again.
    assert _output(restored) == _output(original)


@pytest.mark.parametrize("missing", ["series", "counterfactuals"])
def test_every_file_is_required(missing):
    refs = [ref for ref in _refs() if ref.part != missing]
    with pytest.raises(TimeFValidationError, match="missing"):
        _FixtureConnector().convert(refs)


def _set(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def change(row: dict[str, Any]) -> None:
        row[key] = value

    return change


def _delete(key: str) -> Callable[[dict[str, Any]], None]:
    def change(row: dict[str, Any]) -> None:
        del row[key]

    return change


def _set_first_series_value(value: object) -> Callable[[dict[str, Any]], None]:
    def change(row: dict[str, Any]) -> None:
        row["series"][0] = value

    return change


def _reorder_metadata(row: dict[str, Any]) -> None:
    row["metadata"] = dict(reversed(row["metadata"].items()))


def _set_description_and_its_option(row: dict[str, Any]) -> None:
    # A description row's first option is its description, so both change together.
    row["description"] = row["options"][0] = "changed"


def _cut_options(row: dict[str, Any]) -> None:
    row["options"] = row["options"][:3]


def _rotate_options(row: dict[str, Any]) -> None:
    row["options"] = [*row["options"][1:], row["options"][0]]


@pytest.mark.parametrize(
    ("part", "change", "message"),
    [
        # A series must be a nonempty list of finite numbers that its dtype holds exactly.
        ("series", _set("series", []), "nonempty list"),
        ("series", _set("series", "1,2"), "nonempty list"),
        ("series", _set_first_series_value(True), "finite number"),
        ("series", _set_first_series_value([2.0]), "finite number"),
        ("series", _set_first_series_value("1"), "finite number"),
        ("series", _set_first_series_value(1e999), "finite number"),
        ("series", _set_first_series_value(2**53 + 1), "float64 cannot hold"),
        ("series", _set("series", [1, 2**31]), "int32 cannot hold"),
        # A row needs each field, and its texts must not be blank.
        ("series", _delete("generator"), "missing field 'generator'"),
        ("series", _set("description", " "), "description must be nonblank text"),
        ("series", _set("uuid", 7), "uuid must be nonblank text"),
        ("series", _set("metadata", {"units": ""}), "metadata must map"),
        ("series", _set("metadata", []), "metadata must map"),
        ("questions", _delete("question"), "missing field 'question'"),
        ("questions", _set("question", None), "question must be nonblank text"),
        ("questions", _cut_options, "options must be 4 texts"),
        ("questions", _set("options", ["a", "b", "c", 4]), "options must be 4 texts"),
        ("questions", _set("answer_index", 4), "answer_index must be 0 to 3"),
        ("questions", _set("answer_index", True), "answer_index must be 0 to 3"),
        ("questions", _set("answer_index", "1"), "answer_index must be 0 to 3"),
        ("counterfactuals", _set("category", "contrast"), "category must be one of"),
        ("counterfactuals", _set("counterfactual", ""), "counterfactual must be nonblank text"),
        ("counterfactuals", _set("new_series", []), "new_series: a series must be"),
        ("description_train", _set("scenario_category", "Health and Medical"), "scenario_category must be one of"),
        ("description_train", _rotate_options, "first option is not the row's description"),
        ("description_train", _set("options", ["", "b", "c", "d"]), "options must be 4 texts"),
    ],
)
def test_a_malformed_row_is_rejected_with_its_file_and_line(tmp_path, part, change, message):
    root = _copy_fixtures(tmp_path)
    _rewrite(root, part, 1, change)
    with pytest.raises(TimeFValidationError, match=r"\(line 1\)") as raised:
        _convert(root)
    assert message in str(raised.value)
    assert _FILES[part][0].rsplit("/", maxsplit=1)[-1] in str(raised.value)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("", "Expecting value"),
        ("[]", "expected a row object"),
        ('{"uuid": "x", "uuid": "y"}', "duplicate key 'uuid'"),
        ("{", "Expecting"),
    ],
)
def test_a_line_that_is_not_one_json_object_is_rejected(tmp_path, line, message):
    root = _copy_fixtures(tmp_path)
    _replace_line(root, "questions", 2, line)
    with pytest.raises(TimeFValidationError, match=r"\(line 2\)") as raised:
        _convert(root)
    assert message in str(raised.value)


def test_a_published_duplicate_field_is_rejected_not_overwritten(tmp_path):
    root = _copy_fixtures(tmp_path)
    path = root / _FILES["series"][0]
    path.write_text(path.read_text().replace("{", '{"uuid": "x", ', 1), encoding="utf-8")
    with pytest.raises(TimeFValidationError, match="duplicate key 'uuid'"):
        _convert(root)


def _change_value(row: dict[str, Any]) -> None:
    row["series"][0] += 1


@pytest.mark.parametrize(
    ("part", "change", "message"),
    [
        ("questions", _set("description", "changed"), r"\['description'\] differ"),
        ("questions", _reorder_metadata, r"\['metadata'\] differ"),
        ("questions", _change_value, r"\['series'\] differ"),
        ("questions", _set("uuid", _UUID), r"is not in TS_Dataset\.jsonl"),
        ("counterfactuals", _change_value, r"\['series'\] differ"),
        ("description_val", _set_description_and_its_option, r"\['description'\] differ"),
        ("description_val", _set("uuid", _UUID), r"is not in TS_Dataset\.jsonl"),
    ],
)
def test_a_row_must_repeat_its_series_row(tmp_path, part, change, message):
    root = _copy_fixtures(tmp_path)
    _rewrite(root, part, 1, change)
    with pytest.raises(TimeFValidationError, match=message):
        _convert(root)


def test_a_description_row_may_hold_other_values_than_its_series_row(tmp_path):
    root = _copy_fixtures(tmp_path)
    # Train line 3 holds the same values as its series' row, until they change.
    _rewrite(root, "description_train", 3, _change_value)
    assert len(_convert(root).records) == 17


def test_a_series_uuid_must_not_repeat(tmp_path):
    root = _copy_fixtures(tmp_path)
    path = root / _FILES["series"][0]
    first = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(path.read_bytes() + first)
    with pytest.raises(TimeFValidationError, match="repeats line 1"):
        _convert(root)


def test_the_rows_of_one_counterfactual_must_hold_one_changed_series(tmp_path):
    root = _copy_fixtures(tmp_path)
    path = root / _FILES["counterfactuals"][0]
    row = _published("counterfactuals", root)[0]
    row["new_series"][0] += 1
    path.write_bytes(path.read_bytes() + json.dumps(row).encode() + b"\n")
    with pytest.raises(TimeFValidationError, match=r"\(line 4\).*differs from line 1's"):
        _convert(root)


def test_a_reread_fails_when_a_series_value_changed(tmp_path):
    root = _copy_fixtures(tmp_path)
    dataset = _convert(root)
    _rewrite(root, "series", 1, _change_value)
    (signal,) = dataset.records[0].signals
    with pytest.raises(TimeFFormatError, match=r"TS_Dataset\.jsonl \(line 1\): series differs.*the file changed"):
        signal.to_numpy()


def test_a_reread_fails_when_a_series_changed_length(tmp_path):
    root = _copy_fixtures(tmp_path)
    dataset = _convert(root)
    _rewrite(root, "counterfactuals", 3, lambda row: row["new_series"].pop())
    changed = dataset.records[-1]
    with pytest.raises(TimeFFormatError, match=r"MCQ_2_TS\.jsonl \(line 3\): new_series differs"):
        changed.signals[0].to_numpy()


def test_a_reread_fails_when_a_series_is_no_longer_a_series(tmp_path):
    root = _copy_fixtures(tmp_path)
    dataset = _convert(root)
    _rewrite(root, "series", 1, _set("series", "gone"))
    with pytest.raises(TimeFFormatError, match="the file changed"):
        dataset.records[0].signals[0].to_numpy()


@pytest.mark.parametrize(
    ("part", "line"),
    [("questions", 1), ("counterfactuals", 2), ("description_train", 2), ("description_test", 1)],
)
def test_the_tasks_fail_when_a_task_file_changed_after_conversion(tmp_path, part, line):
    root = _copy_fixtures(tmp_path)
    dataset = _convert(root)
    # A changed option text keeps every row valid, so only the file's digest tells.
    _rewrite(root, part, line, lambda row: row["options"].__setitem__(1, row["options"][1] + " (edited)"))
    with pytest.raises(TimeFFormatError, match="the file changed since conversion"):
        list(dataset.iter_tasks())


def test_a_rounded_copy_must_have_the_length_of_its_series(tmp_path):
    root = _copy_fixtures(tmp_path)
    _rewrite(root, "description_train", 2, lambda row: row["series"].pop())
    with pytest.raises(TimeFValidationError, match=r"\(line 2\): a rounded copy has 9 values, its series' row 10"):
        _convert(root)


@pytest.mark.parametrize(
    ("part", "change"),
    [
        ("questions", _set("uuid", _UUID)),
        ("questions", _set("answer_index", 9)),
        ("counterfactuals", _set("counterfactual", "What if nothing was written?")),
        ("description_val", _set("scenario_category", "Unknown")),
    ],
)
def test_a_task_row_that_no_longer_reads_as_converted_fails_as_a_changed_file(tmp_path, part, change):
    root = _copy_fixtures(tmp_path)
    dataset = _convert(root)
    _rewrite(root, part, 1, change)
    with pytest.raises(TimeFFormatError, match="the file changed since conversion"):
        list(dataset.iter_tasks())


def _question(question: str, options: list[str], answer_index: int) -> _Question:
    series = _Series(_Ref(Path("rows.jsonl"), _File.QUESTIONS, 1, 0), "u", {}, _Shape(1, _ValueType.INT32, "d"))
    return _Question(series, question, options, answer_index)


@pytest.mark.parametrize(
    ("question", "options", "answer_index", "issues"),
    [
        ("q", ["a", "b", "c", "d"], 0, []),
        (" ", ["a", "b", "c", "d"], 0, ["blank_question"]),
        ("q", ["a", "b", "c", ""], 0, ["blank_option"]),
        ("q", ["a", "b", "b", "d"], 0, ["repeated_option"]),
        ("q", ["a", "b", "a", "d"], 0, ["repeated_option", "answer_text_repeated"]),
        ("q", ["a", "b", "a", "d"], 1, ["repeated_option"]),
    ],
)
def test_each_source_issue_is_found_on_its_own(question, options, answer_index, issues):
    assert [issue.value for issue in _question(question, options, answer_index).issues] == issues
