import json
from pathlib import Path
import shutil
from typing import Any, NamedTuple

import h5py
import numpy as np
import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Task
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.pandalin98.enginemt_qa import EngineMTQAConnector
from timenet_connectors.datasets.pandalin98.enginemt_qa.connector import _ClosedQuestion, _FileCounts, _Part, _Stage


_FIXTURES = Path(__file__).parent / "fixtures"
_TRAIN, _TEST, _SIGNALS = "train_qa.jsonl", "test_qa.jsonl", "time_series_data.h5"
_STEPS = 600
_CHANNELS = 33
_N_ROWS = 37_669
"""The largest row id that a fixture line names, so the fixture file has just enough rows."""
_MULTI_TRAIN = (279, 287, 295, 305, 310, 319, 327, 335, 340, 346)
_MULTI_TEST = (279, 287, 295, 304, 309, 319, 326, 334, 339, 346)
_NAMED_ROWS = sorted({278, 8435, 20, 37669, *_MULTI_TRAIN, *_MULTI_TEST})

# See tests/README.md for where each fixture line comes from.
_COUNTS = {
    _Part.TRAIN: _FileCounts(
        lines=4,
        pairs={_Stage.UNDERSTANDING: 24, _Stage.PERCEPTION: 21, _Stage.REASONING: 2, _Stage.DECISION_MAKING: 1},
        unlabelled={_ClosedQuestion.HEALTH_STATE: 1},
    ),
    _Part.TEST: _FileCounts(
        lines=3,
        pairs={_Stage.UNDERSTANDING: 16, _Stage.PERCEPTION: 14, _Stage.REASONING: 2, _Stage.DECISION_MAKING: 1},
        unlabelled={_ClosedQuestion.FAULT_CAUSE: 1},
    ),
}


def _values(row_id: int) -> np.ndarray:
    """Every value names its row, step and channel, so a misplaced value can't pass."""
    steps = np.arange(_STEPS, dtype=np.float64)[:, None]
    channels = np.arange(_CHANNELS, dtype=np.float64)[None, :]
    return row_id * 1000 + steps + channels / 100


def _write_signals(path: Path, rows: int = _N_ROWS) -> None:
    """Write the signals file. Only the named rows hold values, so the chunked file stays small."""
    with h5py.File(path, "w") as file:
        file["data_ID"] = np.arange(1, rows + 1, dtype=np.int64)
        values = file.create_dataset("seq_data", shape=(rows, _STEPS, _CHANNELS), dtype=np.float64, chunks=True)
        for row_id in _NAMED_ROWS:
            if row_id <= rows:
                values[row_id - 1] = _values(row_id)


def _lines(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def _write_qa(path: Path, lines: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _fixture_counts(monkeypatch):
    monkeypatch.setattr(EngineMTQAConnector, "SOURCE_COUNTS", _COUNTS)


@pytest.fixture
def refs(tmp_path) -> list[HubFileRef]:
    """The release in ``tmp_path``: real QA lines and synthetic signals."""
    _write_signals(tmp_path / _SIGNALS)
    for name in (_TRAIN, _TEST):
        shutil.copyfile(_FIXTURES / name, tmp_path / name)
    return [HubFileRef(tmp_path / name, part) for part, name in EngineMTQAConnector.SOURCE_FILES.items()]


@pytest.fixture
def dataset(refs) -> TimeFDataset:
    return EngineMTQAConnector().convert(refs)


def _row_id(record: Record) -> int:
    value = record.metadata["row_id"]
    assert isinstance(value, int)
    return value


class _View(NamedTuple):
    """A task as plain values."""

    kind: str
    split: str
    rows: tuple[int, ...]
    prompt: str
    targets: tuple[str, ...]
    label_set: str | None
    metadata: dict[str, Any]


def _view(task: Task) -> _View:
    assert isinstance(task, AnswerTask | ClassificationTask)
    assert task.split is not None
    return _View(
        kind=type(task).__name__,
        split=task.split.value,
        rows=tuple(_row_id(record) for record in task.inputs),
        prompt=str(read_value(task.prompt)),
        targets=tuple(str(read_value(target)) for target in task.targets or ()),
        label_set=task.label_set.name if isinstance(task, ClassificationTask) and task.label_set else None,
        metadata=dict(task.metadata),
    )


def _views(dataset: TimeFDataset) -> list[_View]:
    return [_view(task) for task in dataset.iter_tasks()]


def _find(views: list[_View], split: str, line: int, offset: int) -> _View:
    return next(
        view
        for view in views
        if view.split == split and view.metadata["source_line"] == line and view.metadata["qa_offset"] == offset
    )


def test_converts_the_fixture(dataset):
    # Each named row once, though rows 20, 279, 287, 295, 319 and 346 are named in both files.
    assert [_row_id(record) for record in dataset.records] == _NAMED_ROWS
    views = _views(dataset)
    # Per single-cycle line: 8 answer and 7 classification tasks, less one for an empty answer.
    # Per multi-cycle line: 2 classification tasks and 1 answer task.
    single = ["AnswerTask"] * 8 + ["ClassificationTask"] * 7
    train = single[:-1] + single + single + ["ClassificationTask"] * 2 + ["AnswerTask"]
    test = single + ["ClassificationTask"] * 2 + ["AnswerTask"] + single[:-2] + single[-1:]
    assert [view.kind for view in views] == train + test
    assert [view.split for view in views] == ["train"] * len(train) + ["test"] * len(test)


def test_fault_cause_targets_its_letters_over_a_multi_label_set(dataset):
    tasks = [task for task in dataset.iter_tasks() if task.metadata.get("closed_question") == "fault_cause"]
    letters = {_view(task).rows: _view(task).targets for task in tasks}
    assert letters[8435,] == ("a", "b", "e")
    assert letters[278,] == ("a",)
    label_set = next(task.label_set for task in tasks if isinstance(task, ClassificationTask))
    assert label_set is not None
    assert label_set.multi_label
    assert [(label.name, label.description) for label in label_set.labels] == [
        ("a", "HPT efficiency modifier"),
        ("b", "LPT efficiency modifier"),
        ("c", "Fan efficiency modifier"),
        ("d", "HPT flow modifier"),
        ("e", "LPT flow modifier"),
        ("f", "Fan flow modifier"),
    ]


def test_closed_questions_keep_their_place_and_label_set(dataset):
    view = _find(_views(dataset), "train", line=1, offset=13)
    published = _lines(_TRAIN)[1]["conversations"][26]["value"]
    assert view == _View(
        kind="ClassificationTask",
        split="train",
        rows=(8435,),
        prompt=published,
        targets=("a", "b", "e"),
        label_set="fault_cause",
        metadata={
            "source_file": _TRAIN,
            "source_line": 1,
            "qa_offset": 13,
            "stage": 2,
            "source_issues": [],
            "closed_question": "fault_cause",
        },
    )
    health = _find(_views(dataset), "train", line=0, offset=8)
    assert (health.label_set, health.metadata["closed_question"]) == ("component_health", "fan_health")


def test_open_questions_keep_the_published_text(dataset):
    lines = _lines(_TEST)
    view = _find(_views(dataset), "test", line=1, offset=2)
    assert view == _View(
        kind="AnswerTask",
        split="test",
        rows=_MULTI_TEST,
        prompt=lines[1]["conversations"][4]["value"],
        targets=(lines[1]["conversations"][5]["value"],),
        label_set=None,
        metadata={
            "source_file": _TEST,
            "source_line": 1,
            "qa_offset": 2,
            "stage": 4,
            "source_issues": [],
            "steps_per_cycle": 60,
        },
    )


def test_reasoning_takes_the_ten_rows_in_published_order(dataset):
    view = _find(_views(dataset), "train", line=3, offset=0)
    assert view.rows == _MULTI_TRAIN
    assert (view.metadata["stage"], view.metadata["closed_question"], view.targets) == (3, "engine_condition", ("b",))


def test_empty_closed_answers_get_no_task_and_are_listed(dataset):
    views = _views(dataset)
    with pytest.raises(StopIteration):
        _find(views, "train", line=0, offset=14)
    with pytest.raises(StopIteration):
        _find(views, "test", line=2, offset=13)
    unlabelled = next(a for a in dataset.annotations if a.key == "unlabelled_pairs")
    assert unlabelled.value == 2
    assert unlabelled.metadata["pairs"] == [
        {"source_file": _TRAIN, "source_line": 0, "qa_offset": 14},
        {"source_file": _TEST, "source_line": 2, "qa_offset": 13},
    ]


def test_pairs_repeated_across_files_are_flagged_on_both_sides(dataset):
    # Test line 0 is a byte copy of train line 2. The multi-cycle lines name other rows.
    issues = {(view.split, view.metadata["source_line"]): view.metadata["source_issues"] for view in _views(dataset)}
    assert issues["train", 2] == ["repeated_across_splits"]
    assert issues["test", 0] == ["repeated_across_splits"]
    assert issues["train", 3] == issues["test", 1] == []


def test_a_pair_repeated_in_its_file_is_flagged(refs, tmp_path, monkeypatch):
    train = _lines(_TRAIN)
    _write_qa(tmp_path / _TRAIN, [*train, train[3]])
    counts = _COUNTS[_Part.TRAIN]
    pairs = {**counts.pairs, _Stage.REASONING: 4, _Stage.DECISION_MAKING: 2}
    monkeypatch.setitem(_COUNTS, _Part.TRAIN, counts._replace(lines=5, pairs=pairs))
    views = _views(EngineMTQAConnector().convert(refs))
    assert {tuple(v.metadata["source_issues"]) for v in views if v.rows == _MULTI_TRAIN} == {("repeated_question",)}


def test_every_signal_of_a_record_holds_its_row(dataset):
    record = dataset.records[0]
    assert [source.name for source in record.sources] == ["N-CMAPSS engine"]
    assert [signal.name for signal in record.signals] == [f"channel_{c:02d}" for c in range(_CHANNELS)]
    for channel, signal in enumerate(record.signals):
        assert isinstance(signal.time_axis, RegularAxis)
        assert signal.time_axis.period_us == 1_000_000
        np.testing.assert_array_equal(signal.to_numpy(), _values(_row_id(record))[:, channel])


def test_storage_round_trip_keeps_every_task_and_value(dataset, tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset)
    with TimeFReader(registry.open_version("pandalin98/enginemt-qa", "1.0.0")) as reader:
        restored = reader.read()
    assert _views(restored) == _views(dataset)
    for record in restored.records:
        for channel, signal in enumerate(record.signals):
            np.testing.assert_array_equal(signal.to_numpy(), _values(_row_id(record))[:, channel])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda lines: lines[:-1], "expected"),
        (lambda lines: [], "expected"),
    ],
    ids=["a-line-short", "empty"],
)
def test_a_qa_file_with_other_counts_fails(refs, tmp_path, change, message):
    _write_qa(tmp_path / _TRAIN, change(_lines(_TRAIN)))
    with pytest.raises(TimeFValidationError, match=message) as raised:
        EngineMTQAConnector().convert(refs)
    assert _TRAIN in str(raised.value)


def test_a_shortened_signals_file_fails_when_read(dataset, tmp_path):
    _write_signals(tmp_path / _SIGNALS, rows=4)
    with pytest.raises(TimeFFormatError, match="cannot read row id"):
        dataset.records[-1].signals[0].to_numpy()


def test_a_missing_signals_file_fails_when_read(dataset, tmp_path):
    (tmp_path / _SIGNALS).unlink()
    with pytest.raises(TimeFFormatError, match="cannot open the signals file"):
        dataset.records[0].signals[0].to_numpy()


def test_a_row_with_a_nan_fails_when_read(dataset, tmp_path):
    with h5py.File(tmp_path / _SIGNALS, "a") as file:
        file["seq_data"][19, 5, 3] = np.nan
    with pytest.raises(TimeFValidationError, match="row id 20 holds a NaN"):
        dataset.records[0].signals[0].to_numpy()


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("changed", lambda lines: [{**lines[0], "id": "8435", "name": "8435.csv"}, *lines[1:]]),
        ("removed", lambda lines: lines[:-1]),
    ],
)
def test_a_qa_file_changed_after_conversion_fails_when_the_tasks_are_read(dataset, tmp_path, name, change):
    _write_qa(tmp_path / _TRAIN, change(_lines(_TRAIN)))
    message = "line 1: the line changed since conversion" if name == "changed" else "lines were removed"
    with pytest.raises(TimeFFormatError, match=message):
        list(dataset.iter_tasks())


def test_a_missing_qa_file_fails_when_the_tasks_are_read(dataset, tmp_path):
    (tmp_path / _TRAIN).unlink()
    with pytest.raises(TimeFFormatError, match="cannot read the file"):
        list(dataset.iter_tasks())


def test_a_file_that_is_not_hdf5_fails_with_its_path(refs, tmp_path):
    (tmp_path / _SIGNALS).write_bytes(b"not hdf5")
    with pytest.raises(TimeFFormatError, match="cannot open the signals file"):
        EngineMTQAConnector().convert(refs)


def _rewrite(path: Path, **datasets: Any) -> None:
    with h5py.File(path, "w") as file:
        for name, value in datasets.items():
            if value is None:
                file.create_group(name)
            else:
                file[name] = value


_IDS = np.arange(1, 4)
_ZEROS = np.zeros((3, _STEPS, _CHANNELS))


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ({"data_ID": _IDS, "seq_data": _ZEROS, "extra": 1}, "expected seq_data and data_ID"),
        ({"data_ID": _IDS, "seq_data": np.zeros((3, _STEPS, _CHANNELS - 1))}, r"seq_data is float64 \(3, 600, 32\)"),
        ({"data_ID": _IDS, "seq_data": _ZEROS.astype(np.float32)}, "seq_data is float32"),
        ({"data_ID": _IDS, "seq_data": None}, "seq_data is a Group, not an HDF5 dataset"),
        ({"data_ID": _IDS.astype(np.float64), "seq_data": _ZEROS}, "data_ID is float64"),
        ({"data_ID": np.array([2, 2, 3]), "seq_data": _ZEROS}, "data_ID is not the row ids 1 to 3"),
    ],
    ids=["extra-key", "32-channels", "float32", "group", "float-ids", "ids-not-1-to-n"],
)
def test_a_signals_file_with_another_layout_fails(refs, tmp_path, datasets, message):
    _rewrite(tmp_path / _SIGNALS, **datasets)
    with pytest.raises(TimeFValidationError, match=message):
        EngineMTQAConnector().convert(refs)


def _with_pair(offset: int, **changes: Any) -> dict[str, Any]:
    """Train line 0 with one pair changed: ``answer`` sets the answer, other keys the question."""
    line = _lines(_TRAIN)[0]
    conversations = [dict(turn) for turn in line["conversations"]]
    question, answer = conversations[2 * offset], conversations[2 * offset + 1]
    answer["value"] = changes.pop("answer", answer["value"])
    question.update(changes)
    return {**line, "conversations": conversations}


def _with(**changes: Any) -> dict[str, Any]:
    """A train line with its own keys changed: the multi-cycle line 3 if ``multi``, else line 0."""
    line = _lines(_TRAIN)[3 if changes.pop("multi", False) else 0]
    return {**line, **changes}


def _health_state(text: str) -> str:
    return _lines(_TRAIN)[0]["conversations"][28]["value"].replace("Imminent", text)


# In a single-cycle line, pairs 8 to 14 are the Perception questions: 13 is the fault cause and 14 the
# health state.
@pytest.mark.parametrize(
    ("line", "message"),
    [
        (_with_pair(13, answer="a, g"), "labels not in label set 'fault_cause'"),
        (_with_pair(13, answer="a,b"), "labels not in label set 'fault_cause'"),
        (_with_pair(13, answer="a, a"), "repeated labels"),
        (_with_pair(14, answer="a, b"), "takes exactly one"),
        (_with_pair(8, value="What is the health status of the fan?\n <ts>"), "unknown stage 2"),
        (_with_pair(14, value=_health_state("Near")), "prints options other than"),
        (_with_pair(0, answer=""), "open question has an empty answer"),
        (_with_pair(0, attribute="close"), "stage 1 questions are not 'close'"),
        (_with_pair(0, stage="5"), "bad stage '5'"),
        (_with_pair(0, stage=[2]), r"bad stage \[2\]"),
        (_with_pair(0, attribute={}), "bad attribute"),
        (_with_pair(0, value="Describe the signal."), "one <ts> marker"),
        (_with_pair(0, **{"from": "gpt"}), "alternate a human question"),
        (_with(name="4.csv"), "doesn't repeat id"),
        (_with(multi=True, id=["1"] * 10, name=["1.csv"] * 10), "repeats a row"),
        (_with(multi=True, id=["1"] * 9, name=["1.csv"] * 9), "names 9 rows"),
        (_with(id="0278", name="0278.csv"), "not decimal row ids"),
        (_with(id="37670", name="37670.csv"), "past the file's 37669 rows"),
        (_with(extra=1), "expected an object with the keys"),
        (_with(conversations=[]), "alternate"),
    ],
)
def test_a_malformed_line_fails_with_its_place(refs, tmp_path, line, message):
    _write_qa(tmp_path / _TRAIN, [_lines(_TRAIN)[0], line])
    with pytest.raises(TimeFValidationError, match=message) as raised:
        EngineMTQAConnector().convert(refs)
    assert "train_qa.jsonl line 2" in str(raised.value)


def test_a_line_that_is_not_json_fails_with_its_place(refs, tmp_path):
    with (tmp_path / _TRAIN).open("a", encoding="utf-8") as file:
        file.write("not json\n")
    with pytest.raises(TimeFValidationError, match=r"train_qa.jsonl line 5"):
        EngineMTQAConnector().convert(refs)


def test_a_missing_part_fails(refs):
    with pytest.raises(TimeFValidationError, match="needs the parts"):
        EngineMTQAConnector().convert(refs[1:])
