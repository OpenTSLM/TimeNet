import json
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.pandalin98.enginemt_qa import EngineMTQAConnector
from timenet_connectors.datasets.pandalin98.enginemt_qa.loader import EngineMTQADatasetLoader
from timenet_connectors.datasets.pandalin98.enginemt_qa.model_input import authors_series
from timenet_connectors.datasets.pandalin98.enginemt_qa.questions import ClosedQuestion, Vocabulary
from timenet_connectors.datasets.pandalin98.enginemt_qa.records import record_metadata
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import (
    N_CHANNELS,
    N_STEPS,
    Part,
    PinnedInventory,
    PinnedRelease,
)
from timenet_connectors.datasets.pandalin98.enginemt_qa.signals import check_layout
from timenet_connectors.datasets.pandalin98.enginemt_qa.tasks import (
    DecisionTask,
    PerceptionTask,
    ReasoningTask,
    UnderstandingTask,
)


_ROWS = 13
# Unsorted, as 256 published lists are. Row 13 is named by no QA.
_MULTI = ["5", "4", "6", "7", "8", "9", "10", "11", "12", "1"]
_PERCEPTION = [question for question in ClosedQuestion if question.stage == 2]
# Each file's two lines hold one pair of each stage layout, with one empty closed answer each.
_INVENTORY = PinnedInventory(lines=2, qa_per_stage={1: 8, 2: 7, 3: 2, 4: 1}, empty_answers={})
_RELEASE = PinnedRelease(
    rows=_ROWS,
    referenced_rows=12,
    qa={
        Part.TRAIN: _INVENTORY._replace(empty_answers={"health_state": 1}),
        Part.TEST: _INVENTORY._replace(empty_answers={"fault_cause": 1}),
    },
)


class _FixtureConnector(EngineMTQAConnector):
    """The connector, pinned to the fixture's counts instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_RELEASE: ClassVar[PinnedRelease] = _RELEASE


def _values(row_id: int) -> np.ndarray:
    """Every value names its row, step and channel, so a misplaced value can't pass."""
    steps = np.arange(N_STEPS, dtype=np.float64)[:, None]
    channels = np.arange(N_CHANNELS, dtype=np.float64)[None, :]
    return row_id * 1000 + steps + channels / 100


def _write_signals(path: Path, rows: int = _ROWS) -> None:
    with h5py.File(path, "w") as file:
        file["data_ID"] = np.arange(1, rows + 1, dtype=np.int64)
        file["seq_data"] = np.stack([_values(row_id) for row_id in range(1, rows + 1)])


def _turns(question: str, answer: str, stage: int, attribute: str) -> list[dict[str, str]]:
    return [
        {"from": "human", "stage": str(stage), "attribute": attribute, "value": question},
        {"from": "gpt", "value": answer},
    ]


def _single(row_id: str, answers: dict[ClosedQuestion, str]) -> dict[str, Any]:
    turns = []
    for k in range(8):
        turns += _turns(f"What does the change in signal {k} represent in one cycle?\n <ts>", f"steady {k}", 1, "open")
    for question in _PERCEPTION:
        turns += _turns(question.text, answers.get(question, "a"), 2, "close")
    return {"id": row_id, "name": f"{row_id}.csv", "conversations": turns}


def _multi(condition: str) -> dict[str, Any]:
    turns = _turns(ClosedQuestion.ENGINE_CONDITION.text, condition, 3, "close")
    turns += _turns(ClosedQuestion.FAILURE_PROBABILITY.text, "d", 3, "close")
    turns += _turns(
        "Hey, what's the plan for this engine signal across 10 cycles?\n <ts>", "Replace the HPT.", 4, "open"
    )
    return {"id": _MULTI, "name": [f"{i}.csv" for i in _MULTI], "conversations": turns}


def _lines(part: Part) -> list[dict[str, Any]]:
    if part is Part.TRAIN:
        single = _single("3", {ClosedQuestion.FAULT_CAUSE: "a, b, e", ClosedQuestion.HEALTH_STATE: ""})
        return [single, _multi("c")]
    # Row 3 is in both files, as published rows are.
    return [_single("2", {ClosedQuestion.FAULT_CAUSE: "", ClosedQuestion.HPT_HEALTH: "b"}), _multi("e")]


def _write_qa(path: Path, lines: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _release(tmp_path: Path) -> list[HubFileRef]:
    paths = {part: tmp_path / part.source_file for part in Part}
    _write_signals(paths[Part.SIGNALS])
    for part in (Part.TRAIN, Part.TEST):
        _write_qa(paths[part], _lines(part))
    return [HubFileRef(path, part.value) for part, path in paths.items()]


def _output(dataset) -> dict:
    """What the dataset holds, as plain values."""
    records = {
        record_metadata(record).row_id: sorted((s.name, s.n_values) for s in record.signals)
        for record in dataset.records
    }
    tasks = [
        (
            type(task).__name__,
            task.split.value,
            tuple(record_metadata(record).row_id for record in task.inputs),
            read_value(task.prompt),
            tuple(read_value(target) for target in task.targets),
            getattr(task, "target_schema", None),
            dict(task.metadata),
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {a.key: (read_value(a.value), dict(a.metadata)) for a in dataset.annotations}
    return {"records": records, "tasks": tasks, "annotations": annotations}


def test_converts_the_fixture(tmp_path):
    output = _output(_FixtureConnector().convert(_release(tmp_path)))
    channels = sorted((f"channel_{index:02d}", N_STEPS) for index in range(N_CHANNELS))
    # Only the rows a QA names, and each once though row 3 is in both files.
    assert output["records"] == dict.fromkeys(range(1, 13), channels)
    # Each file: 8 Understanding, 6 Perception (one empty answer left out), 2 Reasoning, 1 Decision.
    kinds = [(kind, split) for kind, split, *_ in output["tasks"]]
    per_file = ["AnswerTask"] * 8 + ["ClassificationTask"] * 8 + ["AnswerTask"]
    assert kinds == [(kind, "train") for kind in per_file] + [(kind, "test") for kind in per_file]
    assert output["annotations"]["source_repository"] == ("pandalin98/EngineMT-QA", {})
    assert output["annotations"]["source_revision"] == ("ea2051a15512212976c22c1a075ec549c7d48eef", {})


def test_closed_questions_target_their_letters(tmp_path):
    tasks = _output(_FixtureConnector().convert(_release(tmp_path)))["tasks"]
    fault_cause = next(task for task in tasks if task[-1].get("closed_question") == "fault_cause")
    assert fault_cause == (
        "ClassificationTask",
        "train",
        (3,),
        ClosedQuestion.FAULT_CAUSE.text,
        ("a", "b", "e"),
        "fault_cause_vocabulary",
        {"source_file": "train_qa.jsonl", "source_line": 0, "qa_offset": 13, "closed_question": "fault_cause"},
    )
    hpt = next(task for task in tasks if task[1] == "test" and task[-1].get("closed_question") == "hpt_health")
    assert hpt[3:6] == (ClosedQuestion.HPT_HEALTH.text, ("b",), "component_health_vocabulary")
    # The train health state and the test fault cause are empty, so neither has a task.
    asked = [(split, metadata.get("closed_question")) for _, split, _, _, _, _, metadata in tasks]
    assert ("train", "health_state") not in asked
    assert ("test", "fault_cause") not in asked


def test_open_questions_keep_the_published_text(tmp_path):
    tasks = _output(_FixtureConnector().convert(_release(tmp_path)))["tasks"]
    assert tasks[1] == (
        "AnswerTask",
        "train",
        (3,),
        "What does the change in signal 1 represent in one cycle?\n <ts>",
        ("steady 1",),
        None,
        {"source_file": "train_qa.jsonl", "source_line": 0, "qa_offset": 1},
    )
    decision = tasks[-1]
    assert decision[0] == "AnswerTask"
    assert decision[3:5] == (
        "Hey, what's the plan for this engine signal across 10 cycles?\n <ts>",
        ("Replace the HPT.",),
    )
    assert decision[-1] == {"source_file": "test_qa.jsonl", "source_line": 1, "qa_offset": 2, "steps_per_cycle": 60}


def test_multi_cycle_tasks_take_ten_records_in_published_order(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))
    task = next(task for task in dataset.iter_tasks() if len(task.inputs) > 1)
    assert [record_metadata(record).row_id for record in task.inputs] == [int(i) for i in _MULTI]
    # The authors' TsQaDataset: the first 60 steps of each row, joined in published order.
    expected = np.concatenate([_values(int(i))[:60] for i in _MULTI])
    np.testing.assert_array_equal(authors_series(task), expected)


def test_signals_read_their_row(tmp_path):
    dataset = _FixtureConnector().convert(_release(tmp_path))
    record = next(record for record in dataset.records if record_metadata(record).row_id == 7)
    for signal in record.signals:
        channel = int(signal.name.removeprefix("channel_"))
        assert isinstance(signal.time_axis, RegularAxis)
        assert signal.time_axis.period_us == 1_000_000
        np.testing.assert_array_equal(signal.to_numpy(), _values(7)[:, channel])
    single = next(task for task in dataset.iter_tasks() if len(task.inputs) == 1)
    np.testing.assert_array_equal(authors_series(single), _values(3))


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert(_release(tmp_path))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("pandalin98/enginemt-qa", "1.0.0")) as reader:
        restored = reader.read()
    assert _output(restored) == _output(original)
    record = next(record for record in restored.records if record_metadata(record).row_id == 12)
    np.testing.assert_array_equal(
        next(s for s in record.signals if s.name == "channel_32").to_numpy(), _values(12)[:, 32]
    )


def test_vocabularies_name_each_letter(tmp_path):
    annotations = _output(_FixtureConnector().convert(_release(tmp_path)))["annotations"]
    assert annotations["component_health_vocabulary"] == (["a", "b"], {"options": {"a": "Health", "b": "Fault"}})
    assert annotations["failure_probability_vocabulary"][0] == ["a", "b", "c", "d", "e"]
    assert {key for key in annotations if key.endswith("_vocabulary")} == {f"{v}_vocabulary" for v in Vocabulary}


def test_all_rows_builds_a_record_per_row(tmp_path):
    loader = EngineMTQADatasetLoader.parse_from(_release(tmp_path), all_rows=True, release=_RELEASE)
    assert [record_metadata(record).row_id for record in loader.records()] == list(range(1, _ROWS + 1))


def test_tasks_of_selects_a_kind_and_split(tmp_path):
    loader = EngineMTQADatasetLoader.parse_from(_release(tmp_path), release=_RELEASE)
    assert len(list(loader.tasks_of(UnderstandingTask, split=Split.TEST))) == 8
    assert len(list(loader.tasks_of(PerceptionTask, split=Split.TRAIN))) == 6
    assert len(list(loader.tasks_of(ReasoningTask))) == 4
    assert [task.split for task in loader.tasks_of(DecisionTask)] == [Split.TRAIN, Split.TEST]
    assert {kind.TIMENET_TYPE for kind in loader.TASK_KINDS} == {AnswerTask, ClassificationTask}


def test_a_changed_inventory_fails(tmp_path):
    release = _RELEASE._replace(qa={**_RELEASE.qa, Part.TEST: _INVENTORY})
    with pytest.raises(TimeFValidationError, match="the pinned file holds"):
        EngineMTQADatasetLoader.parse_from(_release(tmp_path), release=release)
    with pytest.raises(TimeFValidationError, match="name 12 rows"):
        EngineMTQADatasetLoader.parse_from(_release(tmp_path), release=_RELEASE._replace(referenced_rows=13))


def _with_pair(line: dict[str, Any], offset: int, **changes: str) -> dict[str, Any]:
    conversations = [dict(turn) for turn in line["conversations"]]
    question, answer = conversations[2 * offset], conversations[2 * offset + 1]
    answer["value"] = changes.pop("answer", answer["value"])
    question.update(changes)
    return {**line, "conversations": conversations}


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (_with_pair(_single("3", {}), 13, answer="a, g"), "not distinct letters"),
        (_with_pair(_single("3", {}), 13, answer="a,b"), "not distinct letters"),
        (_with_pair(_single("3", {}), 13, answer="a, a"), "not distinct letters"),
        (_with_pair(_single("3", {}), 14, answer="a, b"), "several letters"),
        (_with_pair(_single("3", {}), 8, value="What is the health status of the fan?\n <ts>"), "unknown stage 2"),
        (_with_pair(_single("3", {}), 0, answer=""), "open question has an empty answer"),
        (_with_pair(_single("3", {}), 0, attribute="close"), "stage 1 questions are not 'close'"),
        (_with_pair(_single("3", {}), 0, value="Describe the signal."), "one <ts> marker"),
        (_with_pair(_multi("c"), 2, stage="1"), "stage 1 question names 10 rows"),
        ({**_single("3", {}), "name": "4.csv"}, "doesn't repeat id"),
        ({**_multi("c"), "id": _MULTI[:9], "name": [f"{i}.csv" for i in _MULTI[:9]]}, "names 9 rows"),
        ({**_single("3", {}), "id": "03", "name": "03.csv"}, "String should match pattern"),
        ({**_single("3", {}), "extra": 1}, "Extra inputs are not permitted"),
        ({**_single("3", {}), "conversations": _single("3", {})["conversations"][:-1]}, "alternate"),
    ],
)
def test_a_malformed_line_fails_with_its_place(tmp_path, line, message):
    refs = _release(tmp_path)
    _write_qa(tmp_path / Part.TRAIN.source_file, [_lines(Part.TRAIN)[0], line])
    with pytest.raises(TimeFValidationError, match=message) as raised:
        EngineMTQADatasetLoader.parse_from(refs, release=_RELEASE)
    assert "train_qa.jsonl line 2" in str(raised.value)


def test_check_layout_rejects_another_file(tmp_path):
    path = tmp_path / "time_series_data.h5"
    _write_signals(path)
    check_layout(path, n_rows=_ROWS)
    with pytest.raises(TimeFValidationError, match="seq_data is"):
        check_layout(path, n_rows=_ROWS + 1)
    with h5py.File(path, "a") as file:
        file["data_ID"][0] = 2
    with pytest.raises(TimeFValidationError, match="not the row ids"):
        check_layout(path, n_rows=_ROWS)


def test_a_file_changed_after_conversion_fails_when_read(tmp_path):
    refs = _release(tmp_path)
    dataset = _FixtureConnector().convert(refs)
    _write_signals(tmp_path / Part.SIGNALS.source_file, rows=4)
    record = next(record for record in dataset.records if record_metadata(record).row_id == 12)
    with pytest.raises(TimeFFormatError, match="cannot read row id 12"):
        record.signals[0].to_numpy()
