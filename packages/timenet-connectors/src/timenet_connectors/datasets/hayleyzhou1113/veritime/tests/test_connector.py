from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any, ClassVar

import pytest

from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, ScalarPredictionTask, TextHandle, ureg
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.hayleyzhou1113.veritime import VeriTimeConnector, request
from timenet_connectors.datasets.hayleyzhou1113.veritime.loader import VeriTimeDatasetLoader
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import (
    PINNED_INVENTORY,
    Family,
    Part,
    PinnedInventory,
    SourceTask,
)
from timenet_connectors.datasets.hayleyzhou1113.veritime.series import SeriesKind
from timenet_connectors.sources import json_stream


_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED_REVISION = "fc88cb1305a96a208093e13b69d766b19c84c99b"
# Marks a field the test removes from a row.
_MISSING = object()
# The fixtures' inventory: two scenario series shared across the splits, and one knowledge series per file.
_FIXTURE_INVENTORY = PinnedInventory(
    rows={
        Part.SCENARIO_TRAIN: {
            SourceTask.ANOMALY_DETECTION: 1,
            SourceTask.SCENARIO_ATTRIBUTION: 1,
            SourceTask.INFERENTIAL_CALCULATION: 1,
        },
        Part.SCENARIO_TEST: {SourceTask.ANOMALY_DETECTION: 1, SourceTask.INFERENTIAL_CALCULATION: 1},
        Part.KNOWLEDGE_TRAIN: {SourceTask.ECG: 1},
        Part.KNOWLEDGE_TEST: {SourceTask.EMG: 1},
    },
    series={Family.SCENARIO: 2, Family.KNOWLEDGE: 2},
)


class _FixtureConnector(VeriTimeConnector):
    """The connector, pinned to the fixtures' inventory instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_INVENTORY: ClassVar[PinnedInventory] = _FIXTURE_INVENTORY


def _refs(directory: Path = _FIXTURES) -> list[HubFileRef]:
    return [HubFileRef(directory / f"{part}.jsonl", part) for part in Part]


def _fixture_rows(part: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_FIXTURES / f"{part}.jsonl").read_text(encoding="utf-8").splitlines()]


def _published_values(row: dict[str, Any]) -> list[float]:
    """A row's series, parsed from the fixture independently of the connector."""
    series = row["timeseries2"]
    return series[0] if isinstance(series, list) else [float(value) for value in series.split(", ")]


def _row_key(task) -> tuple[str, int]:
    return task.metadata["part"], task.metadata["row"]


def _typed(target: Any) -> Any:
    """A text target as is, and a number with its type, since 2 == 2.0: a count must come back as an int."""
    return target if isinstance(target, str | TextHandle) else (type(target), target)


_SERIES_A = [19.3028, 19.3579, -0.2928, 19.3098]
_SERIES_B = [0.2389, 0.239, -1.0168]
_STEP_DESCRIPTIONS = {
    "step1_label": "The process label of step 1, the task intent, as published.",
    "step2_label": "The process label of step 2, the key patterns separated by semicolons, as published.",
    "step4_label": "The process label of step 4, the preliminary answer, as published.",
    "step6_label": "The process label of step 6, the final answer, as published.",
}
_SOURCE_ANSWER_DESCRIPTION = "The ChatTS answer the authors extracted the label from, as published."

# The whole converted fixture, written out by hand from the fixture files, not from connector constants.
# The fixtures hold no CTU or RCW row, so the dataset holds no vocabulary for them.
_EXPECTED_DATASET_ANNOTATIONS: dict[str, Any] = {
    "source_repository": "HayleyZhou1113/VeriTime",
    "source_revision": _EXPECTED_REVISION,
    "anomaly_detection_vocabulary": ["Yes", "No"],
    "ecg_vocabulary": ["normal sinus rhythm", "atrial fibrillation", "other cardiac rhythms", "noise"],
    "emg_vocabulary": ["Healthy", "Myopathy", "Neuropathy"],
}
# Each Record by the rows that publish its series: kind, signal name, values source, axis, spec and unit.
_EXPECTED_RECORDS: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {
    (("scenario_train", 0), ("scenario_train", 1), ("scenario_test", 0)): {
        "kind": "scenario",
        "source": "ChatTS series",
        "signal": ("series", _SERIES_A, (OrdinalAxis, None), "chatts_series", None),
    },
    (("scenario_train", 2), ("scenario_test", 1)): {
        "kind": "scenario",
        "source": "ChatTS series",
        "signal": ("series", _SERIES_B, (OrdinalAxis, None), "chatts_series", None),
    },
    (("knowledge_train", 0),): {
        "kind": "ecg",
        "source": "Single-lead ECG",
        "signal": (
            "ecg",
            _published_values(_fixture_rows("knowledge_train")[0]),
            (RegularAxis, Fraction(20000, 3)),
            "ecg",
            None,
        ),
    },
    (("knowledge_test", 0),): {
        "kind": "emg",
        "source": "Needle EMG",
        "signal": (
            "emg",
            _published_values(_fixture_rows("knowledge_test")[0]),
            (OrdinalAxis, None),
            "emg",
            ureg.millivolt,
        ),
    },
}
# Each task's type, target and its type, and target schema; the other fields follow the row.
_EXPECTED_TASKS: dict[tuple[str, int], tuple[type, Any, str | None]] = {
    ("scenario_train", 0): (ClassificationTask, "Yes", "anomaly_detection_vocabulary"),
    ("scenario_train", 1): (AnswerTask, "Consistent market conditions", None),
    ("scenario_train", 2): (ScalarPredictionTask, 2, None),
    ("scenario_test", 0): (ClassificationTask, "No", "anomaly_detection_vocabulary"),
    # The one decimal label of the release, a two-part question's amplitude, stays a float.
    ("scenario_test", 1): (ScalarPredictionTask, 10641.02, None),
    ("knowledge_train", 0): (ClassificationTask, "other cardiac rhythms", "ecg_vocabulary"),
    ("knowledge_test", 0): (ClassificationTask, "Neuropathy", "emg_vocabulary"),
}


def _expected_task(part: str, index: int) -> dict[str, Any]:
    row = _fixture_rows(part)[index]
    task_type, target, schema = _EXPECTED_TASKS[part, index]
    annotations = {
        key: (row[key], description) for key, description in _STEP_DESCRIPTIONS.items() if row[key] is not None
    }
    if "output" in row:
        annotations["source_answer"] = (row["output"], _SOURCE_ANSWER_DESCRIPTION)
    return {
        "type": task_type,
        "inputs": 1,
        "prompt": row["question"],
        "input_annotations": {},
        "targets": (_typed(target),),
        "target_schema": schema,
        "rationale": row["cot"],
        "metadata": {"source_task": row["task"], "part": part, "row": index},
        "annotations": annotations,
        "split": part.split("_")[1],
    }


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with Records keyed by their rows and tasks by row."""
    records, tasks = {}, {}
    for record in dataset.records:
        (source,) = record.sources
        (signal,) = source.signals
        metadata = record.metadata
        rows = tuple((row["part"], row["row"]) for row in metadata["rows"])
        axis = signal.time_axis
        records[rows] = {
            "kind": metadata["kind"],
            "source": source.name,
            "signal": (
                signal.name,
                signal.to_numpy().tolist(),
                (type(axis), getattr(axis, "period_us", None)),
                signal.spec.spec_type,
                signal.spec.unit_value,
            ),
        }
        # VeriTime's fields, and the shared subject_id stored as null: the release names no subject.
        assert set(record.metadata) == {"subject_id", "kind", "rows"}
        assert record.metadata["subject_id"] is None
        assert str(signal.to_numpy().dtype) == "float64"
    for task in dataset.iter_tasks():
        tasks[_row_key(task)] = {
            "type": type(task),
            "inputs": len(task.inputs),
            "prompt": task.prompt,
            "input_annotations": {a.key: a.value for a in task.input_annotations},
            "targets": tuple(_typed(target) for target in task.targets),
            "target_schema": getattr(task, "target_schema", None),
            "rationale": task.rationale,
            "metadata": dict(task.metadata),
            "annotations": {a.key: (a.value, a.description) for a in task.annotations},
            "split": task.split.value,
        }
    annotations = {a.key: a.value for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


def test_storage_round_trip_keeps_the_whole_conversion(monkeypatch, tmp_path):
    # The lazy loaders seek by byte offset, including across CRLF line ends, UTF-8 text and small reads.
    for ref in _refs():
        (tmp_path / ref.path.name).write_bytes(ref.path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 17)
    original = _FixtureConnector().convert(_refs(tmp_path))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("hayleyzhou1113/veritime", "1.0.0")) as reader:
        restored = reader.read()
    expected = {
        "annotations": _EXPECTED_DATASET_ANNOTATIONS,
        "records": _EXPECTED_RECORDS,
        "tasks": {(part, index): _expected_task(part, index) for part, index in _EXPECTED_TASKS},
    }
    # The writer drew every lazy signal and the tasks already, so this pass also checks they load again.
    assert _output(original) == expected
    assert _output(restored) == expected


def _authors_question(row: dict[str, Any]) -> str:
    """The user message the authors build for a row.

    Ported line for line from ``process_timeseries`` in VeriTime_sft.py, VeriTime_grpo.py and
    VeriTime_evaluate.py, which are the same for a row with ``timeseries2``.
    """
    question = row["question"]
    timeseries2 = row["timeseries2"]
    task = row["task"]
    if task in ["Anomaly detection", "Scenario attribution", "Inferential calculation"]:  # noqa: PLR6201
        ts_data = timeseries2[0] if isinstance(timeseries2[0], list) else timeseries2
        ts_str = ", ".join(map(str, ts_data))
    else:
        ts_str = timeseries2
    return question.replace("<ts><ts/>", ts_str)


def test_request_rebuilds_the_authors_text():
    dataset = _FixtureConnector().convert(_refs())
    tasks = {_row_key(task): task for task in dataset.iter_tasks()}
    for part in Part:
        for index, row in enumerate(_fixture_rows(part)):
            assert request.authors_request(tasks[part.value, index]) == _authors_question(row)
    # The authors' system prompt, as VeriTime_sft.py formats it from its tags.
    assert request.SYSTEM_PROMPT == (
        "You are given a time series reasoning problem.\nThink about the problem and provide your working out.\n"
        "Place it between <THINK> and </THINK>.\nThen, provide your solution between <ANSWER></ANSWER>"
    )


def _authors_final_answer_reward(response: str, answer: str) -> float:
    """Score a response's final answer as the authors' RL reward does.

    Ported line for line from ``check_final_answer`` in VeriTime_grpo.py, for one completion.
    """
    answer_pattern = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
    answer_content_pattern = re.compile(r"The answer is\s*(.+)\.?", re.IGNORECASE)
    score = 0.0
    answer_match = answer_pattern.search(response)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        content_match = answer_content_pattern.search(answer_content)
        if content_match:
            extracted_answer = content_match.group(1).strip()
            score += 5.0 if answer.lower() in extracted_answer.lower() else -2.0
        else:
            score += 5.0 if answer.lower() in answer_content.lower() else -2.0
    else:
        score -= 2.0
    return score


def test_rationale_earns_the_authors_final_answer_reward():
    # The rationale is the published cot, which the authors fine-tune on, so it earns the full reward
    # against the published label, which the authors pass as the answer.
    dataset = _FixtureConnector().convert(_refs())
    for task in dataset.iter_tasks():
        part, index = _row_key(task)
        assert task.rationale is not None
        rationale = task.rationale.read()
        assert rationale == _fixture_rows(part)[index]["cot"]
        assert _authors_final_answer_reward(rationale, _fixture_rows(part)[index]["label"]) == pytest.approx(5.0)


def test_each_task_method_selects_its_rows():
    loader = VeriTimeDatasetLoader.parse_from(_refs(), inventory=_FIXTURE_INVENTORY)
    methods = {
        "Anomaly detection": (loader.anomaly_detection_tasks, ClassificationTask),
        "Scenario attribution": (loader.scenario_attribution_tasks, AnswerTask),
        "Inferential calculation": (loader.inferential_calculation_tasks, ScalarPredictionTask),
        "CTU": (loader.ctu_tasks, ClassificationTask),
        "ECG": (loader.ecg_tasks, ClassificationTask),
        "EMG": (loader.emg_tasks, ClassificationTask),
        "RCW": (loader.rcw_tasks, ClassificationTask),
    }
    selected = []
    for source_task, (method, task_type) in methods.items():
        tasks = list(method())
        assert all(type(task) is task_type and task.metadata["source_task"] == source_task for task in tasks)
        selected.extend(_row_key(task) for task in tasks)
    # tasks() keeps the file order and holds exactly the tasks of the seven methods.
    in_file_order = [_row_key(task) for task in loader.tasks()]
    assert in_file_order == [(part.value, index) for part in Part for index in range(len(_fixture_rows(part)))]
    assert sorted(selected) == sorted(in_file_order)


def test_declares_pinned_source_files():
    # The base's download() turns these declarations into refs; pin the declarations themselves.
    assert VeriTimeConnector.HF_REPO == "HayleyZhou1113/VeriTime"
    assert VeriTimeConnector.SOURCE_REVISION == _EXPECTED_REVISION
    assert VeriTimeConnector.SOURCE_FILES == {
        "scenario_train": "Dataset/Scenario/scenario_shuffled_train.jsonl",
        "scenario_test": "Dataset/Scenario/scenario_shuffled_test.jsonl",
        "knowledge_train": "Dataset/Knowledge/knowldege_shuffled_train.jsonl",
        "knowledge_test": "Dataset/Knowledge/knowledge_test.jsonl",
    }
    # Literal pins, so moving to another revision is a deliberate test edit.
    assert VeriTimeConnector.SOURCE_INVENTORY is PINNED_INVENTORY
    assert {
        part.value: {task.value: n for task, n in rows.items()} for part, rows in PINNED_INVENTORY.rows.items()
    } == {
        "scenario_train": {"Anomaly detection": 1016, "Scenario attribution": 855, "Inferential calculation": 188},
        "scenario_test": {"Anomaly detection": 180, "Scenario attribution": 176, "Inferential calculation": 105},
        "knowledge_train": {"CTU": 131, "ECG": 584, "EMG": 300, "RCW": 127},
        "knowledge_test": {"CTU": 144, "ECG": 200, "EMG": 150, "RCW": 200},
    }
    assert dict(PINNED_INVENTORY.series) == {Family.SCENARIO: 1259, Family.KNOWLEDGE: 1836}


def test_series_kinds_state_the_published_rate_and_length():
    # ECG is every second sample of 300 Hz, RCW every eighth of 2 kHz, CTU one value per 2 minutes.
    expected = {
        SeriesKind.SCENARIO: (None, None, None),
        SeriesKind.CTU: (Fraction(120_000_000), 720, None),
        SeriesKind.ECG: (Fraction(1_000_000, 150), 500, None),
        SeriesKind.EMG: (None, 600, ureg.millivolt),
        SeriesKind.RCW: (Fraction(4000), 500, None),
    }
    for kind, (period, n_values, unit) in expected.items():
        axis = kind.time_axis()
        assert (getattr(axis, "period_us", None), kind.n_values, kind.spec.unit_value) == (period, n_values, unit)
        assert isinstance(axis, OrdinalAxis if period is None else RegularAxis)


def test_pinned_inventory_mismatch_fails():
    # The release's inventory, against the fixtures.
    with pytest.raises(TimeFValidationError, match="differ from the pinned"):
        VeriTimeConnector().convert(_refs())


def test_a_family_series_count_is_checked_only_with_both_files():
    scenario_train = [ref for ref in _refs() if ref.part == "scenario_train"]
    inventory = _FIXTURE_INVENTORY._replace(series={Family.SCENARIO: 99, Family.KNOWLEDGE: 99})
    VeriTimeDatasetLoader.parse_from(scenario_train, inventory=inventory)
    with pytest.raises(TimeFValidationError, match="2 distinct series differ from the pinned 99"):
        VeriTimeDatasetLoader.parse_from(_refs(), inventory=inventory)


def test_a_reread_that_no_longer_matches_its_row_fails(tmp_path):
    for ref in _refs():
        (tmp_path / ref.path.name).write_bytes(ref.path.read_bytes())
    dataset = _FixtureConnector().convert(_refs(tmp_path))
    # Same byte length, so the range still holds one JSON value, but now with other values.
    path = tmp_path / "scenario_train.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("19.3579", "19.3578"), encoding="utf-8")
    record = next(iter(dataset.records))
    with pytest.raises(
        TimeFFormatError, match=r"scenario_train.jsonl \(row 0\).*series differs from the converted row"
    ):
        record.signals[0].to_numpy()


def _knowledge_values(count: int, text: str = "0.1000") -> str:
    return ", ".join([text] * count)


@pytest.mark.parametrize(
    ("part", "field", "value", "message"),
    [
        ("scenario_train", "question", "No marker here, with length of 4: x", "exactly once"),
        ("scenario_train", "question", "with length of 4: <ts><ts/> and <ts><ts/>", "exactly once"),
        (
            "scenario_train",
            "question",
            "This is a metric with length of 5: <ts><ts/>.",
            "does not state the series' length, 4",
        ),
        ("scenario_train", "timeseries2", [[1.0], [2.0]], r"(?s)timeseries2.*at most 1 item"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("scenario_train", "timeseries2", [[1.0, float("nan"), 2.0, 3.0]], r"(?s)timeseries2.*finite number"),
        # A JSON true is an int to Python, and an int is not a published float.
        ("scenario_train", "timeseries2", [[1.0, True, 2.0, 3.0]], r"(?s)timeseries2.*valid number"),
        ("scenario_train", "label", "Maybe", "label 'Maybe' is not one of"),
        ("scenario_train", "output", _MISSING, "output must be present exactly on scenario rows"),
        ("scenario_train", "task", "ECG", "a scenario file holds a 'ECG' row"),
        ("scenario_train", "task", "Forecasting", r"(?s)task.*Input should be"),
        ("scenario_train", "step2_label", None, r"(?s)step2_label.*valid string"),
        ("scenario_train", "cot", " ", r"(?s)cot.*should match pattern"),
        # A field the release doesn't have is rejected, not dropped.
        ("scenario_train", "id", 3, r"(?s)id.*Extra inputs are not permitted"),
        # The row's reference comes from where it was read, never from the JSON.
        ("scenario_train", "ref", "x", "a published row has no 'ref' key"),
        ("scenario_train", None, "x", "expected a row object"),
        ("knowledge_train", "timeseries2", _knowledge_values(500, "0.1"), "four-decimal numbers"),
        ("knowledge_train", "timeseries2", [[0.1] * 500], "must be a string of numbers"),
        ("knowledge_train", "output", "x", "output must be present exactly on scenario rows"),
        ("knowledge_train", "label", "Normal Sinus Rhythm", "is not one of"),
    ],
)
def test_malformed_rows_have_source_context(tmp_path, part, field, value, message):
    data = _fixture_rows(part)
    if field is None:
        data[0] = value
    elif value is _MISSING:
        del data[0][field]
    else:
        data[0][field] = value
    path = tmp_path / f"{part}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data) + "\n", encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=message) as error:
        _FixtureConnector().convert([HubFileRef(path, part)])
    assert f"{path} (row 0)" in str(error.value)


@pytest.mark.parametrize("label", ["two", "-1", "1e3"])
def test_an_inferential_calculation_label_must_be_a_number(tmp_path, label):
    data = _fixture_rows("scenario_train")
    data[2]["label"] = label
    path = tmp_path / "scenario_train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=r"(?s)\(row 2\).*inferential calculation label must be a number"):
        _FixtureConnector().convert([HubFileRef(path, "scenario_train")])


def test_a_knowledge_series_of_another_length_fails(tmp_path):
    data = _fixture_rows("knowledge_train")
    data[0]["timeseries2"] = _knowledge_values(499)
    path = tmp_path / "knowledge_train.jsonl"
    path.write_text(json.dumps(data[0]), encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=r"\(row 0\): a ecg series has 500 values, not 499"):
        _FixtureConnector().convert([HubFileRef(path, "knowledge_train")])


def test_a_file_that_is_not_json_values_fails_with_its_row(tmp_path):
    path = tmp_path / "scenario_test.jsonl"
    # The anonymous repository's copy of the knowledge train file has a stray letter before one row.
    lines = (_FIXTURES / "scenario_test.jsonl").read_text(encoding="utf-8").splitlines()
    path.write_text(f"{lines[0]}\nc{lines[1]}\n", encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=rf"{re.escape(str(path))} \(row 1\)"):
        _FixtureConnector().convert([HubFileRef(path, "scenario_test")])
