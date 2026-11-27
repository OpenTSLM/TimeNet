from collections.abc import Mapping
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
from timenet_connectors.datasets.hayleyzhou1113.veritime import VeriTimeConnector
from timenet_connectors.sources import json_stream


_FIXTURES = Path(__file__).parent / "fixtures"
_PARTS = ("scenario_train", "scenario_test", "knowledge_train", "knowledge_test")
_EXPECTED_REVISION = "fc88cb1305a96a208093e13b69d766b19c84c99b"
# Marks a field the test removes from a row.
_MISSING = object()


class _FixtureConnector(VeriTimeConnector):
    """The connector, with the card beside the connector, since the tests live in a subfolder."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"


def _refs(directory: Path = _FIXTURES) -> list[HubFileRef]:
    return [HubFileRef(directory / f"{part}.jsonl", part) for part in _PARTS]


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
_EXPECTED_DATASET_ANNOTATIONS: dict[str, Any] = {
    "source_repository": "HayleyZhou1113/VeriTime",
    "source_revision": _EXPECTED_REVISION,
}
_ANOMALY = ("anomaly_detection", ("Yes", "No"))
_ECG = ("ecg", ("normal sinus rhythm", "atrial fibrillation", "other cardiac rhythms", "noise"))
_EMG = ("emg", ("Healthy", "Myopathy", "Neuropathy"))
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
# Each task's type, target and its type, and label set as (name, labels); the other fields follow the row.
_EXPECTED_TASKS: dict[tuple[str, int], tuple[type, Any, tuple[str, tuple[str, ...]] | None]] = {
    ("scenario_train", 0): (ClassificationTask, "Yes", _ANOMALY),
    ("scenario_train", 1): (AnswerTask, "Consistent market conditions", None),
    ("scenario_train", 2): (ScalarPredictionTask, 2, None),
    ("scenario_test", 0): (ClassificationTask, "No", _ANOMALY),
    # A two-part question's amplitude is a decimal label, as in the release's one such row. It stays a float.
    ("scenario_test", 1): (ScalarPredictionTask, 10641.02, None),
    ("knowledge_train", 0): (ClassificationTask, "other cardiac rhythms", _ECG),
    ("knowledge_test", 0): (ClassificationTask, "Neuropathy", _EMG),
}


def _expected_task(part: str, index: int) -> dict[str, Any]:
    row = _fixture_rows(part)[index]
    task_type, target, label_set = _EXPECTED_TASKS[part, index]
    # A process label published as null has no annotation.
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
        "label_set": label_set,
        "rationale": row["cot"],
        "metadata": {"source_task": row["task"], "part": part, "row": index},
        "annotations": annotations,
        "split": part.split("_")[1],
    }


def _label_set(task) -> tuple[str, tuple[str, ...]] | None:
    """A classification task's label set as its name and labels, or None for another task type."""
    if not isinstance(task, ClassificationTask):
        return None
    return task.label_set.name, task.label_set.names


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
        assert set(metadata) == {"kind", "rows"}
        assert str(signal.to_numpy().dtype) == "float64"
    for task in dataset.iter_tasks():
        tasks[_row_key(task)] = {
            "type": type(task),
            "inputs": len(task.inputs),
            "prompt": task.prompt,
            "input_annotations": {a.key: a.value for a in task.input_annotations},
            "targets": tuple(_typed(target) for target in task.targets),
            "label_set": _label_set(task),
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


def test_ctu_and_rcw_keep_their_cadence_and_classes(tmp_path):
    # The fixtures hold no CTU or RCW row, so turn ECG rows into them. CTU states a 2-minute cadence,
    # and RCW is every eighth sample of 2 kHz, so 250 Hz.
    rows = _fixture_rows("knowledge_train")
    rows[0].update(task="CTU", label="Laptop")
    rcw = {**rows[0], "task": "RCW", "label": "No Right Whale", "timeseries2": ", ".join(["0.5000"] * 500)}
    path = tmp_path / "knowledge_train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (rows[0], rcw)) + "\n", encoding="utf-8")
    dataset = _FixtureConnector().convert([HubFileRef(path, "knowledge_train")])
    output = _output(dataset)
    signals = {record["kind"]: record["signal"] for record in output["records"].values()}
    assert {kind: (signal[0], signal[2]) for kind, signal in signals.items()} == {
        "ctu": ("consumption", (RegularAxis, Fraction(120_000_000))),
        "rcw": ("audio", (RegularAxis, Fraction(4000))),
    }
    assert [task["label_set"] for task in output["tasks"].values()] == [
        ("ctu", ("Desktop", "Laptop")),
        ("rcw", ("Right Whale Present", "No Right Whale")),
    ]


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
        ("scenario_train", "timeseries2", [[1.0], [2.0]], "list holding one list of numbers"),
        ("scenario_train", "timeseries2", [[]], "at least one value"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("scenario_train", "timeseries2", [[1.0, float("nan"), 2.0, 3.0]], "every value must be finite"),
        # A JSON true is an int to Python, and an int is not a published float.
        ("scenario_train", "timeseries2", [[1.0, True, 2.0, 3.0]], "must hold numbers"),
        ("scenario_train", "label", "Maybe", "label 'Maybe' is not one of"),
        ("scenario_train", "output", _MISSING, r"missing fields \['output'\]"),
        ("scenario_train", "step1_label", _MISSING, r"missing fields \['step1_label'\]"),
        ("scenario_train", "task", "ECG", "a scenario file holds a 'ECG' row"),
        ("scenario_train", "task", "Forecasting", "unknown task 'Forecasting'"),
        ("scenario_train", "task", ["ECG"], "task must be nonblank text"),
        ("scenario_train", "step2_label", None, "step2_label must be nonblank text"),
        ("scenario_train", "cot", " ", "cot must be nonblank text"),
        ("scenario_train", None, "x", "expected a row object"),
        ("knowledge_train", "timeseries2", _knowledge_values(500, "abc"), "numbers separated by"),
        ("knowledge_train", "timeseries2", _knowledge_values(500, "inf"), "every value must be finite"),
        ("knowledge_train", "timeseries2", [[0.1] * 500], "must be a string of numbers"),
        ("knowledge_train", "task", "Anomaly detection", "a knowledge file holds a 'Anomaly detection' row"),
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


@pytest.mark.parametrize("label", ["two", "-1", "1e3", "nan"])
def test_an_inferential_calculation_label_must_be_a_number(tmp_path, label):
    data = _fixture_rows("scenario_train")
    data[2]["label"] = label
    path = tmp_path / "scenario_train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=r"(?s)\(row 2\).*inferential calculation label must be a number"):
        _FixtureConnector().convert([HubFileRef(path, "scenario_train")])


def test_a_file_that_is_not_json_values_fails_with_its_row(tmp_path):
    path = tmp_path / "scenario_test.jsonl"
    # A stray character before a row, as a damaged copy of a file would have, names the row it breaks.
    lines = (_FIXTURES / "scenario_test.jsonl").read_text(encoding="utf-8").splitlines()
    path.write_text(f"{lines[0]}\nc{lines[1]}\n", encoding="utf-8")
    with pytest.raises(TimeFValidationError, match=rf"{re.escape(str(path))} \(row 1\)"):
        _FixtureConnector().convert([HubFileRef(path, "scenario_test")])


def test_equal_values_of_two_kinds_stay_two_records(tmp_path):
    # An ECG and an RCW series can hold the same values, such as flat zeros, but not the same spec or rate.
    flat = ", ".join(["0.0000"] * 500)
    ecg, rcw = (
        {**_fixture_rows("knowledge_train")[0], "task": task, "label": label, "timeseries2": flat}
        for task, label in (("ECG", "noise"), ("RCW", "No Right Whale"))
    )
    path = tmp_path / "knowledge_train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (ecg, rcw)) + "\n", encoding="utf-8")
    dataset = _FixtureConnector().convert([HubFileRef(path, "knowledge_train")])
    assert [record.metadata["kind"] for record in dataset.records] == ["ecg", "rcw"]


def test_a_part_that_is_not_a_veritime_part_fails_naming_it():
    # The base rejects parts that SOURCE_FILES does not declare, so declare one the connector does not know.
    class _Extra(_FixtureConnector):
        SOURCE_FILES: ClassVar[Mapping[str, str]] = {**VeriTimeConnector.SOURCE_FILES, "other_train": "x.jsonl"}

    with pytest.raises(TimeFValidationError, match="'other_train' is not a VeriTime part"):
        _Extra().convert([HubFileRef(_FIXTURES / "scenario_train.jsonl", "other_train")])
