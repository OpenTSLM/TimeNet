import csv
import gzip
import json
from pathlib import Path

import pytest

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, TimeInterval
from timenet.types.text import read_value
from timenet_connectors.datasets.oxwearables.capture24.connector import (
    Capture24Connector,
    Capture24Source,
)


_Row = tuple[str, str, str, str, str]

_SLEEP = "7030 sleeping;MET 0.95"
_SITTING = "occupation;office and administrative support;11580 office/computer work general;MET 1.5"
_SCHEMES = ("label:WillettsSpecific2018", "label:Walmsley2020")
_DICTIONARY = {
    _SLEEP: {"label:WillettsSpecific2018": "sleep", "label:Walmsley2020": "sleep"},
    _SITTING: {"label:WillettsSpecific2018": "sitting", "label:Walmsley2020": "sedentary"},
}
_P001_ROWS: list[_Row] = [
    ("2016-11-13 02:18:00.000000", "-0.46669036", "-0.5333412", "0.65847206", _SLEEP),
    ("2016-11-13 02:18:00.010000", "1", "2", "3", _SLEEP),
    ("2016-11-13 02:18:00.020000", "0.5", "0.5", "0.5", ""),
    ("2016-11-13 02:18:00.030000", "0.25", "0.25", "0.25", _SITTING),
    ("2016-11-13 02:18:00.040000", "0", "0", "-1", _SLEEP),
]
_P002_ROWS: list[_Row] = [
    ("2016-05-21 01:53:00.000000", "0", "0", "1", ""),
    ("2016-05-21 01:53:00.010000", "0", "0", "1", ""),
]


def _write_participant(root: Path, pid: str, rows: list[_Row]) -> None:
    with gzip.open(root / f"{pid}.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "x", "y", "z", "annotation"))
        writer.writerows(rows)


def _write_release(root: Path, p001_rows: list[_Row] = _P001_ROWS) -> Capture24Source:
    root.mkdir(exist_ok=True)
    (root / "metadata.csv").write_text("pid,age,sex\nP001,38-52,F\nP002,30-37,M\n", encoding="utf-8")
    with (root / "annotation-label-dictionary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("annotation", *_SCHEMES))
        writer.writerows((text, *labels.values()) for text, labels in _DICTIONARY.items())
    _write_participant(root, "P001", p001_rows)
    _write_participant(root, "P002", _P002_ROWS)
    return Capture24Source(root=root)


def _scope(task: ClassificationTask) -> tuple[int, int]:
    assert isinstance(task.scope, TimeInterval)
    return task.scope.start_us, task.scope.exclusive_end


def test_each_participant_is_one_record_of_three_axes_at_100_hz(tmp_path):
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24")])
    first, second = dataset.records
    assert first.metadata == {"participant_id": "P001"}
    assert second.metadata == {"participant_id": "P002"}
    signals = {signal.name: signal for signal in first.signals}
    assert sorted(signals) == ["x", "y", "z"]
    assert signals["x"].to_arrow().to_pylist() == [-0.46669036, 1.0, 0.5, 0.25, 0.0]
    assert signals["z"].to_arrow().to_pylist() == [0.65847206, 3.0, 0.5, 0.25, -1.0]
    assert signals["x"].time_axis == RegularAxis.from_rate_hz(100)
    annotations = {item.key: read_value(item.value) for item in first.annotations if item.span is None}
    assert annotations == {
        "subject_ids": ["P001"],
        "age": "38-52",
        "sex": "F",
        "recording_start_local": "2016-11-13 02:18:00.000000",
    }


def test_each_annotated_run_is_one_span_and_one_task_and_blank_rows_get_neither(tmp_path):
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24")])
    first, second = dataset.records
    spans = [(read_value(item.value), item.span) for item in first.annotations if item.span is not None]
    assert spans == [
        (_SLEEP, TimeInterval.micros(0, 20_000)),
        (_SITTING, TimeInterval.micros(30_000, 40_000)),
        (_SLEEP, TimeInterval.micros(40_000, 50_000)),
    ]
    tasks = [task for task in dataset.tasks if isinstance(task, ClassificationTask)]
    assert [(task.targets, _scope(task)) for task in tasks] == [
        ((_SLEEP,), (0, 20_000)),
        ((_SITTING,), (30_000, 40_000)),
        ((_SLEEP,), (40_000, 50_000)),
    ]
    assert all(task.inputs == (first,) for task in tasks)
    assert tasks[0].label_set.names == (_SLEEP, _SITTING)
    assert not [item for item in second.annotations if item.span is not None]


def test_the_label_dictionary_is_one_dataset_annotation(tmp_path):
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24")])
    (annotation,) = dataset.annotations
    assert annotation.key == "annotation_label_dictionary"
    value = read_value(annotation.value)
    assert isinstance(value, str)
    assert json.loads(value) == _DICTIONARY


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            (1, ("2016-11-13 02:18:00.005000", "1", "2", "3", _SLEEP)),
            "row 1 is at 2016-11-13 02:18:00.005000, less than 10000 us after",
        ),
        (
            (1, ("2016-11-13 02:18:00.010000", "1", "2", "3", "walking")),
            r"annotations not in annotation-label-dictionary\.csv: \[.walking.\]",
        ),
    ],
    ids=["a step under 10 ms", "an annotation not in the dictionary"],
)
def test_a_malformed_participant_file_raises(tmp_path, change, message):
    index, row = change
    rows = list(_P001_ROWS)
    rows[index] = row
    with pytest.raises(TimeFFormatError, match=rf"P001\.csv\.gz: {message}"):
        Capture24Connector().convert([_write_release(tmp_path / "capture24", rows)])


def test_a_gap_in_the_timestamps_starts_a_new_segment_and_a_new_run(tmp_path):
    rows = [
        *_P001_ROWS[:2],
        # The recording stops for an hour, and the annotation stays the same across the gap.
        ("2016-11-13 03:18:00.020000", "0.5", "0.5", "0.5", _SLEEP),
        ("2016-11-13 03:18:00.030000", "0.25", "0.25", "0.25", _SLEEP),
    ]
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24", rows)])
    record = dataset.records[0]
    (device,) = record.sources
    first, second = device.sources
    hour_us = 3_600_000_000
    assert [signal.name for signal in first.signals] == ["x", "y", "z"]
    assert {signal.time_axis for signal in first.signals} == {RegularAxis.from_rate_hz(100)}
    assert {signal.time_axis for signal in second.signals} == {
        RegularAxis.from_rate_hz(100, offset_us=hour_us + 20_000)
    }
    assert first.signals[0].to_arrow().to_pylist() == [-0.46669036, 1.0]
    assert second.signals[0].to_arrow().to_pylist() == [0.5, 0.25]
    tasks = [task for task in dataset.tasks if isinstance(task, ClassificationTask)]
    assert [_scope(task) for task in tasks] == [(0, 20_000), (hour_us + 20_000, hour_us + 40_000)]


def test_a_non_finite_value_raises_when_the_axes_are_read(tmp_path):
    rows = list(_P001_ROWS)
    rows[1] = ("2016-11-13 02:18:00.010000", "nan", "2", "3", _SLEEP)
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24", rows)])
    with pytest.raises(TimeFFormatError, match="axis x holds a blank or non-finite value"):
        dataset.records[0].signals[0].to_arrow()


def test_a_participant_missing_from_the_metadata_raises(tmp_path):
    source = _write_release(tmp_path / "capture24")
    _write_participant(source.root, "P003", _P002_ROWS)
    with pytest.raises(TimeFFormatError, match=r"P003 has no row in metadata\.csv"):
        Capture24Connector().convert([source])


def test_the_dataset_round_trips_through_storage(tmp_path):
    dataset = Capture24Connector().convert([_write_release(tmp_path / "capture24")])
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
