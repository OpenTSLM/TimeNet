from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, ForecastingTask, Split, Task, TimeInterval, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events import archive, connector as module
from timenet_connectors.datasets.trace.weather_events.archive import FILES, Kind, SplitName, TraceFile
from timenet_connectors.datasets.trace.weather_events.connector import TraceWeatherEventsConnector


_CHANNELS = ("temperature", "precipitation", "relative_humidity", "visibility", "wind_u", "wind_v", "sky_code")
_SKY_LABELS = [f"{n} oktas" if n != 1 else "1 okta" for n in range(9)] + ["sky obscured", "partial obscuration"]
_CADENCE = {"7day_hourly": timedelta(hours=1), "28day_4hourly": timedelta(hours=4), "180day_daily": timedelta(days=1)}
_START = datetime(2023, 1, 3)
# EVENT_MAP in the authors' src/common.py, in label order.
_VOCABULARY = [
    "Lightning",
    "Debris Flow",
    "Flash Flood",
    "Heavy Rain",
    "Tornado",
    "Funnel Cloud",
    "Hail",
    "Flood",
    "Thunderstorm Wind",
]
# The authors' thresholds, which the fixture windows are long enough to meet or just miss.
_MIN_VALUES = 100
_HORIZON = 24
_HOUR_US = 3_600_000_000


def _steps(count: int, *, start: int = 0, gap_at: int | None = None) -> tuple[int, ...]:
    """Return ``count`` consecutive steps from ``start``, leaving out the step at position ``gap_at``."""
    steps = list(range(start, start + count + (gap_at is not None)))
    if gap_at is not None:
        del steps[gap_at]
    return tuple(steps)


_FULL_WEEK = _steps(168)


def _window(station: str, mode: str, seed: int, location, *, steps=_FULL_WEEK, **extra) -> dict:
    """Return one published window: values derive from ``seed``, timestamps from ``steps`` on the cadence."""
    values = {
        name: [round(seed + 0.25 * index + channel, 2) for index in range(len(steps))]
        for channel, name in enumerate(_CHANNELS[:-1])
    }
    return {
        "station_id": station,
        "latitude": 36.95,
        "longitude": -120.06,
        "location": location,
        "mode": mode,
        "DATE": [(_START + step * _CADENCE[mode]).isoformat() for step in steps],
        **values,
        "sky_code": [(seed + index) % 10 for index in range(len(steps))],
        **extra,
    }


# Windows 12 and 15 hold the same measurements: two events followed the same week at station S2.
_COPY = {"station": "S2", "mode": "7day_hourly", "seed": 12}
_WINDOWS = {
    "pretrain/train": {
        # Exactly 100 values: the shortest window that gets a task. Lightning is label 0.
        "0": _window("S1", "7day_hourly", 0, "MADERA,CALIFORNIA", steps=_steps(_MIN_VALUES), event_type=0),
        "1": _window("S1", "28day_4hourly", 1, "MADERA,CALIFORNIA", event_type=0),
        "12": _window(**_COPY, location="FILLMORE,NEBRASKA", event_type=8),
        # A gap at step 2, and no event.
        "30": _window("S3", "7day_hourly", 30, "KERN,CALIFORNIA", steps=_steps(120, gap_at=2), event_type=-100),
        # A daily window with a gap, the third position of its anchor triplet.
        "2": _window("S1", "180day_daily", 2, "MADERA,CALIFORNIA", steps=_steps(150, gap_at=2), event_type=-100),
        # One value too short for a task, though labelled.
        "33": _window("S3", "7day_hourly", 33, "KERN,CALIFORNIA", steps=_steps(_MIN_VALUES - 1), event_type=2),
    },
    "pretrain/val": {"15": _window(**_COPY, location="FURNAS,NEBRASKA", event_type=6)},
    "pretrain/test": {
        "3": _window("S1", "7day_hourly", 3, "MADERA,CALIFORNIA", steps=_steps(_MIN_VALUES, start=20), event_type=2)
    },
    "forecasting/train": {
        "60": _window("S4", "7day_hourly", 60, ["Doña Ana County", "New Mexico"]),
        # One value too short for a forecast.
        "61": _window("S4", "28day_4hourly", 61, ["Doña Ana County", "New Mexico"], steps=_steps(_MIN_VALUES - 1)),
    },
    "forecasting/val": {
        "63": _window("S4", "7day_hourly", 63, "DOÑA ANA,NEW MEXICO", steps=_steps(_MIN_VALUES, start=9))
    },
    "forecasting/test": {
        "66": _window("S4", "7day_hourly", 66, "DOÑA ANA,NEW MEXICO", steps=_steps(120, start=20, gap_at=2))
    },
}
_REPORTS = {
    "retrieval/train": {"0": (1001, 0), "1": (1001, 0), "12": (1012, 8), "30": None},
    "retrieval/test": {"3": (1003, 2), "15": (1015, 6), "60": None},
}
_EVENT = pa.struct([("event_id", pa.int64()), ("event_type", pa.int64()), ("narrative", pa.string())])
_DESCRIPTION = pa.struct([(name, pa.string()) for name in ("DATE", "labels", "location", *_CHANNELS)])


def _description(key: str) -> dict:
    return {name: f"{name} of {key}" for name in ("DATE", "labels", "location", *_CHANNELS)}


def _text(key: str) -> str:
    """Return a fixture row's description as the authors' generate_dsp composes it, spacing included."""
    channels = " \n ".join(f"{name} of {key}" for name in _CHANNELS)
    return (
        f"Weather time series location: location of {key} Time range: DATE of {key} "
        f"The weather is labels of {key}. {channels}"
    )


def _channel_sentences(task) -> list[tuple[str, str]]:
    """Return a retrieval task's channel sentences as (channel, sentence) pairs, in task order."""
    return [(a.metadata["channel"], a.value) for a in task.annotations if a.key == "channel_description"]


def _write_retrieval(root: Path, split: str, rows: dict, *, narrative: str = "Report") -> None:
    path = root / "dataset" / "retrieval" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        None if event is None else {"event_id": event[0], "event_type": event[1], "narrative": f"{narrative} {key}."}
        for key, event in rows.items()
    ]
    table = pa.table(
        {
            "id": pa.array([int(key) for key in rows], pa.int64()),
            "description": pa.array([_description(key) for key in rows], _DESCRIPTION),
            "events": pa.array(events, _EVENT),
            "timeseries": pa.array([b"unused"] * len(rows), pa.binary()),
        }
    )
    pq.write_table(table, path)


def _write(root: Path, windows=_WINDOWS, reports=_REPORTS) -> list[TraceFile]:
    for part, rows in windows.items():
        path = root / "dataset" / f"{part}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    for part, rows in reports.items():
        _write_retrieval(root, part.split("/")[1], rows)
    return [
        TraceFile(root / "dataset" / f"{kind}" / f"{split}.{'parquet' if kind == 'retrieval' else 'json'}", kind, split)
        for kind, split in FILES
    ]


def _convert(tmp_path, **kwargs):
    return TraceWeatherEventsConnector().convert(_write(tmp_path, **kwargs))


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """The fixture archive, converted once for the tests that only read the dataset."""
    return _convert(tmp_path_factory.mktemp("trace"))


def _windows_with(part: str, key: str, row) -> dict:
    """Return a copy of the fixture windows with the window ``key`` of ``part`` set to ``row``."""
    windows = {name: dict(rows) for name, rows in _WINDOWS.items()}
    windows[part][key] = row
    return windows


def _record_of(dataset, window_id: int):
    (record,) = [record for record in dataset.records if record.metadata["window_id"] == window_id]
    return record


_T = TypeVar("_T", bound=Task)


def _tasks(dataset, kind: type[_T]) -> dict[int, _T]:
    return {task.metadata["window_id"]: task for task in dataset.iter_tasks() if isinstance(task, kind)}


def _task_key(task) -> tuple[str, int]:
    # A window has at most one task of each type, so its type and window id name a task.
    return type(task).__name__, task.metadata["window_id"]


def _offsets(signal) -> list[int]:
    axis = signal.time_axis
    if isinstance(axis, RegularAxis):
        return [axis.time_offset_us(index) for index in range(signal.n_values)]
    return signal.time_offsets_us().tolist()


def _read_all_values(dataset) -> None:
    for record in dataset.records:
        for signal in record.signals:
            signal.to_numpy()


def test_identical_windows_get_their_own_records_and_name_their_original(converted):
    dataset = converted

    original, copy = _record_of(dataset, 12), _record_of(dataset, 15)
    assert original is not copy
    window_keys = ("window_id", "anchor", "source_file", "county", "event_type", "copy_of")
    assert {key: original.metadata[key] for key in window_keys} == {
        "window_id": 12,
        "anchor": 4,
        "source_file": "dataset/pretrain/train.json",
        "county": {"name": "FILLMORE", "state": "NEBRASKA"},
        "event_type": 8,
        "copy_of": None,
    }
    assert {key: copy.metadata[key] for key in window_keys} == {
        "window_id": 15,
        "anchor": 5,
        "source_file": "dataset/pretrain/val.json",
        "county": {"name": "FURNAS", "state": "NEBRASKA"},
        "event_type": 6,
        "copy_of": 12,
    }
    # The copy holds the same series as its original, in Signals of its own.
    for kept, copied in zip(original.signals, copy.signals, strict=True):
        assert kept is not copied
        np.testing.assert_array_equal(kept.to_numpy(), copied.to_numpy())
    # No other window has a copy.
    assert {record.metadata["window_id"] for record in dataset.records if record.metadata["copy_of"] is not None} == {
        15
    }
    # A window without an event-type task keeps the label its file states: -100, or a label too short for a task.
    assert _record_of(dataset, 30).metadata["event_type"] == -100
    assert _record_of(dataset, 33).metadata["event_type"] == 2
    # A forecasting window publishes no label.
    assert _record_of(dataset, 60).metadata["event_type"] is None
    # Each window keeps its own split and targets its own Record.
    events = _tasks(dataset, ClassificationTask)
    assert (events[12].split, events[15].split) == (Split.TRAIN, Split.VALIDATION)
    retrievals = _tasks(dataset, TSCorrespondenceTask)
    assert (retrievals[12].split, retrievals[15].split) == (Split.TRAIN, Split.TEST)
    assert (retrievals[12].targets, retrievals[15].targets) == ((original,), (copy,))


def test_every_window_becomes_a_record_with_its_signals(converted):
    dataset = converted

    rows = {key: row for part in _WINDOWS.values() for key, row in part.items()}
    assert sorted(record.metadata["window_id"] for record in dataset.records) == sorted(map(int, rows))
    for record in dataset.records:
        window_id = record.metadata["window_id"]
        assert isinstance(window_id, int)
        row = rows[str(window_id)]
        # The station is TRACE's own field, beside where it stands, its sampling, its dates and the window.
        assert record.metadata.keys() == {
            "station",
            "coordinates",
            "sampling",
            "window_start",
            "window_end",
            "window_id",
            "anchor",
            "source_file",
            "county",
            "event_type",
            "copy_of",
        }
        assert record.metadata["station"] == row["station_id"]
        assert record.metadata["coordinates"] == {"latitude": row["latitude"], "longitude": row["longitude"]}
        assert record.metadata["sampling"] == row["mode"]
        # The source states no time zone, so the Record has no absolute anchor, only its published dates.
        assert not record.has_absolute_time
        assert (record.metadata["window_start"], record.metadata["window_end"]) == (row["DATE"][0], row["DATE"][-1])
        # The dates repeat what a description starts with, so they stay out of the annotations.
        assert not record.annotations
        (source,) = record.sources
        # The station id is a prior on the label, so it stays out of the source name.
        assert (source.name, source.metadata) == ("station", {"channel_order": list(_CHANNELS)})
        assert [signal.name for signal in source.signals] == list(_CHANNELS)
        (axis,) = {id(signal.time_axis): signal.time_axis for signal in record.signals}.values()
        assert isinstance(axis, IrregularAxis if window_id in {2, 30, 66} else RegularAxis)
        first = datetime.fromisoformat(row["DATE"][0])
        want_offsets = [(datetime.fromisoformat(date) - first) // timedelta(microseconds=1) for date in row["DATE"]]
        for signal in record.signals:
            expected = (
                np.asarray([_SKY_LABELS[code] for code in row[signal.name]])
                if signal.name == "sky_code"
                else np.asarray(row[signal.name], dtype=np.float32)
            )
            np.testing.assert_array_equal(signal.to_numpy(), expected)
            assert _offsets(signal) == want_offsets

    # The event types are the event-type tasks' label set, so the dataset needs no annotation for them.
    assert not dataset.annotations


def test_classified_windows_get_an_event_type_task_with_their_report(converted):
    dataset = converted

    events = _tasks(dataset, ClassificationTask)
    # 30 has no event, and 33 is too short.
    assert sorted(events) == [0, 1, 3, 12, 15]
    for window_id, task in events.items():
        report = next(part[str(window_id)] for part in _REPORTS.values() if str(window_id) in part)
        assert report is not None
        event_id, event_type = report
        assert task.targets == (_VOCABULARY[event_type],)
        assert task.label_set is not None
        assert (task.label_set.name, task.label_set.names, task.label_set.multi_label) == (
            "event_type",
            tuple(_VOCABULARY),
            False,
        )
        # The authors' classifier sees only the series.
        assert (task.prompt, task.input_annotations, task.scope) == (None, (), None)
        assert task.rationale == f"The weather event is {_VOCABULARY[event_type]}. Report {window_id}."
        assert task.metadata["event_id"] == event_id
        assert task.metadata["event_type_index"] == event_type


def test_every_retrieval_row_becomes_a_retrieval_task(converted):
    dataset = converted

    retrievals = _tasks(dataset, TSCorrespondenceTask)
    assert sorted(retrievals) == [0, 1, 3, 12, 15, 30, 60]
    test_records = {_record_of(dataset, window_id).id for window_id in (3, 15, 60)}
    for window_id, task in retrievals.items():
        record = _record_of(dataset, window_id)
        # The query is the description alone, and the answer is the window's Record.
        assert (task.prompt, task.inputs, task.targets) == (_text(str(window_id)), (), (record,))
        assert (task.rationale, task.input_annotations) == (None, ())
        # The seven channel sentences, in channel order, as the task's own annotations.
        assert _channel_sentences(task) == [(name, f"{name} of {window_id}") for name in _CHANNELS]
        # Test tasks rank the test windows' Records. Train tasks pass none, which TimeF reads as the whole dataset.
        pool = {candidate.id for candidate in task.candidate_records}
        assert pool == (test_records if task.split == Split.TEST else set())
    # An event row's task carries its event id, which joins it to the row's event-type task.
    assert retrievals[12].metadata["event_id"] == 1012
    assert "event_id" not in retrievals[30].metadata
    # Both published forms split into a name and a state, as published.
    assert _record_of(dataset, 60).metadata["county"] == {"name": "Doña Ana County", "state": "New Mexico"}
    assert _record_of(dataset, 63).metadata["county"] == {"name": "DOÑA ANA", "state": "NEW MEXICO"}


def test_forecasting_windows_forecast_their_last_values(converted):
    dataset = converted

    forecasts = _tasks(dataset, ForecastingTask)
    # 61 is too short.
    assert sorted(forecasts) == [60, 63, 66]
    assert {window_id: task.split for window_id, task in forecasts.items()} == {
        60: Split.TRAIN,
        63: Split.VALIDATION,
        66: Split.TEST,
    }
    for task in forecasts.values():
        (record,) = task.inputs
        assert (task.prompt, task.input_annotations) == (None, ())
        scope, targets = task.scope, task.targets
        assert isinstance(scope, TimeInterval)
        assert targets is not None
        (target,) = targets
        assert isinstance(target, TimeInterval)
        for signal in record.signals:
            # The scope ends where the target starts: the model sees every value but the last 24.
            assert signal.step_range(scope) == (0, signal.n_values - _HORIZON)
            assert signal.step_range(target) == (signal.n_values - _HORIZON, signal.n_values)
    # Window 66 has hours 0, 1, then 3 to 120 after its first timestamp, so its last 24 values span hours 97 to 120.
    assert forecasts[66].targets == (TimeInterval(start_us=97 * _HOUR_US, end_us=120 * _HOUR_US + 1),)
    assert forecasts[66].scope == TimeInterval(start_us=0, end_us=97 * _HOUR_US)


def test_storage_round_trip_keeps_values_timestamps_and_tasks(tmp_path):
    original = _convert(tmp_path)
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("trace/weather-events", "1.0.0")) as reader:
        restored = reader.read()

    rows = {key: row for part in _WINDOWS.values() for key, row in part.items()}
    converted = {record.id: record for record in original.records}
    assert {record.id for record in restored.records} == converted.keys()
    for record in restored.records:
        # The writer pulls the values through the lazy loaders, which must read each window's own bytes.
        signals = {signal.name: signal for signal in converted[record.id].signals}
        for signal in record.signals:
            np.testing.assert_array_equal(signal.to_numpy(), signals[signal.name].to_numpy())
        # The published timestamps rebuild from the first one and the local-clock offsets.
        start = record.metadata["window_start"]
        assert isinstance(start, str)
        first = datetime.fromisoformat(start)
        rebuilt = [(first + timedelta(microseconds=offset)).isoformat() for offset in _offsets(record.signals[0])]
        assert rebuilt == rows[str(record.metadata["window_id"])]["DATE"]

    # The writer drew the task stream once already, so this pass checks that it streams again.
    task_keys = [_task_key(task) for task in original.iter_tasks()]
    assert len(task_keys) == len(set(task_keys)) == 5 + 7 + 3
    assert sorted(map(_task_key, restored.tasks)) == sorted(task_keys)
    # The reader gives every event-type task the one stored label set.
    label_sets = {id(task.label_set) for task in restored.tasks if isinstance(task, ClassificationTask)}
    assert len(label_sets) == 1


def test_a_window_file_that_changed_after_conversion_fails_on_read(tmp_path):
    dataset = _convert(tmp_path)
    path = tmp_path / "dataset" / "pretrain" / "train.json"
    # Same length, other value: only a check of the values can see it.
    path.write_text(path.read_text().replace("0.25", "0.75"))
    with pytest.raises(TimeFFormatError, match=r"train\.json \(bytes .*the window differs"):
        _read_all_values(dataset)


def test_a_window_file_that_no_longer_parses_fails_on_read(tmp_path):
    dataset = _convert(tmp_path)
    path = tmp_path / "dataset" / "pretrain" / "train.json"
    path.write_text(path.read_text().replace('"temperature"', '"temperatura"'))
    with pytest.raises(TimeFFormatError, match="changed since conversion: missing field 'temperature'"):
        _read_all_values(dataset)


@pytest.mark.parametrize("edit", ["narrative", "rows", "invalid"])
def test_a_retrieval_file_that_changed_after_conversion_fails_on_read(tmp_path, edit):
    dataset = _convert(tmp_path)
    reports = _REPORTS["retrieval/train"]
    if edit == "narrative":
        _write_retrieval(tmp_path, "train", reports, narrative="Changed")
    elif edit == "invalid":
        _write_retrieval(tmp_path, "train", reports, narrative=" ")
    else:
        _write_retrieval(tmp_path, "train", {**reports, "99": None})
    with pytest.raises(TimeFFormatError, match=r"train\.parquet: changed since it was checked"):
        list(dataset.iter_tasks())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("DATE", ["2023-01-03T00:00:00", "2023-01-03T00:30:00", "2023-01-03T01:00:00"], "7day_hourly cadence"),
        ("DATE", ["2023-01-03T01:00:00", "2023-01-03T00:00:00", "2023-01-03T02:00:00"], "7day_hourly cadence"),
        # NumPy parses this; the format check must reject it before it does.
        ("DATE", ["NaT", "2023-01-03T01:00:00", "2023-01-03T02:00:00"], r"DATE\[0\]: expected YYYY-MM-DDTHH"),
        ("DATE", ["2023-02-30T00:00:00", "2023-02-30T01:00:00", "2023-02-30T02:00:00"], "DATE: .*2023-02-30"),
        ("DATE", [], "DATE: expected a non-empty list"),
        ("temperature", [1.0, 2.0], "temperature must be a list of 3 values"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("temperature", [1.0, float("nan"), 2.0], "temperature: expected finite numbers"),
        # A JSON true is an int to Python, so a number check must reject it explicitly.
        ("temperature", [1.0, True, 2.0], "temperature: expected finite numbers"),
        ("temperature", [1.0, "2.0", 2.0], "temperature: expected finite numbers"),
        ("sky_code", [0, 1, 11], "sky_code: expected integer codes from 0 to 10"),
        ("sky_code", [0, 1, -1], "sky_code: expected integer codes from 0 to 10"),
        ("sky_code", [0, 1, 2.0], "sky_code: expected integer codes from 0 to 10"),
        ("sky_code", [0, 1, 10**30], "sky_code: expected integer codes from 0 to 10, got a value out of range"),
        ("temperature", [1.0, 2.0, 10**400], "temperature: expected finite numbers, got a value out of range"),
        ("mode", "28day_4hourly", "mode '28day_4hourly' does not match the id's anchor position"),
        ("mode", "weekly", "mode: expected one of"),
        ("event_type", True, "event_type: expected an integer, got True"),
        ("event_type", 9, "event_type: no event type is numbered 9"),
        # A pretrain window must publish its label.
        ("event_type", None, "event_type: expected an integer, got None"),
        ("station_id", " ", "station_id: expected text"),
        ("latitude", 91.0, "latitude: expected a number from -90 to 90"),
        ("longitude", float("nan"), "longitude: expected a number from -180 to 180"),
        # A county must split into a name and a state.
        ("location", "MADERA CALIFORNIA", r'location: expected "COUNTY,STATE"'),
        ("location", "MADERA,", "location state: expected text"),
        ("location", ["MADERA"], r'location: expected "COUNTY,STATE"'),
    ],
)
def test_malformed_windows_have_source_context(tmp_path, field, value, message):
    # A three-value window: the checks fail before any length threshold applies.
    row = _window("S1", "7day_hourly", 0, "MADERA,CALIFORNIA", steps=(0, 1, 2), event_type=0)
    if field == "DATE":
        row = {**row, **{name: row[name][: len(value)] for name in _CHANNELS}}
    row[field] = value
    with pytest.raises(TimeFValidationError, match=message) as error:
        _convert(tmp_path, windows=_windows_with("pretrain/train", "0", row))
    assert "train.json (key 0)" in str(error.value)


@pytest.mark.parametrize("field", ["station_id", "DATE", "visibility", "location", "mode", "event_type"])
def test_a_window_without_a_field_names_it(tmp_path, field):
    row = {name: value for name, value in _WINDOWS["pretrain/train"]["0"].items() if name != field}
    with pytest.raises(TimeFValidationError, match=f"train.json \\(key 0\\): missing field '{field}'"):
        _convert(tmp_path, windows=_windows_with("pretrain/train", "0", row))


@pytest.mark.parametrize(
    ("part", "key", "edit", "message"),
    [
        ("pretrain/train", "0", lambda row: [1, 2, 3], r"train.json \(key 0\): expected a JSON object, got list"),
        # Only pretrain windows publish a label.
        (
            "forecasting/train",
            "60",
            lambda row: {**row, "event_type": 3},
            r"train.json \(key 60\): a forecasting window must not publish",
        ),
        # Copies of one measurement must stand at one place.
        (
            "pretrain/val",
            "15",
            lambda row: {**row, "latitude": 40.6},
            r"val.json \(key 15\): same measurements as window 12, but at .*40.6",
        ),
        # A second window 0, in another file.
        ("pretrain/test", "0", lambda row: _WINDOWS["pretrain/train"]["0"], "window 0 also appears in"),
    ],
)
def test_windows_that_contradict_their_file_or_copies_fail(tmp_path, part, key, edit, message):
    row = edit(_WINDOWS[part].get(key))
    with pytest.raises(TimeFValidationError, match=message):
        _convert(tmp_path, windows=_windows_with(part, key, row))


def test_window_ids_must_be_canonical(tmp_path):
    windows = _windows_with("pretrain/train", "012", _WINDOWS["pretrain/train"]["12"])
    del windows["pretrain/train"]["12"]
    with pytest.raises(TimeFValidationError, match="canonical decimal window id, got '012'"):
        _convert(tmp_path, windows=windows)


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        # A classified window without a retrieval row, or with another type's report.
        (
            {"retrieval/train": {"0": (1001, 0), "1": (1001, 0), "30": None}},
            r"key 12\): classified window has no retrieval row",
        ),
        (
            {"retrieval/train": {"0": (1001, 0), "1": (1001, 0), "12": (1012, 6), "30": None}},
            r"key 12\): no event report with event_type THUNDERSTORM_WIND",
        ),
        # Lightning is label 0, so the message must still name it.
        (
            {"retrieval/train": {"0": (1001, 2), "1": (1001, 0), "12": (1012, 8), "30": None}},
            r"key 0\): no event report with event_type LIGHTNING",
        ),
        # A report on a window the authors don't classify.
        (
            {"retrieval/train": {"0": (1001, 0), "1": (1001, 0), "12": (1012, 8), "30": (1030, 2)}},
            r"key 30\): event report for",
        ),
        # One window id in both retrieval files.
        (
            {"retrieval/test": {"3": (1003, 2), "15": (1015, 6), "60": None, "12": (1012, 8)}},
            "window id 12 appears in more than one row",
        ),
        # A description without a window.
        (
            {"retrieval/train": {"0": (1001, 0), "1": (1001, 0), "12": (1012, 8), "99": None}},
            "1 retrieval rows have no window, such as 99",
        ),
    ],
)
def test_event_reports_and_descriptions_must_match_windows(tmp_path, reports, message):
    with pytest.raises(TimeFValidationError, match=message):
        _convert(tmp_path, reports={**_REPORTS, **reports})


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("description", {**_description("0"), "labels": " "}, "labels: expected text"),
        ("description", {**_description("0"), "DATE": None}, "DATE: expected text"),
        ("events", {"event_id": 1, "event_type": 1, "narrative": " "}, "narrative: expected text"),
        ("events", {"event_id": 1, "event_type": 99, "narrative": "x"}, "event_type: no event type is numbered 99"),
    ],
)
def test_malformed_retrieval_rows_name_their_file_and_id(tmp_path, column, value, message):
    files = _write(tmp_path)
    path = tmp_path / "dataset" / "retrieval" / "train.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][column] = value
    schema = pq.read_schema(path)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    with pytest.raises(TimeFValidationError, match=rf"train\.parquet \(id 0\): .*{message}"):
        TraceWeatherEventsConnector().convert(files)


@pytest.mark.parametrize("column", ["id", "description", "events"])
def test_a_retrieval_file_without_a_needed_column_is_a_format_error(tmp_path, column):
    files = _write(tmp_path)
    path = tmp_path / "dataset" / "retrieval" / "train.parquet"
    pq.write_table(pq.read_table(path).drop_columns([column]), path)
    with pytest.raises(TimeFFormatError, match=rf"train\.parquet: no column \['{column}'\]"):
        TraceWeatherEventsConnector().convert(files)


@pytest.mark.parametrize(
    ("windows", "reports", "message"),
    [
        # A truncated file can still be valid JSON or parquet.
        ({**_WINDOWS, "forecasting/test": {}}, _REPORTS, r"forecasting/test\.json: holds no windows"),
        (_WINDOWS, {**_REPORTS, "retrieval/test": {}}, r"retrieval/test\.parquet: holds no retrieval rows"),
    ],
)
def test_an_empty_split_file_fails(tmp_path, windows, reports, message):
    with pytest.raises(TimeFValidationError, match=message):
        _convert(tmp_path, windows=windows, reports=reports)


def test_a_file_that_is_not_parquet_fails_as_a_format_error(tmp_path):
    files = _write(tmp_path)
    (tmp_path / "dataset" / "retrieval" / "train.parquet").write_text("not parquet")
    with pytest.raises(TimeFFormatError, match=r"train\.parquet"):
        TraceWeatherEventsConnector().convert(files)


def test_split_file_references_must_be_the_archives_files():
    # A repeated file: one sorted comparison also rejects a missing or unknown one.
    kinds = [*FILES, (Kind.PRETRAIN, SplitName.TRAIN)]
    files = [TraceFile(Path(f"{kind}/{split}"), kind, split) for kind, split in kinds]
    with pytest.raises(TimeFValidationError, match="requires each of the split files"):
        TraceWeatherEventsConnector().convert(files)


def test_download_locates_split_files_in_the_pinned_archive(monkeypatch, tmp_path):
    calls = []

    async def fake_ensure_archive(url, target, *, filename, sha256):
        calls.append((url, target, filename, sha256))
        return tmp_path / "archive"

    monkeypatch.setattr(module, "ensure_archive", fake_ensure_archive)
    with pytest.raises(TimeNetDownloadError, match="TRACE archive is missing"):
        TraceWeatherEventsConnector().download(tmp_path / "cache")

    _write(tmp_path / "archive")
    files = TraceWeatherEventsConnector().download(tmp_path / "cache")
    assert calls[-1] == (archive.SOURCE_URL, tmp_path / "cache", "trace_dataset.zip", archive.SOURCE_SHA256)
    assert [file.path.relative_to(tmp_path / "archive").as_posix() for file in files] == [
        "dataset/pretrain/train.json",
        "dataset/pretrain/val.json",
        "dataset/pretrain/test.json",
        "dataset/forecasting/train.json",
        "dataset/forecasting/val.json",
        "dataset/forecasting/test.json",
        "dataset/retrieval/train.parquet",
        "dataset/retrieval/test.parquet",
    ]
    assert [(file.kind, file.split) for file in files] == list(FILES)
