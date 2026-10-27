from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, ForecastingTask, Split, Task, TimeInterval, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events import archive, connector as module, records
from timenet_connectors.datasets.trace.weather_events.archive import (
    SOURCE_FILES,
    PinnedCounts,
    TraceFileRef,
    TraceSplit,
    TraceTaskKind,
)
from timenet_connectors.datasets.trace.weather_events.connector import TraceWeatherEventsConnector
from timenet_connectors.datasets.trace.weather_events.loader import TraceDatasetLoader
from timenet_connectors.datasets.trace.weather_events.records import TraceRecordMetadata
from timenet_connectors.datasets.trace.weather_events.tasks import (
    TraceEventTypeTask,
    TraceForecastTask,
    TraceRetrievalTask,
)
from timenet_connectors.datasets.trace.weather_events.vocabulary import EventLabel, SkyCover, WindowSampling
from timenet_connectors.metadata import RecordMetadata


_CHANNELS = ("temperature", "precipitation", "relative_humidity", "visibility", "wind_u", "wind_v", "sky_code")
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


@pytest.fixture(autouse=True)
def _small_windows(monkeypatch):
    # The fixtures hold a few values per window, so lower the authors' thresholds where they apply.
    monkeypatch.setattr(records, "MIN_TASK_VALUES", 3)
    monkeypatch.setattr(records, "FORECAST_HORIZON", 2)


def _window(station: str, mode: str, seed: int, location, *, steps=(0, 1, 2, 3), **extra) -> dict:
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
        "0": _window("S1", "7day_hourly", 0, "MADERA,CALIFORNIA", event_type=1),
        "1": _window("S1", "28day_4hourly", 1, "MADERA,CALIFORNIA", event_type=1),
        "12": _window(**_COPY, location="FILLMORE,NEBRASKA", event_type=8),
        # A gap at step 2, and no event.
        "30": _window("S3", "7day_hourly", 30, "KERN,CALIFORNIA", steps=(0, 1, 3, 4), event_type=-100),
        # Too short for a task, though labelled.
        "33": _window("S3", "7day_hourly", 33, "KERN,CALIFORNIA", steps=(0,), event_type=2),
    },
    "pretrain/val": {"15": _window(**_COPY, location="FURNAS,NEBRASKA", event_type=6)},
    "pretrain/test": {"3": _window("S1", "7day_hourly", 3, "MADERA,CALIFORNIA", steps=(20, 21, 22, 23), event_type=2)},
    "forecasting/train": {
        "60": _window("S4", "7day_hourly", 60, ["Doña Ana County", "New Mexico"], steps=(0, 1, 2, 3, 4)),
        "61": _window("S4", "28day_4hourly", 61, ["Doña Ana County", "New Mexico"], steps=(0, 1)),
    },
    "forecasting/val": {"63": _window("S4", "7day_hourly", 63, "DOÑA ANA,NEW MEXICO", steps=(9, 10, 11, 12, 13))},
    "forecasting/test": {"66": _window("S4", "7day_hourly", 66, "DOÑA ANA,NEW MEXICO", steps=(20, 21, 23, 24, 25))},
}
_REPORTS = {
    "retrieval/train": {"0": (1001, 1), "1": (1001, 1), "12": (1012, 8), "30": None},
    "retrieval/test": {"3": (1003, 2), "15": (1015, 6), "60": None},
}
_EVENT = pa.struct([("event_id", pa.int64()), ("event_type", pa.int64()), ("narrative", pa.string())])
_DESCRIPTION = pa.struct([(name, pa.string()) for name in ("DATE", "labels", "location", *_CHANNELS)])


def _description(key: str) -> dict:
    return {name: f"{name} of {key}" for name in ("DATE", "labels", "location", *_CHANNELS)}


def _text(key: str) -> str:
    """Return a fixture row's description as the authors' generate_dsp composes it, spacing included."""
    channels = " \n ".join(f"{name} of {key}" for name in _CHANNELS)
    return f"Weather time series location: location of {key} Time range: DATE of {key} The weather is labels of {key}. {channels}"


def _channel_sentences(task) -> list[tuple[str, str]]:
    """Return a retrieval task's channel sentences as (channel, sentence) pairs, in task order."""
    return [(a.metadata["channel"], a.value) for a in task.annotations if a.key == "channel_description"]


def _write(root: Path, windows=_WINDOWS, reports=_REPORTS, expected=None) -> list[TraceFileRef]:
    refs = []
    for part, rows in windows.items():
        task, split = part.split("/")
        path = root / "dataset" / task / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
        refs.append(
            TraceFileRef(path, TraceTaskKind(task), TraceSplit(split), None if expected is None else expected[part])
        )
    for part, rows in reports.items():
        split = part.split("/")[1]
        path = root / "dataset" / "retrieval" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "id": pa.array([int(key) for key in rows], pa.int64()),
                "description": pa.array([_description(key) for key in rows], _DESCRIPTION),
                "events": pa.array(
                    [
                        None
                        if event is None
                        else {"event_id": event[0], "event_type": event[1], "narrative": f"Report {key}."}
                        for key, event in rows.items()
                    ],
                    _EVENT,
                ),
                "timeseries": pa.array([b"unused"] * len(rows), pa.binary()),
            }
        )
        pq.write_table(table, path)
        refs.append(
            TraceFileRef(path, TraceTaskKind.RETRIEVAL, TraceSplit(split), None if expected is None else expected[part])
        )
    return refs


def _convert(tmp_path, **kwargs):
    return TraceWeatherEventsConnector().convert(_write(tmp_path, **kwargs))


def _window_of(record) -> dict[int, dict]:
    return {window["window_id"]: window for window in record.metadata["trace"]["windows"]}


def _record_of(dataset, window_id: int):
    (record,) = [record for record in dataset.records if window_id in _window_of(record)]
    return record


_T = TypeVar("_T", bound=Task)


def _tasks(dataset, kind: type[_T]) -> dict[int, _T]:
    return {task.metadata["window_id"]: task for task in dataset.iter_tasks() if isinstance(task, kind)}


def _task_key(task) -> tuple[str, int]:
    # A window has at most one task of each type, so its type and window id name a task.
    return type(task).__name__, task.metadata["window_id"]


def _offsets(axis, signal) -> list[int]:
    if isinstance(axis, RegularAxis):
        return [axis.time_offset_us(index) for index in range(signal.n_values)]
    return signal.time_offsets_us().tolist()


def test_identical_windows_share_one_record_and_keep_their_own_tasks(tmp_path):
    dataset = _convert(tmp_path)

    record = _record_of(dataset, 12)
    assert _record_of(dataset, 15) is record
    assert _window_of(record) == {
        12: {
            "window_id": 12,
            "anchor": 4,
            "source_file": "dataset/pretrain/train.json",
            "county": {"name": "FILLMORE", "state": "NEBRASKA"},
            "event_type": 8,
        },
        15: {
            "window_id": 15,
            "anchor": 5,
            "source_file": "dataset/pretrain/val.json",
            "county": {"name": "FURNAS", "state": "NEBRASKA"},
            "event_type": 6,
        },
    }
    # Each copy's county belongs to its event, so it stays in the window's metadata, not an annotation.
    assert not [a for a in record.annotations if a.key in {"county", "location"}]
    # Each window keeps its own split, though both tasks target one Record.
    events = _tasks(dataset, ClassificationTask)
    assert (events[12].split, events[15].split) == (Split.TRAIN, Split.VALIDATION)
    retrievals = _tasks(dataset, TSCorrespondenceTask)
    assert (retrievals[12].split, retrievals[15].split) == (Split.TRAIN, Split.TEST)


def test_every_distinct_window_becomes_a_record_with_its_signals(tmp_path):
    dataset = _convert(tmp_path)

    rows = {key: row for part in _WINDOWS.values() for key, row in part.items()}
    assert len(dataset.records) == len(rows) - 1
    for record in dataset.records:
        (window_id, *_) = sorted(_window_of(record))
        row = rows[str(window_id)]
        # The station is the core subject; the rest is TRACE's extension, under its own name.
        assert record.metadata.keys() == {"subject_id", "trace"}
        assert record.metadata["subject_id"] == row["station_id"]
        assert {k: v for k, v in record.metadata["trace"].items() if k != "windows"} == {
            "coordinates": {"latitude": row["latitude"], "longitude": row["longitude"]},
            "sampling": row["mode"],
        }
        # The stored metadata reads back as typed metadata, with its extension restored as its own type.
        restored = RecordMetadata.from_json(record.metadata, [TraceRecordMetadata])
        assert restored.to_json() == record.metadata
        assert restored.extension(TraceRecordMetadata).sampling is WindowSampling(row["mode"])
        # The source states no time zone, so the Record has no absolute anchor, only its published dates.
        assert not record.has_absolute_time
        notes = {a.key: a for a in record.annotations}
        assert notes.keys() == {"window_start", "window_end"}
        assert (notes["window_start"].value, notes["window_end"].value) == (row["DATE"][0], row["DATE"][-1])
        (source,) = record.sources
        # The station id is a prior on the label, so it stays out of the source name.
        assert (source.name, source.metadata) == ("station", {"channel_order": list(_CHANNELS)})
        (axis,) = {id(signal.time_axis): signal.time_axis for signal in record.signals}.values()
        gapped = window_id in {30, 66}
        assert isinstance(axis, IrregularAxis if gapped else RegularAxis)
        for signal in record.signals:
            expected = (
                np.asarray([SkyCover(code).label for code in row[signal.name]])
                if signal.name == "sky_code"
                else np.asarray(row[signal.name], dtype=np.float32)
            )
            np.testing.assert_array_equal(signal.to_numpy(), expected)

    annotations = {annotation.key: annotation.value for annotation in dataset.annotations}
    assert annotations == {"no_event_label": -100, "event_type_vocabulary": _VOCABULARY}


def test_classified_windows_get_an_event_type_task_with_their_report(tmp_path):
    dataset = _convert(tmp_path)

    events = _tasks(dataset, ClassificationTask)
    # 30 has no event, and 33 is too short.
    assert sorted(events) == [0, 1, 3, 12, 15]
    for window_id, task in events.items():
        report = next(part[str(window_id)] for part in _REPORTS.values() if str(window_id) in part)
        assert report is not None
        event_id, event_type = report
        assert task.targets == (EventLabel(event_type).label,)
        # The authors' classifier sees only the series.
        assert (task.prompt, task.input_annotations, task.scope) == (None, (), None)
        assert task.rationale == f"The weather event is {EventLabel(event_type).label}. Report {window_id}."
        assert task.metadata["event_id"] == event_id
        assert task.metadata["event_type_index"] == event_type


def test_every_retrieval_row_becomes_a_retrieval_task(tmp_path):
    dataset = _convert(tmp_path)

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
        # Test tasks rank the test windows' Records, and train tasks name no pool.
        pool = {candidate.id for candidate in task.candidate_records}
        assert pool == (test_records if task.split == Split.TEST else set())
    # An event row's task carries its event id, which joins it to the row's event-type task.
    assert retrievals[12].metadata["event_id"] == 1012
    assert "event_id" not in retrievals[30].metadata
    # Both published forms split into a name and a state, as published.
    assert _window_of(_record_of(dataset, 60))[60]["county"] == {"name": "Doña Ana County", "state": "New Mexico"}
    assert _window_of(_record_of(dataset, 63))[63]["county"] == {"name": "DOÑA ANA", "state": "NEW MEXICO"}


def test_forecasting_windows_forecast_their_last_values(tmp_path):
    dataset = _convert(tmp_path)

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
            # The scope ends where the target starts: the model sees every value but the last two.
            assert signal.step_range(scope) == (0, signal.n_values - 2)
            assert signal.step_range(target) == (signal.n_values - 2, signal.n_values)


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
        signal = record.signals[0]
        first = datetime.fromisoformat(next(a.value for a in record.annotations if a.key == "window_start"))
        rebuilt = [
            (first + timedelta(microseconds=offset)).isoformat() for offset in _offsets(signal.time_axis, signal)
        ]
        for window_id in _window_of(record):
            assert rebuilt == rows[str(window_id)]["DATE"]

    # The writer drew the task stream once already, so this pass checks that it streams again.
    task_keys = [_task_key(task) for task in original.iter_tasks()]
    assert len(task_keys) == len(set(task_keys)) == 5 + 7 + 3
    assert sorted(map(_task_key, restored.tasks)) == sorted(task_keys)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("DATE", ["2023-01-03T00:00:00", "2023-01-03T00:30:00", "2023-01-03T01:00:00"], "7day_hourly cadence"),
        ("DATE", ["2023-01-03T01:00:00", "2023-01-03T00:00:00", "2023-01-03T02:00:00"], "7day_hourly cadence"),
        # NumPy parses this; the row model must reject it before it does.
        ("DATE", ["NaT", "2023-01-03T01:00:00", "2023-01-03T02:00:00"], r"(?s)DATE\.0.*should match pattern"),
        ("DATE", ["2023-02-30T00:00:00", "2023-02-30T01:00:00", "2023-02-30T02:00:00"], "DATE: .*2023-02-30"),
        ("temperature", [1.0, 2.0], "temperature has 2 values for 4 timestamps"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("temperature", [1.0, float("nan"), 2.0, 3.0], r"(?s)temperature\.1.*finite number"),
        ("sky_code", [0, 1, 2, 11], r"(?s)sky_code\.3.*less than 11"),
        ("mode", "28day_4hourly", "mode '28day_4hourly' does not match the id's anchor position"),
        # A JSON true is an int to Python, so the label must reject it explicitly.
        ("event_type", True, r"(?s)event_type.*expected an integer, got True"),
        # A pretrain window must publish its label.
        ("event_type", None, "a pretrain window must publish an event_type"),
        # A county must split into a name and a state.
        ("location", "MADERA CALIFORNIA", r"(?s)location.*expected \"COUNTY,STATE\""),
    ],
)
def test_malformed_windows_have_source_context(tmp_path, field, value, message):
    windows = {part: dict(rows) for part, rows in _WINDOWS.items()}
    row = dict(windows["pretrain/train"]["0"])
    if field == "DATE":
        row = {**row, **{name: row[name][: len(value)] for name in _CHANNELS}}
    row[field] = value
    windows["pretrain/train"]["0"] = row
    with pytest.raises(TimeFValidationError, match=message) as error:
        _convert(tmp_path, windows=windows)
    assert "train.json (key 0)" in str(error.value)


def test_forecasting_rows_must_not_carry_an_event_type(tmp_path):
    windows = {part: dict(rows) for part, rows in _WINDOWS.items()}
    windows["forecasting/train"]["60"] = {**windows["forecasting/train"]["60"], "event_type": 3}
    with pytest.raises(TimeFValidationError, match=r"train.json \(key 60\): a forecasting window must not publish"):
        _convert(tmp_path, windows=windows)


def test_copies_must_agree_on_their_coordinates(tmp_path):
    windows = {part: dict(rows) for part, rows in _WINDOWS.items()}
    windows["pretrain/val"]["15"] = {**windows["pretrain/val"]["15"], "latitude": 40.6}
    with pytest.raises(TimeFValidationError, match=r"key 15\): same measurements as window 12, but at \(40.6"):
        _convert(tmp_path, windows=windows)


def test_window_ids_must_be_canonical_and_distinct(tmp_path):
    windows = {part: dict(rows) for part, rows in _WINDOWS.items()}
    windows["pretrain/train"]["012"] = windows["pretrain/train"].pop("12")
    with pytest.raises(TimeFValidationError, match="canonical decimal window id, got '012'"):
        _convert(tmp_path, windows=windows)
    windows = {part: dict(rows) for part, rows in _WINDOWS.items()}
    windows["pretrain/test"]["0"] = windows["pretrain/train"]["0"]
    with pytest.raises(TimeFValidationError, match="window 0 also appears in"):
        _convert(tmp_path, windows=windows)


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        # A classified window without a retrieval row, or with another type's report.
        (
            {"retrieval/train": {"0": (1001, 1), "1": (1001, 1), "30": None}},
            r"key 12\): classified window has no retrieval row",
        ),
        (
            {"retrieval/train": {"0": (1001, 1), "1": (1001, 1), "12": (1012, 6), "30": None}},
            r"key 12\): no event report",
        ),
        # A report on a window the authors don't classify.
        (
            {"retrieval/train": {"0": (1001, 1), "1": (1001, 1), "12": (1012, 8), "30": (1030, 2)}},
            r"key 30\): event report for",
        ),
        # A description without a window.
        (
            {"retrieval/train": {"0": (1001, 1), "1": (1001, 1), "12": (1012, 8), "99": None}},
            "1 retrieval rows have no window",
        ),
    ],
)
def test_event_reports_and_descriptions_must_match_windows(tmp_path, reports, message):
    with pytest.raises(TimeFValidationError, match=message):
        _convert(tmp_path, reports={**_REPORTS, **reports})


_FIXTURE_COUNTS = {
    "pretrain/train": PinnedCounts(5, 3),
    "pretrain/val": PinnedCounts(1, 1),
    "pretrain/test": PinnedCounts(1, 1),
    "forecasting/train": PinnedCounts(2, 1),
    "forecasting/val": PinnedCounts(1, 1),
    "forecasting/test": PinnedCounts(1, 1),
    "retrieval/train": PinnedCounts(4, 3),
    "retrieval/test": PinnedCounts(3, 2),
}


def test_file_counts_and_distinct_windows_are_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "EXPECTED_DISTINCT_WINDOWS", 10)
    TraceWeatherEventsConnector().convert(_write(tmp_path / "ok", expected=_FIXTURE_COUNTS))

    monkeypatch.setattr(archive, "EXPECTED_DISTINCT_WINDOWS", 11)
    with pytest.raises(TimeFValidationError, match="expected 11 distinct measurement windows, found 10"):
        TraceWeatherEventsConnector().convert(_write(tmp_path / "distinct", expected=_FIXTURE_COUNTS))

    wrong = {**_FIXTURE_COUNTS, "forecasting/val": PinnedCounts(1, 0)}
    with pytest.raises(TimeFValidationError, match=r"val.json: expected 1 rows and 0 tasks or events, found 1 and 1"):
        TraceWeatherEventsConnector().convert(_write(tmp_path / "wrong", expected=wrong))


def test_split_file_references_must_be_the_archives_files():
    # A repeated file: one sorted comparison also rejects a missing or unknown one.
    parts = [*SOURCE_FILES, (TraceTaskKind.PRETRAIN, TraceSplit.TRAIN)]
    refs = [TraceFileRef(Path(f"{kind}/{split}"), kind, split) for kind, split in parts]
    with pytest.raises(TimeFValidationError, match="requires each of the split files"):
        TraceWeatherEventsConnector().convert(refs)


def test_download_locates_split_files_in_the_pinned_archive(monkeypatch, tmp_path):
    calls = []

    async def fake_ensure_archive(url, target, *, filename, sha256):
        calls.append((url, target, filename, sha256))
        return tmp_path / "archive"

    monkeypatch.setattr(module, "ensure_archive", fake_ensure_archive)
    with pytest.raises(TimeNetDownloadError, match="TRACE archive is missing"):
        TraceWeatherEventsConnector().download(tmp_path / "cache")

    _write(tmp_path / "archive")
    refs = TraceWeatherEventsConnector().download(tmp_path / "cache")
    assert calls[-1] == (module.SOURCE_URL, tmp_path / "cache", "trace_dataset.zip", module.SOURCE_SHA256)
    assert [
        (ref.path.relative_to(tmp_path / "archive").as_posix(), ref.kind, ref.split, ref.expected) for ref in refs
    ] == [
        (f"dataset/{task}/{split}.{'parquet' if task == 'retrieval' else 'json'}", task, split, expected)
        for (task, split), expected in SOURCE_FILES.items()
    ]


def test_the_loader_selects_tasks_by_split_and_kind(tmp_path):
    trace = TraceDatasetLoader.parse_from(_write(tmp_path))
    every = list(trace.tasks())
    assert [(task.TIMENET_TYPE.__name__, task.window.ref.window_id) for task in every] == [
        _task_key(task) for task in trace.timenet_tasks()
    ]
    for kind in (TraceEventTypeTask, TraceRetrievalTask, TraceForecastTask):
        for split in Split:
            selected = list(trace.tasks_of(kind, split=split))
            assert selected == [task for task in every if isinstance(task, kind) and task.split is split]
