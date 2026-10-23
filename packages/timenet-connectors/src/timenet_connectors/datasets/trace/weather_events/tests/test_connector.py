from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, ForecastingTask, Split, TimeInterval, ureg
from timenet_connectors.datasets.trace.weather_events import TraceWeatherEventsConnector, connector as module
from timenet_connectors.datasets.trace.weather_events.connector import (
    SKY_COVER,
    EventType,
    Inventory,
    TraceFileRef,
)


_FIXTURES = Path(__file__).parent / "fixtures" / "dataset"
# convert accepts any subset of the window files, with both retrieval files or neither; these three
# hold the fixture windows.
_PARTS = ("pretrain/train", "pretrain/val", "forecasting/train")
_UNITS = {
    "temperature": ureg.degC,
    "precipitation": ureg.mm,
    "relative_humidity": ureg.percent,
    "visibility": ureg.km,
    "wind_u": ureg.m / ureg.s,
    "wind_v": ureg.m / ureg.s,
    "sky_code": ureg.dimensionless,
}
# Window 12 skips 02:00, so only it keeps explicit offsets; the others are regular.
_IRREGULAR = {"12"}


@pytest.fixture(autouse=True)
def _short_windows_get_tasks(monkeypatch):
    """Let the few-value fixture windows stand in for the archive's 100-value minimum and horizon."""
    monkeypatch.setattr(module, "MIN_TASK_VALUES", 3)
    monkeypatch.setattr(module, "FORECAST_HORIZON", 2)


def _ref(path: Path, part: str, expected: Inventory | None = None) -> TraceFileRef:
    task, split = part.split("/")
    return TraceFileRef(path, task, split, expected)


def _refs() -> list[TraceFileRef]:
    return [_ref(_FIXTURES / f"{part}.json", part) for part in _PARTS]


def _windows(part: str) -> dict[str, dict]:
    return json.loads((_FIXTURES / f"{part}.json").read_text(encoding="utf-8"))


def _write_windows(tmp_path: Path, part: str, windows: dict) -> TraceFileRef:
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(windows))
    return _ref(path, part)


def _rows() -> dict[str, tuple[str, dict]]:
    return {key: (part, row) for part in _PARTS for key, row in _windows(part).items()}


_NARRATIVES = {12: "A strong low pressure system moved through central California.", 15: "Rain fell on burn scars."}
# The archive's Stage 2 files: an event report for each classified window, and null for the others.
_REPORTS = {
    "retrieval/train": [(12, {"event_id": 1065296, "event_type": 1, "narrative": _NARRATIVES[12]}), (13, None)],
    "retrieval/test": [(15, {"event_id": 1065300, "event_type": 2, "narrative": _NARRATIVES[15]}), (42, None)],
}
_CHANNELS = ("temperature", "precipitation", "relative_humidity", "visibility", "wind_u", "wind_v", "sky_code")
_DESCRIPTION = pa.struct([(name, pa.string()) for name in ("DATE", "labels", "location", *_CHANNELS)])
_EVENT = pa.struct([("event_id", pa.int64()), ("event_type", pa.int64()), ("narrative", pa.string())])


def _description(window_id: int) -> dict[str, str]:
    return {field: f"{field} of {window_id}" for field in _DESCRIPTION.names}


def _write_reports(root: Path, reports: dict[str, list] = _REPORTS, description=_description) -> list[TraceFileRef]:
    """Write the Stage 2 parquet files with the archive's schema, which the connector reads in part."""
    refs = []
    for part, rows in reports.items():
        path = root / f"{part}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "id": pa.array([window_id for window_id, _ in rows], type=pa.int64()),
                "description": pa.array([description(window_id) for window_id, _ in rows], type=_DESCRIPTION),
                "timeseries": pa.array([b"npy" for _ in rows], type=pa.binary()),
                "events": pa.array([event for _, event in rows], type=_EVENT),
            }
        )
        pq.write_table(table, path)
        refs.append(_ref(path, part))
    return refs


def _caption(key: str) -> str:
    """Return the caption target for a fixture row: the authors' generate_dsp text, spacing included."""
    channels = " \n ".join(f"{name} of {key}" for name in _CHANNELS)
    return (
        f"Weather time series location: location of {key} Time range: DATE of {key} "
        f"The weather is labels of {key}. {channels}"
    )


def _context(key: str, row: dict) -> list[tuple[str, str, str]]:
    """Return the id, key, and value of each context annotation expected on a fixture window."""
    # A forecasting window publishes its location as a list, such as ["Doña Ana County", "New Mexico"].
    location = row["location"] if isinstance(row["location"], str) else ", ".join(row["location"])
    return [
        (f"trace-{key}-location", "location", location),
        (f"trace-{key}-window-start", "window_start", row["DATE"][0]),
        (f"trace-{key}-window-end", "window_end", row["DATE"][-1]),
    ]


def _timing(row: dict) -> tuple[datetime, list[int]]:
    """Return a fixture window's first timestamp in UTC and its offsets, parsed without NumPy."""
    dates = [datetime.fromisoformat(date).replace(tzinfo=UTC) for date in row["DATE"]]
    return dates[0], [(date - dates[0]) // timedelta(microseconds=1) for date in dates]


def _expected_values(name: str, row: dict) -> np.ndarray:
    if name == "sky_code":
        return np.asarray([SKY_COVER[code] for code in row["sky_code"]])
    return np.asarray(row[name], dtype=np.float32)


def test_every_window_becomes_a_record_with_its_signals():
    dataset = TraceWeatherEventsConnector().convert(_refs())

    rows = _rows()
    records = {record.id: record for record in dataset.records}
    assert records.keys() == {f"trace-{key}" for key in rows}
    for key, (part, row) in rows.items():
        start, offsets = _timing(row)
        axis_type = IrregularAxis if key in _IRREGULAR else RegularAxis
        record = records[f"trace-{key}"]
        assert record.metadata == {
            "source_id": int(key),
            "anchor": int(key) // 3,
            "source_file": f"dataset/{part}.json",
            "station_id": row["station_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "mode": row["mode"],
        } | ({"event_type": row["event_type"]} if part.startswith("pretrain/") else {})
        # The location and dates are caption context, so they are annotations, not metadata.
        assert [(a.id, a.key, a.value) for a in record.annotations] == _context(key, row)
        assert record.start_time.timestamp == int(start.timestamp()) * 1_000_000
        (source,) = record.sources
        assert source.name == row["station_id"]
        assert source.metadata == {"channel_order": list(_UNITS)}
        assert {signal.name for signal in record.signals} == set(_UNITS)
        # The seven channels share one axis.
        (axis,) = {id(signal.time_axis): signal.time_axis for signal in record.signals}.values()
        assert type(axis) is axis_type
        if isinstance(axis, RegularAxis):
            assert [axis.time_offset_us(index) for index in range(len(offsets))] == offsets
        for signal in record.signals:
            assert signal.spec.unit_value == _UNITS[signal.name]
            if not isinstance(axis, RegularAxis):
                assert signal.time_offsets_us().tolist() == offsets
            np.testing.assert_array_equal(signal.to_numpy(), _expected_values(signal.name, row))

    annotations = {annotation.id: annotation.value for annotation in dataset.annotations}
    assert annotations == {"trace-no-event-label": -100, "trace-noaa-event-types": [event.label for event in EventType]}


def test_classified_event_windows_get_an_event_type_task_with_their_report(tmp_path):
    dataset = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path))
    records = {record.id: record for record in dataset.records}
    tasks = {task.id: task for task in dataset.iter_tasks() if type(task) is ClassificationTask}

    # Window 13 is a non-event, 14 an event window too short to classify, and 42 a forecasting window.
    assert tasks.keys() == {"trace-12-event-type", "trace-15-event-type"}
    # Windows without a task still keep their published label.
    assert records["trace-13"].metadata["event_type"] == -100
    assert records["trace-14"].metadata["event_type"] == 8
    for key, target, index, split, event_id in (
        (12, "Debris Flow", 1, Split.TRAIN, 1065296),
        (15, "Flash Flood", 2, Split.VALIDATION, 1065300),
    ):
        task = tasks[f"trace-{key}-event-type"]
        assert task.inputs == (records[f"trace-{key}"],)
        assert task.targets == (target,)
        assert task.prompt is None
        # The authors' generate_er text: it names the target, so it is never an input.
        assert task.rationale == f"The weather event is {target}. {_NARRATIVES[key]}"
        assert task.input_annotations == ()
        assert task.split is split
        assert task.target_schema == "trace-noaa-event-types"
        assert task.metadata == {"event_type_index": index, "event_id": event_id}


def test_every_retrieval_row_becomes_a_caption_task(tmp_path):
    dataset = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path))
    records = {record.id: record for record in dataset.records}
    tasks = {task.id: task for task in dataset.iter_tasks() if type(task) is AnswerTask}
    rows = _rows()

    # Event and non-event rows alike, including the forecasting window 42.
    assert tasks.keys() == {"trace-12-caption", "trace-13-caption", "trace-15-caption", "trace-42-caption"}
    for key, split in (("12", Split.TRAIN), ("13", Split.TRAIN), ("15", Split.TEST), ("42", Split.TEST)):
        task = tasks[f"trace-{key}-caption"]
        assert task.inputs == (records[f"trace-{key}"],)
        assert task.prompt is None
        assert task.split is split
        assert task.targets == (_caption(key),)
        assert task.rationale is None
        # The caption states the window's place and dates, which the series alone does not carry.
        assert [(a.id, a.key, a.value) for a in task.input_annotations] == _context(key, rows[key][1])
    # One caption written out in full, as the authors' generate_dsp composes it, so the _caption
    # helper above is pinned to the published format rather than to the connector's template.
    assert tasks["trace-42-caption"].targets == (
        "Weather time series location: location of 42 Time range: DATE of 42 The weather is labels of 42. "
        "temperature of 42 \n precipitation of 42 \n relative_humidity of 42 \n visibility of 42 \n "
        "wind_u of 42 \n wind_v of 42 \n sky_code of 42",
    )


_HOUR_US = 3_600_000_000


@pytest.mark.parametrize(
    ("dates", "target"),
    [
        # Window 42 publishes three hourly values; the horizon of two leaves one as input.
        (None, (_HOUR_US, 2 * _HOUR_US + 1)),
        # The horizon counts values, so across a gap it spans more time.
        (["2021-12-20T00:00:00", "2021-12-20T01:00:00", "2021-12-20T03:00:00"], (_HOUR_US, 3 * _HOUR_US + 1)),
        # Two values fall under the minimum of three.
        (["2021-12-20T00:00:00", "2021-12-20T01:00:00"], None),
    ],
)
def test_forecasting_windows_forecast_their_last_values(tmp_path, dates, target):
    data = _windows("forecasting/train")
    if dates is not None:
        data["42"] |= {"DATE": dates, **{name: data["42"][name][: len(dates)] for name in _CHANNELS}}
    dataset = TraceWeatherEventsConnector().convert([_write_windows(tmp_path, "forecasting/train", data)])
    tasks = list(dataset.iter_tasks())
    if target is None:
        assert tasks == []
        return
    (task,) = tasks
    (record,) = dataset.records
    assert type(task) is ForecastingTask
    assert task.id == "trace-42-forecast"
    assert task.inputs == (record,)
    assert task.scope == TimeInterval(start_us=0, end_us=target[0])
    forecast = TimeInterval(start_us=target[0], end_us=target[1])
    assert task.targets == (forecast,)
    assert task.prompt is None
    assert task.split is Split.TRAIN
    # The target covers exactly the last values of every channel.
    for signal in record.signals:
        assert signal.step_range(forecast) == (signal.n_values - 2, signal.n_values)


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        ({"retrieval/train": [(13, None)]}, r"key 12\): no event report with event_type 1"),
        (
            {"retrieval/train": [(12, {"event_id": 1, "event_type": 8, "narrative": "Wind."})]},
            r"key 12\): no event report with event_type 1",
        ),
        # Window 14 is an event window, but too short to classify.
        (
            {**_REPORTS, "retrieval/test": [*_REPORTS["retrieval/test"], (14, _REPORTS["retrieval/train"][0][1])]},
            r"key 14\): event report for a window without an event-type task",
        ),
        ({"retrieval/train": _REPORTS["retrieval/train"] * 2}, "distinct integer window id, got 12"),
        ({"retrieval/test": [*_REPORTS["retrieval/test"], (99, None)]}, "1 retrieval rows have no window, such as 99"),
    ],
)
def test_event_reports_and_captions_must_match_windows(tmp_path, reports, message):
    refs = _refs() + _write_reports(tmp_path, {**_REPORTS, **reports})
    with pytest.raises(TimeFValidationError, match=message):
        TraceWeatherEventsConnector().convert(refs)


@pytest.mark.parametrize(
    ("reports", "ids"),
    [
        (
            True,
            [
                "trace-12-event-type",
                "trace-12-caption",
                "trace-13-caption",
                "trace-15-event-type",
                "trace-15-caption",
                "trace-42-caption",
                "trace-42-forecast",
            ],
        ),
        # Without the retrieval files there are no captions, and no rationales.
        (False, ["trace-12-event-type", "trace-15-event-type", "trace-42-forecast"]),
    ],
)
def test_tasks_stream_on_every_pass(tmp_path, reports, ids):
    dataset = TraceWeatherEventsConnector().convert(_refs() + (_write_reports(tmp_path) if reports else []))
    # The writer iterates more than once, so each pass must reread the files rather than run dry.
    first, second = (list(dataset.iter_tasks()) for _ in range(2))
    assert [task.id for task in first] == [task.id for task in second] == ids
    if not reports:
        assert all(task.rationale is None and "event_id" not in task.metadata for task in first)


def test_storage_round_trip_keeps_values_axes_and_tasks(tmp_path):
    original = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path / "reports"))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("trace/weather-events", "1.0.0")) as reader:
        restored = reader.read()

    # The other tests check the converted dataset against the fixtures; this one checks that storage keeps it.
    converted = {record.id: record for record in original.records}
    assert {record.id for record in restored.records} == converted.keys()
    for record in restored.records:
        signals = {signal.name: signal for signal in converted[record.id].signals}
        for signal in record.signals:
            assert type(signal.time_axis) is type(signals[signal.name].time_axis)
            np.testing.assert_array_equal(signal.to_numpy(), signals[signal.name].to_numpy())

    def fields(task):
        return task.id, task.targets, task.scope, task.prompt, task.rationale, task.split, task.metadata

    assert sorted(map(fields, restored.tasks)) == sorted(map(fields, original.iter_tasks()))


def test_download_locates_split_files_in_the_pinned_archive(monkeypatch, tmp_path):
    calls = []

    async def ensure_archive(url, target, *, filename, sha256):
        calls.append((url, target, filename, sha256))
        for task, split, suffix in (
            *((task, split, "json") for task in ("pretrain", "forecasting") for split in ("train", "val", "test")),
            ("retrieval", "train", "parquet"),
            ("retrieval", "test", "parquet"),
        ):
            (target / "dataset" / task).mkdir(parents=True, exist_ok=True)
            (target / "dataset" / task / f"{split}.{suffix}").touch()
        return target

    monkeypatch.setattr(module, "ensure_archive", ensure_archive)
    refs = TraceWeatherEventsConnector().download(tmp_path)

    assert calls == [
        (
            "https://drive.usercontent.google.com/download?id=1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY&export=download&confirm=t",
            tmp_path,
            "trace_dataset.zip",
            "a5d24d75a0b9f0f846a185caa46c4f77e0ef98272c12808df76fac25a181be58",
        )
    ]
    # Row counts are paper Table 8's, classified counts the archive's classification/*/labels.npy,
    # and the retrieval counts the archive's retrieval parquet rows and event reports.
    assert [(ref.path.relative_to(tmp_path).as_posix(), ref.task, ref.split, ref.expected) for ref in refs] == [
        ("dataset/pretrain/train.json", "pretrain", "train", (45339, 10110)),
        ("dataset/pretrain/val.json", "pretrain", "val", (12953, 2884)),
        ("dataset/pretrain/test.json", "pretrain", "test", (6484, 1452)),
        ("dataset/forecasting/train.json", "forecasting", "train", (6690, 0)),
        ("dataset/forecasting/val.json", "forecasting", "val", (1914, 0)),
        ("dataset/forecasting/test.json", "forecasting", "test", (957, 0)),
        ("dataset/retrieval/train.parquet", "retrieval", "train", (42180, 13787)),
        ("dataset/retrieval/test.parquet", "retrieval", "test", (2006, 659)),
    ]


@pytest.mark.parametrize(
    ("pins", "found"),
    [
        # pretrain/train holds three windows, one classified; retrieval/train two rows, one with a report.
        ({"pretrain/train": Inventory(3, 1), "retrieval/train": Inventory(2, 1)}, None),
        ({"pretrain/train": Inventory(2, 1)}, "found 3 and 1"),
        ({"retrieval/train": Inventory(2, 0)}, "found 2 and 1"),
    ],
)
def test_file_inventories_are_pinned(tmp_path, pins, found):
    refs = [
        TraceFileRef(ref.path, ref.task, ref.split, pins.get(f"{ref.task}/{ref.split}"))
        for ref in _refs() + _write_reports(tmp_path)
    ]
    if found is None:
        TraceWeatherEventsConnector().convert(refs)
    else:
        with pytest.raises(TimeFValidationError, match=rf"expected \d+ rows and \d+ events, {found}"):
            TraceWeatherEventsConnector().convert(refs)


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        (["pretrain/train", "pretrain/train"], "distinct split files"),
        (["pretrain/holdout"], "distinct split files"),
        (["retrieval/train"], "including a window file"),
        (["pretrain/train", "retrieval/train"], "both retrieval files or neither"),
    ],
)
def test_split_file_references_must_be_valid(parts, message):
    with pytest.raises(TimeFValidationError, match=message):
        TraceWeatherEventsConnector().convert([_ref(_FIXTURES / "pretrain/train.json", part) for part in parts])


_HOURS = ("2023-01-03T00:00:00", "2023-01-03T01:00:00", "2023-01-03T02:00:00", "2023-01-03T03:00:00")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("DATE", ["2023-01-03T00:00:00", "2023-01-03T00:30:00", *_HOURS[1:3]], "7day_hourly cadence"),
        ("DATE", [_HOURS[1], _HOURS[0], *_HOURS[2:]], "7day_hourly cadence"),
        # NumPy parses each of these; the row model must reject them before it does.
        ("DATE", ["NaT", *_HOURS[1:]], r"(?s)DATE\.0.*should match pattern"),
        ("DATE", [*_HOURS[:3], "2023-01-03T03:00:00Z"], r"(?s)DATE\.3.*should match pattern"),
        ("DATE", ["2023-02-30T00:00:00", *_HOURS[1:]], "DATE: .*2023-02-30"),
        ("temperature", [1.0, 2.0, 3.0], "temperature has 3 values for 4 timestamps"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("temperature", [1.0, float("nan"), 2.0, 3.0], r"(?s)temperature\.1.*finite number"),
        ("sky_code", [0, 1, 2, 11], r"(?s)sky_code\.3.*less than 11"),
        ("mode", "28day_4hourly", "mode '28day_4hourly' does not match the id's anchor position"),
        ("event_type", 9, r"(?s)event_type.*less than 9"),
    ],
)
def test_malformed_windows_have_source_context(tmp_path, field, value, message):
    data = _windows("pretrain/train")
    data["12"][field] = value
    ref = _write_windows(tmp_path, "pretrain/train", data)
    with pytest.raises(TimeFValidationError, match=message) as error:
        TraceWeatherEventsConnector().convert([ref])
    assert f"{ref.path} (key 12)" in str(error.value)


def test_single_value_window_is_stored_without_a_task(tmp_path):
    # The source holds windows with a single value, which have no step to infer a cadence from.
    row = _windows("pretrain/train")["12"]
    single = {**row, "DATE": ["2023-01-03T00:00:00"], **{name: row[name][:1] for name in _CHANNELS}}
    dataset = TraceWeatherEventsConnector().convert([_write_windows(tmp_path, "pretrain/train", {"12": single})])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset)
    with TimeFReader(registry.open_version("trace/weather-events", "1.0.0")) as reader:
        restored = reader.read()

    (record,) = restored.records
    signals = {signal.name: signal for signal in record.signals}
    assert signals["temperature"].to_numpy().tolist() == [np.float32(row["temperature"][0])]
    assert signals["sky_code"].to_numpy().tolist() == ["0 oktas"]
    assert list(restored.tasks) == []


def test_window_ids_must_be_canonical_decimals(tmp_path):
    # A retrieval row is keyed by its integer id, so "012" could never meet its row.
    ref = _write_windows(tmp_path, "pretrain/train", {"012": _windows("pretrain/train")["12"]})
    with pytest.raises(TimeFValidationError, match="expected a canonical decimal window id, got '012'"):
        TraceWeatherEventsConnector().convert([ref])
