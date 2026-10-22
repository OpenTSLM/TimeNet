from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import AnswerTask, ClassificationTask, Split, ureg
from timenet_connectors.datasets.trace.weather_events import TraceWeatherEventsConnector, connector as module
from timenet_connectors.datasets.trace.weather_events.connector import Inventory, TraceFileRef
from timenet_connectors.sources import json_stream


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
_H, _D = 3_600_000_000, 86_400_000_000
# What each fixture window must become, written out from the fixture files rather than recomputed
# the way the connector computes it: its anchor, its first timestamp in UTC, its time offsets,
# and its axis. Window 12 skips 02:00, so only it keeps explicit offsets.
_WINDOWS = {
    "12": (4, datetime(2023, 1, 3, tzinfo=UTC), [0, _H, 3 * _H, 4 * _H], IrregularAxis),
    "13": (4, datetime(2022, 12, 12, tzinfo=UTC), [0, 4 * _H], RegularAxis),
    "14": (4, datetime(2022, 7, 13, tzinfo=UTC), [0, _D], RegularAxis),
    "15": (5, datetime(2021, 3, 1, tzinfo=UTC), [0, _H, 2 * _H], RegularAxis),
    "42": (14, datetime(2021, 12, 20, tzinfo=UTC), [0, _H, 2 * _H], RegularAxis),
}
# Each fixture window's location and first and last timestamps. Window 42 is a forecasting window,
# which publishes its location as the list ["Doña Ana County", "New Mexico"].
_CONTEXT = {
    "12": ("MADERA,CALIFORNIA", "2023-01-03T00:00:00", "2023-01-03T04:00:00"),
    "13": ("MADERA,CALIFORNIA", "2022-12-12T00:00:00", "2022-12-12T04:00:00"),
    "14": ("MADERA,CALIFORNIA", "2022-07-13T00:00:00", "2022-07-14T00:00:00"),
    "15": ("DOÑA ANA,NEW MEXICO", "2021-03-01T00:00:00", "2021-03-01T02:00:00"),
    "42": ("Doña Ana County, New Mexico", "2021-12-20T00:00:00", "2021-12-20T02:00:00"),
}
# The labels of each fixture window's published sky codes. Window 12 publishes 0, 1, 8, and 9, the
# ends of the okta range and the obscured code; the others publish 2, 3, 4, ... in order.
_SKY_LABELS = {
    "12": ["0 oktas", "1 okta", "8 oktas", "sky obscured"],
    "13": ["2 oktas", "3 oktas"],
    "14": ["2 oktas", "3 oktas"],
    "15": ["2 oktas", "3 oktas", "4 oktas"],
    "42": ["2 oktas", "3 oktas", "4 oktas"],
}
_EVENT_TYPES = [
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
_CAPTION_PROMPT = (
    "You are an expert in time series analysis. Please generate a detailed caption for this "
    "time-series, describing it as accurately as possible."
)


@pytest.fixture(autouse=True)
def _short_windows_classify(monkeypatch):
    """Let the few-value fixture windows stand in for the archive's 100-value minimum."""
    monkeypatch.setattr(module, "MIN_CLASSIFICATION_VALUES", 3)


def _ref(path: Path, part: str, expected: Inventory | None = None) -> TraceFileRef:
    task, split = part.split("/")
    return TraceFileRef(path, task, split, expected)


def _refs(root: Path = _FIXTURES) -> list[TraceFileRef]:
    return [_ref(root / f"{part}.json", part) for part in _PARTS]


def _rows() -> dict[str, tuple[str, dict]]:
    return {
        key: (part, row)
        for part in _PARTS
        for key, row in json.loads((_FIXTURES / f"{part}.json").read_text(encoding="utf-8")).items()
    }


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


def _context(key: str) -> list[tuple[str, str, str]]:
    """Return the id, key, and value of each context annotation expected on a fixture window."""
    location, start, end = _CONTEXT[key]
    return [
        (f"trace-{key}-location", "location", location),
        (f"trace-{key}-window-start", "window_start", start),
        (f"trace-{key}-window-end", "window_end", end),
    ]


def _expected_values(key: str, name: str, row: dict) -> np.ndarray:
    if name == "sky_code":
        return np.asarray(_SKY_LABELS[key])
    return np.asarray(row[name], dtype=np.float32)


def test_every_window_becomes_a_record_with_its_signals(monkeypatch, tmp_path):
    # The lazy loaders seek by byte offset, including across CRLF and UTF-8 text.
    for ref in _refs():
        path = tmp_path / ref.task / f"{ref.split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ref.path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 17)
    dataset = TraceWeatherEventsConnector().convert(_refs(tmp_path))

    rows = _rows()
    records = {record.id: record for record in dataset.records}
    assert records.keys() == {f"trace-{key}" for key in _WINDOWS}
    for key, (anchor, start, offsets, axis_type) in _WINDOWS.items():
        part, row = rows[key]
        record = records[f"trace-{key}"]
        assert record.metadata == {
            "source_id": int(key),
            "anchor": anchor,
            "source_file": f"dataset/{part}.json",
            "station_id": row["station_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "mode": row["mode"],
        }
        # The location and dates are caption context, so they are annotations, not metadata.
        assert [(a.id, a.key, a.value) for a in record.annotations] == _context(key)
        assert record.start_time.timestamp == int(start.timestamp()) * 1_000_000
        (source,) = record.sources
        assert source.name == row["station_id"]
        assert source.metadata == {"channel_order": list(_UNITS)}
        assert {signal.name for signal in record.signals} == set(_UNITS)
        # The seven channels share one axis.
        (axis,) = {id(signal.time_axis): signal.time_axis for signal in record.signals}.values()
        assert type(axis) is axis_type
        for signal in record.signals:
            assert signal.spec.unit_value == _UNITS[signal.name]
            if isinstance(axis, RegularAxis):
                assert [axis.time_offset_us(index) for index in range(len(offsets))] == offsets
            else:
                assert signal.time_offsets_us().tolist() == offsets
            np.testing.assert_array_equal(signal.to_numpy(), _expected_values(key, signal.name, row))

    annotations = {annotation.id: annotation.value for annotation in dataset.annotations}
    assert annotations == {"trace-no_event_label": -100, "trace-noaa-event-types": _EVENT_TYPES}


def test_classified_event_windows_get_an_event_type_task_with_their_report(tmp_path):
    dataset = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path))
    records = {record.id: record for record in dataset.records}
    tasks = {task.id: task for task in dataset.iter_tasks() if type(task) is ClassificationTask}

    # Window 13 is a non-event, 14 an event window too short to classify, and 42 a forecasting window.
    assert tasks.keys() == {"trace-12-event-type", "trace-15-event-type"}
    for key, target, index, split, event_id in (
        ("12", "Debris Flow", 1, Split.TRAIN, 1065296),
        ("15", "Flash Flood", 2, Split.VALIDATION, 1065300),
    ):
        task = tasks[f"trace-{key}-event-type"]
        assert task.inputs == (records[f"trace-{key}"],)
        assert task.targets == (target,)
        assert task.prompt is None
        # The authors' generate_er text: it names the target, so it is never an input.
        assert task.rationale == f"The weather event is {target}. {_NARRATIVES[int(key)]}"
        assert task.input_annotations == ()
        assert task.split is split
        assert task.target_schema == "trace-noaa-event-types"
        assert task.metadata == {"event_type_index": index, "event_id": event_id}


def test_every_retrieval_row_becomes_a_caption_task(tmp_path):
    dataset = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path))
    records = {record.id: record for record in dataset.records}
    tasks = {task.id: task for task in dataset.iter_tasks() if type(task) is AnswerTask}

    # Event and non-event rows alike, including the forecasting window 42.
    assert tasks.keys() == {"trace-12-caption", "trace-13-caption", "trace-15-caption", "trace-42-caption"}
    for key, split in (("12", Split.TRAIN), ("13", Split.TRAIN), ("15", Split.TEST), ("42", Split.TEST)):
        task = tasks[f"trace-{key}-caption"]
        assert task.inputs == (records[f"trace-{key}"],)
        assert task.prompt == _CAPTION_PROMPT
        assert task.split is split
        assert task.targets == (_caption(key),)
        assert task.rationale is None
        # The caption states the window's place and dates, which the series alone does not carry.
        assert [(a.id, a.key, a.value) for a in task.input_annotations] == _context(key)


def test_without_retrieval_files_there_are_no_rationales_or_captions():
    tasks = list(TraceWeatherEventsConnector().convert(_refs()).iter_tasks())
    assert sorted(task.id for task in tasks) == ["trace-12-event-type", "trace-15-event-type"]
    assert all(task.rationale is None and "event_id" not in task.metadata for task in tasks)


@pytest.mark.parametrize(
    ("description", "reports", "message"),
    [
        (lambda window_id: {**_description(window_id), "wind_v": " "}, _REPORTS, r"(?s)wind_v.*should match pattern"),
        (
            _description,
            {**_REPORTS, "retrieval/test": [*_REPORTS["retrieval/test"], (99, None)]},
            "1 retrieval rows have no window, such as 99",
        ),
    ],
)
def test_captions_must_be_complete_and_belong_to_a_window(tmp_path, description, reports, message):
    refs = _refs() + _write_reports(tmp_path, reports, description)
    with pytest.raises(TimeFValidationError, match=message):
        TraceWeatherEventsConnector().convert(refs)


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
        (
            {"retrieval/train": [(12, {"event_id": 1, "event_type": 1, "narrative": " "})]},
            r"(?s)narrative.*should match pattern",
        ),
        (
            {"retrieval/train": [(12, {"event_id": 1, "event_type": 9, "narrative": "x"})]},
            r"(?s)event_type.*less than 9",
        ),
        ({"retrieval/train": _REPORTS["retrieval/train"] * 2}, "distinct integer window id, got 12"),
    ],
)
def test_event_reports_must_match_classified_windows(tmp_path, reports, message):
    refs = _refs() + _write_reports(tmp_path, reports)
    with pytest.raises(TimeFValidationError, match=message):
        TraceWeatherEventsConnector().convert(refs)


@pytest.mark.parametrize(
    ("expected", "passes"),
    # The retrieval/train fixture holds two rows, one of them with an event report.
    [(Inventory(2, 1), True), (Inventory(3, 1), False), (Inventory(2, 0), False)],
)
def test_retrieval_file_inventory_is_pinned(tmp_path, expected, passes):
    refs = [
        TraceFileRef(ref.path, ref.task, ref.split, expected if ref.split == "train" else None)
        for ref in _write_reports(tmp_path)
    ]
    if passes:
        tasks = TraceWeatherEventsConnector().convert(_refs() + refs).iter_tasks()
        assert sorted(task.id for task in tasks) == [
            "trace-12-caption",
            "trace-12-event-type",
            "trace-13-caption",
            "trace-15-caption",
            "trace-15-event-type",
            "trace-42-caption",
        ]
    else:
        with pytest.raises(TimeFValidationError, match=r"expected \d+ rows and \d+ events, found 2 and 1"):
            TraceWeatherEventsConnector().convert(_refs() + refs)


def test_storage_round_trip_keeps_values_axes_and_tasks(tmp_path):
    original = TraceWeatherEventsConnector().convert(_refs() + _write_reports(tmp_path / "reports"))
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("trace/weather-events", "1.0.0")) as reader:
        restored = reader.read()

    # Check the stored values against the fixtures themselves, not against the converted dataset.
    rows = _rows()
    records = {record.id: record for record in restored.records}
    assert records.keys() == {f"trace-{key}" for key in _WINDOWS}
    for key, (_, _, _, axis_type) in _WINDOWS.items():
        for signal in records[f"trace-{key}"].signals:
            assert type(signal.time_axis) is axis_type
            np.testing.assert_array_equal(signal.to_numpy(), _expected_values(key, signal.name, rows[key][1]))
    tasks = {task.id: task for task in restored.tasks}
    assert tasks["trace-12-event-type"].targets == ("Debris Flow",)
    assert tasks["trace-12-event-type"].rationale == f"The weather event is Debris Flow. {_NARRATIVES[12]}"
    assert tasks["trace-12-event-type"].metadata == {"event_type_index": 1, "event_id": 1065296}
    assert tasks["trace-15-event-type"].split is Split.VALIDATION
    assert tasks["trace-42-caption"].prompt == _CAPTION_PROMPT
    assert tasks["trace-42-caption"].split is Split.TEST
    assert tasks["trace-42-caption"].targets == (_caption("42"),)


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


def test_download_reports_a_missing_split_file(monkeypatch, tmp_path):
    async def ensure_archive(url, target, *, filename, sha256):
        (target / "dataset" / "pretrain").mkdir(parents=True)
        (target / "dataset" / "pretrain" / "train.json").touch()
        return target

    monkeypatch.setattr(module, "ensure_archive", ensure_archive)
    with pytest.raises(TimeNetDownloadError, match=r"missing \[.*pretrain/val\.json"):
        TraceWeatherEventsConnector().download(tmp_path)


@pytest.mark.parametrize(
    ("windows", "classified", "passes"),
    [
        # pretrain/train holds three fixture windows, one of which becomes a task.
        (3, 1, True),
        (2, 1, False),
        (3, 0, False),
    ],
)
def test_window_file_inventory_is_pinned(windows, classified, passes):
    expected = Inventory(windows, classified)
    refs = [
        TraceFileRef(ref.path, ref.task, ref.split, expected) if (ref.task, ref.split) == ("pretrain", "train") else ref
        for ref in _refs()
    ]
    if passes:
        assert sorted(record.id for record in TraceWeatherEventsConnector().convert(refs).records) == [
            f"trace-{key}" for key in sorted(_WINDOWS)
        ]
    else:
        with pytest.raises(TimeFValidationError, match=r"expected \d+ rows and \d+ events, found 3 and 1"):
            TraceWeatherEventsConnector().convert(refs)


@pytest.mark.parametrize("parts", [[], ["pretrain/train", "pretrain/train"], ["pretrain/holdout"], ["retrieval/train"]])
def test_split_file_references_must_be_known_and_distinct(parts):
    with pytest.raises(TimeFValidationError, match="distinct split files"):
        TraceWeatherEventsConnector().convert([_ref(_FIXTURES / "pretrain/train.json", part) for part in parts])


_HOURS = ("2023-01-03T00:00:00", "2023-01-03T01:00:00", "2023-01-03T02:00:00", "2023-01-03T03:00:00")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("DATE", ["2023-01-03T00:00:00", "2023-01-03T00:30:00", *_HOURS[1:3]], "7day_hourly cadence"),
        ("DATE", [_HOURS[1], _HOURS[0], *_HOURS[2:]], "7day_hourly cadence"),
        # NumPy parses each of these; the row model must reject them before it does.
        ("DATE", ["NaT", *_HOURS[1:]], r"(?s)DATE\.0.*should match pattern"),
        ("DATE", ["now", *_HOURS[1:]], r"(?s)DATE\.0.*should match pattern"),
        ("DATE", [*_HOURS[:3], "2023-01-03T03:00:00+02:00"], r"(?s)DATE\.3.*should match pattern"),
        ("DATE", [*_HOURS[:3], "2023-01-03T03:00:00Z"], r"(?s)DATE\.3.*should match pattern"),
        ("DATE", ["2023-01-03", *_HOURS[1:]], r"(?s)DATE\.0.*should match pattern"),
        ("DATE", ["2023-02-30T00:00:00", *_HOURS[1:]], "DATE: .*2023-02-30"),
        ("temperature", [1.0, 2.0, 3.0], "temperature has 3 values for 4 timestamps"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("temperature", [1.0, float("nan"), 2.0, 3.0], r"(?s)temperature\.1.*finite number"),
        ("wind_u", [True, 1.0, 2.0, 3.0], r"(?s)wind_u\.0.*valid number"),
        ("sky_code", [1.5, 2, 3, 4], r"(?s)sky_code\.0.*valid integer"),
        ("sky_code", [0, 1, 2, 11], r"(?s)sky_code\.3.*less than 11"),
        ("mode", "28day_4hourly", "mode '28day_4hourly' does not match the id's anchor position"),
        ("event_type", 9, r"(?s)event_type.*less than 9"),
        ("location", [], r"(?s)location.*at least 1 item"),
        ("event_type", None, r"(?s)event_type.*Field required"),
    ],
)
def test_malformed_windows_have_source_context(tmp_path, field, value, message):
    data = json.loads((_FIXTURES / "pretrain/train.json").read_text(encoding="utf-8"))
    if field == "event_type" and value is None:
        del data["12"]["event_type"]
    else:
        data["12"][field] = value
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TimeFValidationError, match=message) as error:
        TraceWeatherEventsConnector().convert([_ref(path, "pretrain/train")])
    assert f"{path} (key 12)" in str(error.value)


def test_single_value_window_is_stored_without_a_task(tmp_path):
    # The source holds windows with a single value, which have no step to infer a cadence from.
    row = json.loads((_FIXTURES / "pretrain/train.json").read_text(encoding="utf-8"))["12"]
    single = {**row, "DATE": ["2023-01-03T00:00:00"], **{name: row[name][:1] for name in _CHANNELS}}
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"12": single}))
    dataset = TraceWeatherEventsConnector().convert([_ref(path, "pretrain/train")])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset)
    with TimeFReader(registry.open_version("trace/weather-events", "1.0.0")) as reader:
        restored = reader.read()

    (record,) = restored.records
    assert record.start_time.timestamp == int(datetime(2023, 1, 3, tzinfo=UTC).timestamp()) * 1_000_000
    signals = {signal.name: signal for signal in record.signals}
    assert signals["temperature"].to_numpy().tolist() == [np.float32(row["temperature"][0])]
    assert signals["sky_code"].to_numpy().tolist() == ["0 oktas"]
    assert list(restored.tasks) == []


def test_window_ids_must_be_canonical_decimals(tmp_path):
    # A retrieval row is keyed by its integer id, so "012" could never meet its row.
    row = json.loads((_FIXTURES / "pretrain/train.json").read_text(encoding="utf-8"))["12"]
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"012": row}))
    with pytest.raises(TimeFValidationError, match="expected a canonical decimal window id, got '012'"):
        TraceWeatherEventsConnector().convert([_ref(path, "pretrain/train")])


def test_forecasting_windows_must_not_carry_an_event_label(tmp_path):
    data = json.loads((_FIXTURES / "forecasting/train.json").read_text(encoding="utf-8"))
    data["42"]["event_type"] = 1
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TimeFValidationError, match=r"(?s)key 42.*event_type.*Extra inputs are not permitted"):
        TraceWeatherEventsConnector().convert([_ref(path, "forecasting/train")])
