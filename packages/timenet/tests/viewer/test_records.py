"""Record API isolation, bounded lazy reads, projections, and cleanup."""

from pathlib import Path
from threading import get_ident
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetRegistryError,
)
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import Annotation, TimeSeriesSpec
from timenet.viewer import records as routes
from timenet.viewer.app import create_app
from timenet.viewer.records import RecordLimits
from timenet.writer import TimeFWriter


HEADERS = {"Authorization": "Bearer secret"}
IDENTITY = {"dataset_id": "timenet/hello-world", "version": "1.0.0"}
OPERATIONS = ["record", "signal-window"]
# Routes that take a dataset reference but no record.
DATASET_OPERATIONS = [*OPERATIONS, "dataset-annotations"]


def _spec(**fields):
    return TimeSeriesSpec(spec_type="test", name="Test", unit_value=None, nullable=True, **fields)


@pytest.fixture
def dataset():
    dataset = make_dataset()
    spec = _spec(dtype="float64")
    child = Source(
        name="Child",
        metadata={"nested": [False, 1]},
        signals=(
            Signal(name="Ordinal", spec=spec, time_axis=OrdinalAxis(), data=[1, None]),
            Signal.from_loader(
                name="Irregular",
                spec=spec,
                time_axis=IrregularAxis(first_us=5, last_us=10),
                n_values=2,
                loader=lambda: pa.array([1.0, 2.0]),
                time_offsets_loader=lambda: pa.array([5, 10], type=pa.int64()),
            ),
        ),
    )
    child.annotate(Annotation(key="label", value=True))
    regular = Signal(
        name="Regular", spec=spec, time_axis=RegularAxis.from_rate_hz(2), data=[float(n) for n in range(16)]
    )
    record = Record(
        start_time=dataset.records[0].start_time,
        sources=(Source(name="Monitor", signals=(regular,), sources=(child,)),),
        metadata={"subject": 7},
    )
    record.annotate(Annotation(key="cohort", value="A"))
    dataset.add_record(record=record)
    dataset.annotate(Annotation(key="answer_options", value=["yes", "no"], description="Allowed answers."))
    return dataset


@pytest.fixture
def record_params(dataset):
    record = dataset.records[-1]
    regular = next(signal for signal in record.signals if signal.name == "Regular")
    return {**IDENTITY, "record_id": record.id, "signal_id": regular.id, "start": 0, "stop": 4}


@pytest.fixture
def registry(tmp_path, dataset):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset)
    return registry


@pytest.fixture
def get(registry, record_params):
    def request(operation="record", params=None, **options):
        client = TestClient(create_app(registry, token="secret", origin="http://testserver", **options))
        return client.get(f"/api/v1/{operation}", params={**record_params, **(params or {})}, headers=HEADERS)

    return request


@pytest.fixture
def select_record(monkeypatch):
    def select(record):
        def records(self, *, record_ids, with_annotations):
            assert record_ids == [record.id]
            assert with_annotations is True
            yield record

        monkeypatch.setattr(TimeFReader, "iter_records", records)

    return select


@pytest.fixture
def no_global_reads(monkeypatch):
    forbidden = MagicMock(side_effect=AssertionError("listed records or hydrated tasks"))
    for name in ("record_ids", "iter_tasks", "read_tasks"):
        monkeypatch.setattr(TimeFReader, name, forbidden)


@pytest.fixture
def reads(monkeypatch):
    original, calls = Signal.read_steps, []

    def read(self, start, stop):
        calls.append((self.id, start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(Signal, "read_steps", read)
    return calls


def test_record_metadata_remains_lazy(get, dataset, monkeypatch, no_global_reads):
    for name in ("read_steps", "to_arrow", "time_offsets_us"):
        monkeypatch.setattr(Signal, name, MagicMock(side_effect=AssertionError("loaded values or offsets")))
    response = get()
    assert response.status_code == 200
    detail = response.json()
    record = dataset.records[-1]
    assert detail["id"] == str(record.id)
    assert detail["start_time"] == 9_007_199_254_740_993
    assert "task_ids" not in detail
    assert detail["annotations"][0]["key"] == "cohort"
    assert [(s["id"], s["parent_source_id"]) for s in detail["sources"]] == [("0", None), ("0.0", "0")]
    assert {s["id"] for s in detail["signals"]} == {str(signal.id) for signal in record.signals}
    assert {s["name"]: s["source_id"] for s in detail["signals"]} == {
        "Regular": "0",
        "Ordinal": "0.0",
        "Irregular": "0.0",
    }
    assert {s["axis"]["kind"] for s in detail["signals"]} == {"regular", "ordinal", "irregular"}
    assert detail["metadata"] == {"subject": 7}
    assert detail["sources"][1]["metadata"] == {"nested": [False, 1]}
    assert detail["sources"][1]["annotations"][0]["value_type"] == "bool"
    assert next(s for s in detail["signals"] if s["name"] == "Irregular")["axis"]["first_us"] == 5


def test_dataset_annotations_skip_record_hydration(get, monkeypatch, no_global_reads):
    monkeypatch.setattr(TimeFReader, "iter_records", MagicMock(side_effect=AssertionError("hydrated a record")))
    response = get("dataset-annotations")
    assert response.status_code == 200
    [annotation] = response.json()["annotations"]
    assert annotation["key"] == "answer_options"
    assert annotation["value_type"] == "list"
    assert annotation["value"] == ["yes", "no"]
    assert annotation["description"] == "Allowed answers."
    assert annotation["span"] is None


def test_dataset_without_annotations_returns_an_empty_list(tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(make_dataset())
    client = TestClient(create_app(registry, token="secret", origin="http://testserver"))
    response = client.get("/api/v1/dataset-annotations", params=IDENTITY, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"annotations": []}


@pytest.mark.parametrize(
    "extra,status",
    [
        ({"start": -1}, 422),
        ({"stop": 2, "start": 2}, 422),
        ({"stop": 2001}, 413),
    ],
)
def test_invalid_windows_rejected_before_open(get, registry, monkeypatch, extra, status):
    opener = MagicMock(side_effect=AssertionError("opened"))
    monkeypatch.setattr(registry, "open_version", opener)
    response = get("signal-window", extra)
    assert response.status_code == status
    assert response.json()["detail"]["operation"] == "signal-window"
    opener.assert_not_called()


@pytest.mark.parametrize(
    "operation,params,status",
    [
        ("record", {"record_id": 999_999}, 404),
        ("signal-window", {"record_id": 999_999}, 404),
        # Signal 1 belongs to the first record, not the requested one.
        ("signal-window", {"signal_id": 1}, 404),
        ("signal-window", {"stop": 17}, 422),
        ("record", {"record_id": ""}, 422),
        ("record", {"record_id": "absent"}, 422),
        ("record", {"record_id": 0}, 422),
    ],
)
def test_missing_objects_and_invalid_arguments(get, reads, operation, params, status):
    assert get(operation, params).status_code == status
    assert reads == []


@pytest.mark.parametrize("operation", DATASET_OPERATIONS)
def test_authentication_and_direct_isolation(registry, record_params, monkeypatch, operation):
    manifest = registry.get_manifest(**IDENTITY)
    forbidden = MagicMock(side_effect=AssertionError("accessed registry"))
    for name in ("open_version", "get_manifest"):
        monkeypatch.setattr(registry, name, forbidden)
    client = TestClient(create_app(registry, token="secret", origin="http://testserver", initial_manifest=manifest))
    path = f"/api/v1/{operation}"
    for extra, status in [
        ({"version": "1.1.0"}, 404),
        ({"dataset_id": "other/name"}, 404),
    ]:
        assert client.get(path, params={**record_params, **extra}, headers=HEADERS).status_code == status
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "operation,field,limit",
    [
        ("record", "hierarchy", 1),
        ("record", "annotations", 0),
        ("record", "response_bytes", 80),
        ("signal-window", "response_bytes", 80),
        ("dataset-annotations", "annotations", 0),
        ("dataset-annotations", "response_bytes", 80),
    ],
)
def test_collection_and_response_caps(get, reads, operation, field, limit):
    response = get(operation, record_limits=RecordLimits(**{field: limit}))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "operation_limit"
    assert response.json()["detail"]["limit"] == limit
    windowed = operation == "signal-window" and field == "response_bytes"
    assert [(start, stop) for _, start, stop in reads] == ([(0, 4)] if windowed else [])


@pytest.mark.parametrize(
    "case",
    [
        (TimeNetRegistryError, 503, "temporarily_unavailable"),
        (NotImplementedError, 501, "unsupported_operation"),
    ],
)
def test_sanitized_expected_failures(get, registry, monkeypatch, case):
    error, status, code = case
    monkeypatch.setattr(registry, "open_version", MagicMock(side_effect=error("secret /server/path")))
    response = get()
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "/server/path" not in response.text


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "failure,expected", [(None, 200), ("iteration", 500), ("projection", 500), ("cap", 413), ("missing", 404)]
)
def test_same_worker_cleanup(get, monkeypatch, operation, failure, expected):
    events, iterators = [], []

    class TrackingReader(TimeFReader):
        def __init__(self, version):
            super().__init__(version)
            events.append(("open", get_ident()))

        def iter_records(self, *args, **kwargs):
            def iterate():
                try:
                    if failure == "iteration":
                        raise TimeFFormatError("private")
                    yield from super(TrackingReader, self).iter_records(*args, **kwargs)
                finally:
                    events.append(("iterator", get_ident()))

            iterator = iterate()
            iterators.append(iterator)  # Prevent GC from masking a missing explicit close.
            return iterator

        def read_signal(self, *args, **kwargs):
            try:
                if failure == "iteration":
                    raise TimeFFormatError("private")
                return super().read_signal(*args, **kwargs)
            finally:
                events.append(("lookup", get_ident()))

        def close(self):
            events.append(("reader", get_ident()))
            super().close()

    monkeypatch.setattr(routes, "TimeFReader", TrackingReader)
    if failure == "projection":
        monkeypatch.setattr(
            routes,
            "_signal" if operation == "record" else "_window",
            MagicMock(side_effect=TimeFFormatError("private")),
        )
    response = get(
        operation,
        {"record_id": 999_999} if failure == "missing" else None,
        record_limits=(
            RecordLimits(**{"hierarchy" if operation == "record" else "response_bytes": 0})
            if failure == "cap"
            else RecordLimits()
        ),
    )
    assert response.status_code == expected
    assert [event for event, _ in events] == ["open", "iterator" if operation == "record" else "lookup", "reader"]
    assert len({thread for _, thread in events}) == 1


@pytest.mark.parametrize(
    "case",
    [
        ("int32", (), [2**31 - 1, None], None),
        ("bool", (), [True], "unsupported_plot"),
        ("str", (), ["a"], "unsupported_plot"),
        ("enum", (), ["a"], "unsupported_plot"),
        ("float32", (2,), [[1, 2]], "unsupported_plot"),
    ],
)
def test_plot_eligibility(get, monkeypatch, reads, case):
    dtype, shape, values, code = case
    spec = _spec(dtype=dtype, value_shape=shape, categories=("a",) if dtype == "enum" else ())
    data = pa.array(values, type=pa.list_(pa.float32(), 2)) if shape else values
    signal = Signal(name="Test", spec=spec, time_axis=OrdinalAxis(), data=data)
    record = TimeFDataset(metadata=make_dataset().metadata).add_record(
        record=Record(sources=(Source(name="source", signals=(signal,)),))
    )
    monkeypatch.setattr(TimeFReader, "read_signal", MagicMock(return_value=signal))
    response = get("signal-window", {"record_id": record.id, "signal_id": signal.id, "stop": len(values)})
    assert response.status_code == (422 if code else 200)
    if code:
        assert response.json()["detail"]["code"] == code
        assert reads == []
    else:
        assert response.json()["values"] == values
        assert reads == [(signal.id, 0, len(values))]


@pytest.mark.parametrize(
    "values,status,code",
    [
        ([None, float("nan"), float("inf"), -float("inf")], 200, None),
        ([2**53 - 1, None, 1, 2], 200, None),
        ([2**53, 1, 2, 3], 422, "unsupported_plot"),
        ([1.0], 500, "corrupt_artifact"),
    ],
)
def test_window_value_validation(get, monkeypatch, *, values, status, code):
    monkeypatch.setattr(Signal, "read_steps", lambda *args: pa.array(values))
    response = get("signal-window", {"start": 4, "stop": 8})
    assert response.status_code == status
    if code:
        assert response.json()["detail"]["code"] == code
    else:
        assert response.json()["values"] == ([None] * 4 if isinstance(values[1], float) else values)
        assert response.json()["nonfinite_steps"] == ([5, 6, 7] if isinstance(values[1], float) else [])


def test_large_metadata_rejected_before_projection(get, select_record, monkeypatch):
    record = make_dataset().records[0]
    record.metadata["large"] = "😺" * 10000
    select_record(record)
    monkeypatch.setattr(routes, "_signal", MagicMock(side_effect=AssertionError("projected before byte guard")))
    response = get(params={"record_id": record.id}, record_limits=RecordLimits(response_bytes=100))
    assert response.status_code == 413


def test_real_window_across_storage_chunks(tmp_path, reads, no_global_reads, monkeypatch):
    dataset = make_dataset()
    record = dataset.records[1]
    signal = max(record.signals, key=lambda item: item.n_values)
    expected = signal.to_arrow().slice(5, 20).to_pylist()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "registry", dataset, chunk_max_bytes=32) as writer:
        writer.write()
    monkeypatch.setattr(Signal, "time_offsets_us", MagicMock(side_effect=AssertionError("loaded offsets")))
    client = TestClient(create_app(LocalRegistry(tmp_path / "registry"), token="secret", origin="http://testserver"))
    response = client.get(
        "/api/v1/signal-window",
        params={**IDENTITY, "record_id": record.id, "signal_id": signal.id, "start": 5, "stop": 25},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["values"] == expected
    assert response.json()["nonfinite_steps"] == []
    assert reads == [(signal.id, 5, 25)]


@pytest.mark.parametrize("name", ["Regular", "Ordinal", "Irregular"])
def test_windows_skip_record_hydration_and_record_collection_caps(get, dataset, monkeypatch, reads, name):
    signal = next(item for item in dataset.records[-1].signals if item.name == name)
    forbidden = MagicMock(side_effect=AssertionError("record, annotation, or offset hydration"))
    monkeypatch.setattr(TimeFReader, "iter_records", forbidden)
    monkeypatch.setattr(DuckDBControlReader, "read_records", forbidden)
    monkeypatch.setattr(DuckDBControlReader, "_read_annotations", forbidden)
    monkeypatch.setattr(Signal, "time_offsets_us", forbidden)
    response = get(
        "signal-window", {"signal_id": signal.id, "stop": 2}, record_limits=RecordLimits(hierarchy=0, annotations=0)
    )
    assert response.status_code == 200
    assert response.json()["signal_id"] == str(signal.id)
    assert len(response.json()["values"]) == 2
    assert reads == [(signal.id, 0, 2)]
    forbidden.assert_not_called()


def test_window_reports_corrupt_signal_as_format_error_not_missing(get, registry, record_params):
    version = registry.open_version(IDENTITY["dataset_id"], IDENTITY["version"])
    with connect_control(Path(version.resolve_file_path("control.duckdb"))) as connection:
        connection.execute("UPDATE signals SET dtype = 'broken' WHERE signal_id = ?", [record_params["signal_id"]])
    response = get("signal-window")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "corrupt_artifact"
    assert "broken" not in response.text


@pytest.mark.parametrize(("record_id", "expected"), [(999_999, (404, "not_found")), (None, (500, "corrupt_artifact"))])
def test_record_validation_failure_is_404_only_for_an_unknown_id(get, monkeypatch, record_id, expected):
    def records(self, *, record_ids, with_annotations):
        raise TimeFValidationError("stored record failed validation")
        yield

    monkeypatch.setattr(TimeFReader, "iter_records", records)
    response = get("record", None if record_id is None else {"record_id": record_id})
    assert (response.status_code, response.json()["detail"]["code"]) == expected
    assert response.json()["detail"]["operation"] == "record"


@pytest.mark.parametrize("operation", DATASET_OPERATIONS)
def test_record_routes_use_the_canonical_version(registry, get, operation):
    manifest = registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    opener = MagicMock(wraps=registry.open_version)
    registry.open_version = opener
    assert get(operation, {"version": "01.0.0"}, initial_manifest=manifest).status_code == 200
    opener.assert_called_once_with(IDENTITY["dataset_id"], "1.0.0")
