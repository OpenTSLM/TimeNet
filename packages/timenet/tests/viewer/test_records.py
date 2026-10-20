"""Record API isolation, bounded lazy reads, projections, and cleanup."""

from threading import get_ident
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, OrdinalAxis
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetDownloadError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import Annotation, TimeSeriesSpec, Version
from timenet.viewer import records as routes
from timenet.viewer.app import create_app
from timenet.viewer.records import RecordLimits
from timenet.writer import TimeFWriter


HEADERS = {"Authorization": "Bearer secret"}
IDENTITY = {"dataset_id": "timenet/hello-world", "version": "1.0.0"}
WINDOW = {**IDENTITY, "record_id": "record-0", "signal_id": "ts-shared", "start": 0, "stop": 4}
OPERATIONS = ["record", "signal-window"]


def _spec(**fields):
    return TimeSeriesSpec(spec_type="test", name="Test", unit_value=None, nullable=True, **fields)


@pytest.fixture
def registry(tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    dataset = make_dataset()
    spec = _spec(dtype="float64")
    child = Source(
        id="child",
        name="Child",
        metadata={"nested": [False, 1]},
        signals=(
            Signal(id="ordinal", name="Ordinal", spec=spec, time_axis=OrdinalAxis(), data=[1, None]),
            Signal.from_loader(
                id="irregular",
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
    dataset.records[0].sources[0].sources = (child,)
    dataset.records[0].metadata["subject"] = 7
    registry.store(dataset)
    return registry


@pytest.fixture
def get(registry):
    def request(operation="record", params=None, **options):
        client = TestClient(create_app(registry, token="secret", origin="http://testserver", **options))
        return client.get(f"/api/v1/{operation}", params={**WINDOW, **(params or {})}, headers=HEADERS)

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


def test_record_metadata_remains_lazy(get, monkeypatch, no_global_reads):
    for name in ("read_steps", "to_arrow", "time_offsets_us"):
        monkeypatch.setattr(Signal, name, MagicMock(side_effect=AssertionError("loaded values or offsets")))
    response = get()
    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == "record-0"
    assert detail["start_time"] == 9_007_199_254_740_993
    assert "task_ids" not in detail
    assert detail["annotations"][0]["occurrence_id"]
    assert detail["sources"][0]["parent_source_id"] is None
    assert {s["id"] for s in detail["signals"]} == {"ts-shared", "ts-cos-0", "ordinal", "irregular"}
    assert {s["axis"]["kind"] for s in detail["signals"]} == {"regular", "ordinal", "irregular"}
    assert detail["metadata"] == {"subject": 7}
    assert detail["sources"][1]["parent_source_id"] == "source-record-0"
    assert detail["sources"][1]["metadata"] == {"nested": [False, 1]}
    assert detail["sources"][1]["annotations"][0]["value_type"] == "bool"
    assert next(s for s in detail["signals"] if s["id"] == "irregular")["axis"]["first_us"] == 5


@pytest.mark.parametrize(
    "extra,status",
    [
        ({"start": -1}, 422),
        ({"stop": 0}, 422),
        ({"stop": 2, "start": 2}, 422),
        ({"start": "1.5"}, 422),
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
        ("record", {"record_id": "absent"}, 404),
        ("signal-window", {"record_id": "absent"}, 404),
        ("signal-window", {"signal_id": "ts-long-1"}, 404),
        ("signal-window", {"stop": 17}, 422),
        ("record", {"record_id": ""}, 422),
    ],
)
def test_missing_objects_and_invalid_arguments(get, reads, operation, params, status):
    assert get(operation, params).status_code == status
    assert reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
def test_authentication_and_direct_isolation(registry, monkeypatch, operation):
    manifest = registry.get_manifest(**IDENTITY)
    forbidden = MagicMock(side_effect=AssertionError("accessed registry"))
    for name in ("open_version", "get_manifest"):
        monkeypatch.setattr(registry, name, forbidden)
    client = TestClient(create_app(registry, token="secret", origin="http://testserver", initial_manifest=manifest))
    path = f"/api/v1/{operation}"
    assert client.get(path, params=WINDOW).status_code == 401
    assert client.get(path, params=WINDOW, headers={**HEADERS, "Origin": "https://foreign.test"}).status_code == 403
    for extra, status in [
        ({"version": "1.1.0"}, 404),
        ({"dataset_id": "other/name"}, 404),
        ({"version": "latest"}, 422),
        ({"dataset_id": "../escape"}, 422),
    ]:
        assert client.get(path, params={**WINDOW, **extra}, headers=HEADERS).status_code == status
    forbidden.assert_not_called()


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("field,limit", [("hierarchy", 1), ("annotations", 0), ("response_bytes", 80)])
def test_collection_and_response_caps(get, reads, operation, field, limit):
    response = get(operation, record_limits=RecordLimits(**{field: limit}))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "operation_limit"
    assert response.json()["detail"]["limit"] == limit
    assert reads == ([("ts-shared", 0, 4)] if operation == "signal-window" and field == "response_bytes" else [])


@pytest.mark.parametrize(
    "case",
    [
        (TimeFFormatError, 500, "corrupt_artifact"),
        (TimeNetDatasetNotFoundError, 404, "not_found"),
        (TimeNetAccessError, 403, "restricted_access"),
        (TimeNetDownloadError, 503, "temporarily_unavailable"),
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
        {"record_id": "absent" if failure == "missing" else "record-0"},
        record_limits=RecordLimits(hierarchy=0) if failure == "cap" else RecordLimits(),
    )
    assert response.status_code == expected
    assert [event for event, _ in events] == ["open", "iterator", "reader"]
    assert len({thread for _, thread in events}) == 1


@pytest.mark.parametrize(
    "case",
    [
        ("int32", (), [2**31 - 1, None], None),
        ("bool", (), [True], "unsupported_plot"),
        ("str", (), ["a"], "unsupported_plot"),
        ("enum", (), ["a"], "unsupported_plot"),
        ("float32", (2,), [[1, 2]], "unsupported_plot"),
        ("float64", (), [], "invalid_parameters"),
    ],
)
def test_plot_eligibility(get, select_record, reads, case):
    dtype, shape, values, code = case
    spec = _spec(dtype=dtype, value_shape=shape, categories=("a",) if dtype == "enum" else ())
    data = pa.array(values, type=pa.list_(pa.float32(), 2)) if shape else values or [1.0]
    signal = Signal(id="test", name="Test", spec=spec, time_axis=OrdinalAxis(), data=data)
    if not values:
        object.__setattr__(signal, "n_values", 0)  # Current TimeF forbids empty signals.
    select_record(Record(record_id="test-record", sources=(Source(name="source", signals=(signal,)),)))
    response = get("signal-window", {"record_id": "test-record", "signal_id": "test", "stop": max(1, len(values))})
    assert response.status_code == (422 if code else 200)
    if code:
        assert response.json()["detail"]["code"] == code
        assert reads == []
    else:
        assert response.json()["values"] == values
        assert reads == [("test", 0, len(values))]


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
    response = get(record_limits=RecordLimits(response_bytes=100))
    assert response.status_code == 413


def test_real_window_across_storage_chunks(tmp_path, reads, no_global_reads, monkeypatch):
    dataset = make_dataset()
    signal = next(s for s in dataset.records[1].walk_signals() if s.id == "ts-long-1")
    expected = signal.to_arrow().slice(5, 20).to_pylist()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "registry", dataset, chunk_max_bytes=32) as writer:
        writer.write()
    monkeypatch.setattr(Signal, "time_offsets_us", MagicMock(side_effect=AssertionError("loaded offsets")))
    client = TestClient(create_app(LocalRegistry(tmp_path / "registry"), token="secret", origin="http://testserver"))
    response = client.get(
        "/api/v1/signal-window",
        params={**WINDOW, "record_id": "record-1", "signal_id": "ts-long-1", "start": 5, "stop": 25},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["values"] == expected
    assert response.json()["nonfinite_steps"] == []
    assert reads == [("ts-long-1", 5, 25)]


def test_explicit_catalog_version(get, registry):
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    newer.add_record(record=Record(record_id="new-only", sources=(Source(name="New source"),)))
    registry.store(newer)
    assert get(params={"record_id": "new-only"}).status_code == 404
    assert get(params={"record_id": "new-only", "version": "1.1.0"}).status_code == 200


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("record_id", "expected"), [("absent", (404, "not_found")), ("record-0", (500, "corrupt_artifact"))]
)
def test_record_validation_failure_is_404_only_for_an_unknown_id(get, monkeypatch, operation, record_id, expected):
    def records(self, *, record_ids, with_annotations):
        raise TimeFValidationError("stored record failed validation")
        yield

    monkeypatch.setattr(TimeFReader, "iter_records", records)
    response = get(operation, {"record_id": record_id})
    assert (response.status_code, response.json()["detail"]["code"]) == expected
    assert response.json()["detail"]["operation"] == operation


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("direct_launch", [False, True])
def test_record_routes_use_the_canonical_version(registry, get, operation, direct_launch):
    manifest = registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    opener = MagicMock(wraps=registry.open_version)
    registry.open_version = opener
    options = {"initial_manifest": manifest} if direct_launch else {}
    assert get(operation, {"version": "01.0.0"}, **options).status_code == 200
    opener.assert_called_once_with(IDENTITY["dataset_id"], "1.0.0")
