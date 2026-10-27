"""Record API isolation, bounded lazy reads, projections, and cleanup."""

from threading import get_ident
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import Annotation, TimeSeriesSpec
from timenet.viewer import records as routes
from timenet.viewer.app import create_app
from timenet.viewer.records import RecordLimits


HEADERS = {"Authorization": "Bearer secret"}
IDENTITY = {"dataset_id": "timenet/hello-world", "version": "1.0.0"}
OPERATIONS = ["record"]


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
    regular = Signal(name="Regular", spec=spec, time_axis=RegularAxis.from_rate_hz(2), data=[1.0, 2.0])
    record = Record(
        start_time=dataset.records[0].start_time,
        sources=(Source(name="Monitor", signals=(regular,), sources=(child,)),),
        metadata={"subject": 7},
    )
    record.annotate(Annotation(key="cohort", value="A"))
    dataset.add_record(record=record)
    return dataset


@pytest.fixture
def record_params(dataset):
    return {**IDENTITY, "record_id": dataset.records[-1].id}


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


@pytest.mark.parametrize(
    "operation,params,status",
    [
        ("record", {"record_id": 999_999}, 404),
        ("record", {"record_id": ""}, 422),
        ("record", {"record_id": "absent"}, 422),
        ("record", {"record_id": 0}, 422),
    ],
)
def test_missing_objects_and_invalid_arguments(get, reads, operation, params, status):
    assert get(operation, params).status_code == status
    assert reads == []


@pytest.mark.parametrize("operation", OPERATIONS)
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


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("field,limit", [("hierarchy", 1), ("annotations", 0), ("response_bytes", 80)])
def test_collection_and_response_caps(get, reads, operation, field, limit):
    response = get(operation, record_limits=RecordLimits(**{field: limit}))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "operation_limit"
    assert response.json()["detail"]["limit"] == limit
    assert reads == []


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
@pytest.mark.parametrize("failure,expected", [(None, 200), ("projection", 500)])
def test_same_worker_cleanup(get, monkeypatch, operation, failure, expected):
    events, iterators = [], []

    class TrackingReader(TimeFReader):
        def __init__(self, version):
            super().__init__(version)
            events.append(("open", get_ident()))

        def iter_records(self, *args, **kwargs):
            def iterate():
                try:
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
        monkeypatch.setattr(routes, "_signal", MagicMock(side_effect=TimeFFormatError("private")))
    response = get(operation)
    assert response.status_code == expected
    assert [event for event, _ in events] == ["open", "iterator", "reader"]
    assert len({thread for _, thread in events}) == 1


def test_large_metadata_rejected_before_projection(get, select_record, monkeypatch):
    record = make_dataset().records[0]
    record.metadata["large"] = "😺" * 10000
    select_record(record)
    monkeypatch.setattr(routes, "_signal", MagicMock(side_effect=AssertionError("projected before byte guard")))
    response = get(params={"record_id": record.id}, record_limits=RecordLimits(response_bytes=100))
    assert response.status_code == 413


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("record_id", "expected"), [(999_999, (404, "not_found")), (None, (500, "corrupt_artifact"))])
def test_record_validation_failure_is_404_only_for_an_unknown_id(get, monkeypatch, operation, record_id, expected):
    def records(self, *, record_ids, with_annotations):
        raise TimeFValidationError("stored record failed validation")
        yield

    monkeypatch.setattr(TimeFReader, "iter_records", records)
    response = get(operation, None if record_id is None else {"record_id": record_id})
    assert (response.status_code, response.json()["detail"]["code"]) == expected
    assert response.json()["detail"]["operation"] == operation


@pytest.mark.parametrize("operation", OPERATIONS)
def test_record_routes_use_the_canonical_version(registry, get, operation):
    manifest = registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    opener = MagicMock(wraps=registry.open_version)
    registry.open_version = opener
    assert get(operation, {"version": "01.0.0"}, initial_manifest=manifest).status_code == 200
    opener.assert_called_once_with(IDENTITY["dataset_id"], "1.0.0")
