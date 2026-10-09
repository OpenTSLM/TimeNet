"""Inspector regressions for reduction, bounds, composition and API security."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from typing import Any, cast

from fastapi.testclient import TestClient
import pyarrow as pa
import pytest

from timenet.client import TimeNet
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.reader import TimeFReader
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, DatasetRef, LockedDependency, TimeSeriesSpec
from timenet.viewer.app import create_app
from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.launch import open_reader
from timenet.viewer.schemas import WindowQuery
from timenet.viewer.windows import _reduce, window_result
from timenet.viewer.workers import clone_reader
from timenet.writer import TimeFWriter


HEADERS = {"Authorization": "Bearer test"}


def _typed_window(reader: TimeFReader, query: WindowQuery) -> dict[str, Any]:
    return cast("dict[str, Any]", window_result(ViewerInspection(reader), query))


def store(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def dataset_with_signals(*signals):
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(
        record=Record(record_id="record", sources=(Source(id="source", name="Sensor", signals=signals),))
    )
    return dataset


def signal(values, *, signal_id="signal", rate=100, dtype="float64"):
    return Signal(
        id=signal_id,
        name=signal_id,
        spec=TimeSeriesSpec(spec_type=signal_id, name=signal_id, dtype=dtype, unit_value=None, nullable=True),
        time_axis=RegularAxis.from_rate_hz(rate),
        data=values,
    )


@pytest.fixture
def client(tmp_path):
    dataset = make_dataset()
    store(tmp_path, dataset)
    with (
        TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader,
        TestClient(
            create_app(reader, token="test", port=8000), base_url="http://127.0.0.1:8000", headers=HEADERS
        ) as client,
    ):
        yield client


def test_reduction_preserves_order_endpoints_extrema_and_gaps():
    values = [8, 7, None, 100, -10, 3, 2, 1, 0, 9]
    reduced = _reduce(tuple((i, i // 2, value) for i, value in enumerate(values)), 2)
    indexes = [item[0] for item in reduced]
    assert indexes == sorted(set(indexes))
    assert {0, 2, 3, 4, 9} <= set(indexes)
    assert reduced[1][2] is None


def test_streaming_large_windows_and_independent_raw_pages(tmp_path, monkeypatch):
    dataset = dataset_with_signals(signal(pa.array(range(100_001), type=pa.float64())))
    store(tmp_path, dataset)
    reads = []
    original = TimeFReader._load_signal_range

    def bounded(self, key, signal_id, spec, start, stop):  # noqa: PLR0913, PLR0917
        reads.append(stop - start)
        assert stop - start <= 20_000
        return original(self, key, signal_id, spec, start, stop)

    monkeypatch.setattr(TimeFReader, "_load_signal_range", bounded)
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader:
        full = _typed_window(reader, WindowQuery(record_id="record", signal_id="signal", full=True, width=256))
        assert full["scanned_count"] == "100001"
        assert full["reduced"]
        assert len(full["items"]) <= 5 * 256
        assert full["items"][0]["index"] == "0"
        assert full["items"][-1]["index"] == "100000"
        raw = _typed_window(reader, WindowQuery(record_id="record", signal_id="signal", stop=100001, mode="raw"))
        assert len(raw["items"]) == 200
        assert raw["next_start"] == "200"
        page = _typed_window(
            reader, WindowQuery(record_id="record", signal_id="signal", start=200, stop=100001, mode="raw")
        )
        assert page["items"][0]["index"] == "200"
        initial = _typed_window(reader, WindowQuery(record_id="record", signal_id="signal"))
        assert initial["scanned_count"] == "1000"
    assert max(reads) <= 20_000


def test_time_window_resolves_each_sample_rate(tmp_path):
    dataset = dataset_with_signals(
        signal(pa.array(range(1000), type=pa.float64()), signal_id="slow"),
        signal(pa.array(range(2500), type=pa.float64()), signal_id="fast", rate=250),
    )
    store(tmp_path, dataset)
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader:
        results = [
            _typed_window(reader, WindowQuery(record_id="record", signal_id=name, start_us=1_000_000, end_us=2_000_000))
            for name in ("slow", "fast")
        ]
    assert [result["scanned_count"] for result in results] == ["100", "250"]
    assert [result["items"][0]["x"] for result in results] == ["1000000", "1000000"]


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("str", pa.array(["hello", None, "world"]), ["hello", None, "world"]),
        ("bool", pa.array([True, None, False]), ["True", None, "False"]),
        ("int32", pa.array([2**30 + 1, None, -(2**30)], type=pa.int32()), [str(2**30 + 1), None, str(-(2**30))]),
    ],
)
def test_typed_raw_values(tmp_path, dtype, values, expected):
    dataset = dataset_with_signals(signal(values, dtype=dtype))
    store(tmp_path, dataset)
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader:
        result = _typed_window(reader, WindowQuery(record_id="record", signal_id="signal", mode="raw"))
    assert [item["display"] for item in result["items"]] == expected
    assert result["items"][1]["kind"] == "null"


def test_security_checks_precede_parsing_and_cover_shell(client):
    assert client.post("/api/v1/records/query", content="invalid", headers={"Authorization": "bad"}).status_code == 401
    assert client.post("/api/v1/records/query", json={"unexpected": 1}).status_code == 422
    assert client.post("/api/v1/records/query", content=b" " * 65537).status_code == 413
    for path in ("/", "/static/app.js", "/api/v1/session"):
        response = client.get(path, headers={"Host": "attacker.invalid"})
        assert response.status_code == 403
        assert response.headers["Cache-Control"] == "no-store"
    assert client.post("/api/v1/records/query", json={}, headers={"Origin": "https://evil.invalid"}).status_code == 403


def test_interactive_reader_calls_have_one_thread_owner(client, monkeypatch):
    threads = set()
    original = TimeFReader._control_reader

    def track(self):
        threads.add(get_ident())
        return original(self)

    monkeypatch.setattr(TimeFReader, "_control_reader", track)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post("/api/v1/records/query", json={}), range(12)))
    assert all(response.status_code == 200 for response in responses)
    assert len(threads) == 1


def test_composed_browsing_path_and_parent_verification(tmp_path):
    parent = make_dataset()
    parent_path = store(tmp_path, parent)
    metadata = parent.metadata.model_copy(
        update={
            "dataset_id": "test/child",
            "parents": (DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version),),
        }
    )
    child = TimeFDataset(metadata=metadata)
    child.set_dependencies(
        (
            LockedDependency(
                dataset_id=parent.metadata.dataset_id,
                version=parent.metadata.dataset_version,
                manifest_checksum=file_checksum(parent_path / "manifest.json"),
            ),
        )
    )
    with TimeNet(tmp_path).open_reader(parent.metadata.dataset_id) as reader:
        record = child.import_record(next(reader.iter_records(["record-0"])), parent=parent.metadata.dataset_id)
    record.annotate(Annotation(key="overlay", value="child"))
    child_path = store(tmp_path, child)
    with open_reader(None, None, str(tmp_path), child_path) as reader:
        sources = ViewerInspection(reader).sources_page("record-0", None)
        assert sources.items
        signals = ViewerInspection(reader).signals_page("record-0", sources.items[0].source_id)
        assert signals.items
        assert ViewerInspection(reader).read_steps("record-0", "ts-shared", 0, 2).to_pylist()


def test_inspection_does_not_hydrate_record_hierarchy(client, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("signal inspection hydrated its whole record")

    monkeypatch.setattr(TimeFReader, "iter_records", forbidden)
    response = client.post("/api/v1/windows", json={"record_id": "record-0", "signal_id": "ts-shared", "stop": 4})
    assert response.status_code == 200


def test_annotation_window_includes_record_owners_and_continues(client):
    items = []
    after = None
    while True:
        response = client.post(
            "/api/v1/annotations/window",
            json={"record_id": "record-0", "signal_id": "ts-shared", "limit": 1, "after": after},
        )
        assert response.status_code == 200, response.text
        page = response.json()
        items.extend(page["items"])
        after = page["next_cursor"]
        if after is None:
            break
    assert {item["object_type"] for item in items} >= {"Record", "Signal"}
    assert all(item["dataset_id"] == "timenet/hello-world" for item in items)


def test_tree_pages_expose_items_after_fifty(tmp_path):
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    sources = tuple(Source(id=f"source-{i:03}", name=f"Source {i}") for i in range(55))
    dataset.add_record(record=Record(record_id="record", sources=sources))
    store(tmp_path, dataset)
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader:
        page = ViewerInspection(reader).sources_page("record", None)
        assert len(page.items) == 50
        remaining = ViewerInspection(reader).sources_page("record", None, after=page.next_after)
        assert len(remaining.items) == 5
        assert remaining.next_after is None


def test_task_relationship_pagination_and_absent_inline_targets(tmp_path):
    dataset = make_dataset()
    dataset.add_task(
        task=AnswerTask(
            id="many", inputs=(dataset.records[0],), prompt="x" * 600, targets=tuple(f"answer-{i}" for i in range(205))
        )
    )
    dataset.add_task(
        task=AnswerTask(
            id="absent", inputs=(dataset.records[0],), target_annotations=(dataset.records[0].annotations[0],)
        )
    )
    store(tmp_path, dataset)
    with (
        TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as reader,
        TestClient(
            create_app(reader, token="test", port=8000), base_url="http://127.0.0.1:8000", headers=HEADERS
        ) as client,
    ):
        first = client.post("/api/v1/tasks/detail", json={"task_id": "many"}).json()
        assert len(first["targets"]) == 200
        assert first["prompt_truncated"]
        assert first["has_inline_targets"]
        assert "scope" in first and "configuration" in first
        second = client.post("/api/v1/tasks/detail", json={"task_id": "many", "after": first["next_after"]}).json()
        assert [item["value"] for item in second["targets"]] == [f"answer-{i}" for i in range(200, 205)]
        assert second["next_after"] is None
        absent = client.post("/api/v1/tasks/detail", json={"task_id": "absent"}).json()
        assert absent["has_inline_targets"] is False
        filtered = client.post(
            "/api/v1/tasks/query", json={"task_type": first["task_type"], "split": "__unassigned__"}
        ).json()
        assert "many" in {item["task_id"] for item in filtered["items"]}


def test_nonlocal_range_registry_is_opened_in_place(tmp_path, monkeypatch):
    from timenet.registry import S3Registry  # noqa: PLC0415

    client = TimeNet(tmp_path)
    registry = object.__new__(S3Registry)
    client._registry = registry
    sentinel = object()
    monkeypatch.setattr(registry, "resolve_versions", lambda *args: ())
    monkeypatch.setattr(registry, "open_closure", lambda closure: sentinel)
    monkeypatch.setattr(registry, "download_version", lambda *args, **kwargs: pytest.fail("S3 was materialized"))
    assert client.open_reader("test/data") is sentinel


def test_grandchild_irregular_offsets_overlays_and_locks(tmp_path):
    irregular = Signal.from_irregular(
        [1.0, 2.0, 3.0],
        time_offsets_us=[0, 0, 1000],
        id="signal",
        name="Irregular",
        spec=TimeSeriesSpec(spec_type="irregular", name="Irregular", unit_value=None),
    )
    parent = dataset_with_signals(irregular)
    parent_path = store(tmp_path, parent)
    current = parent
    for name in ("test/child", "test/grandchild"):
        with TimeNet(tmp_path).open_reader(current.metadata.dataset_id) as reader:
            child = TimeFDataset(
                metadata=current.metadata.model_copy(
                    update={
                        "dataset_id": name,
                        "parents": (
                            DatasetRef(
                                dataset_id=current.metadata.dataset_id, version=current.metadata.dataset_version
                            ),
                        ),
                    }
                )
            )
            child.set_dependencies(
                (
                    *reader._manifest.dependencies,
                    LockedDependency(
                        dataset_id=current.metadata.dataset_id,
                        version=current.metadata.dataset_version,
                        manifest_checksum=file_checksum(
                            tmp_path
                            / current.metadata.dataset_id
                            / str(current.metadata.dataset_version)
                            / "manifest.json"
                        ),
                    ),
                )
            )
            imported = child.import_record(next(reader.iter_records(["record"])), parent=current.metadata.dataset_id)
            imported.annotate(Annotation(key=name, value=True))
            child.add_task(
                task=AnswerTask(
                    id="inherited-annotation", inputs=(imported,), target_annotations=(imported.annotations[-1],)
                )
            )
        directory = store(tmp_path, child)
        current = child
    with (
        open_reader(None, None, None, directory) as reader,
        TestClient(
            create_app(reader, token="test", port=8000), base_url="http://127.0.0.1:8000", headers=HEADERS
        ) as client,
    ):
        assert ViewerInspection(reader).offsets("record", "signal", 1, 3).to_pylist() == [0, 1000]
        overlay = client.post("/api/v1/annotations/window", json={"record_id": "record", "signal_id": "signal"}).json()
        assert {item["name"] for item in overlay["items"]} == {"test/child", "test/grandchild"}
        browsed = client.post("/api/v1/annotations/query", json={"object_type": "Record"}).json()
        assert {item["name"] for item in browsed["items"]} == {"test/child", "test/grandchild"}
        detail = client.post("/api/v1/tasks/detail", json={"task_id": "inherited-annotation"}).json()
        assert detail["annotation_refs"][0]["name"] == "test/grandchild"
        owner = client.post("/api/v1/owners/record", json={"object_type": "Signal", "object_id": "signal"}).json()
        assert owner["record_id"] == "record"
    manifest = parent_path / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(TimeFFormatError, match="checksum"):
        open_reader(None, None, str(tmp_path), directory)


def test_worker_clone_preserves_shared_dependency_identity(tmp_path):
    dataset = make_dataset()
    store(tmp_path, dataset)
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as parent:
        graph = TimeFReader(parent._version, parents={"left": parent, "right": parent})
        with clone_reader(graph) as cloned:
            assert cloned._parents["left"] is cloned._parents["right"]
            assert cloned._parents["left"] is not parent
            assert cloned._parents["left"]._control_reader() is not parent._control_reader()
