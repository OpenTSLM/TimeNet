"""Task browsing uses public iterators, explicit versions, and request-owned readers."""

from threading import get_ident
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from timenet.dataset import Signal
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    StepInterval,
    StepPoint,
    TimeInterval,
    TimePoint,
    Version,
)
from timenet.types.splits import Split
from timenet.viewer import tasks as routes
from timenet.viewer.app import create_app
from timenet.viewer.models import project_span, project_task


HEADERS = {"Authorization": "Bearer secret"}
IDENTITY = {"dataset_id": "timenet/hello-world", "version": "1.0.0"}


@pytest.fixture
def task_registry(tmp_path):
    dataset = make_dataset()
    record = dataset.records[0]
    for index in range(25):
        dataset.add_task(
            task=ClassificationTask(
                id=f"page-{index:03}",
                inputs=(record,),
                targets=("normal",),
                split=Split.TRAIN,
                prompt="Classify " + "x" * 250,
            )
        )
    rich = AnswerTask(
        id="rich",
        inputs=(record,),
        split=Split.TEST,
        prompt="Full prompt",
        rationale="Because",
        from_tasks=(dataset.tasks[0],),
        scope=TimeInterval.seconds(0, 1),
        targets=(True, 7, 2.5, "answer", record, next(iter(record.walk_signals())), TimePoint.seconds(0.5)),
        metadata={"nested": {"items": [1, False]}},
    )
    rich.annotate(Annotation(key="review", value=True, source="test", confidence=0.8))
    dataset.add_task(task=rich)
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset)
    return registry


@pytest.fixture
def client(task_registry):
    return TestClient(create_app(task_registry, token="secret", origin="http://testserver"))


def test_task_pages_split_and_empty_state(client):
    first = client.get("/api/v1/tasks", params={**IDENTITY, "split": "train"}, headers=HEADERS).json()
    assert first["offset"] == 0
    assert first["limit"] == 20
    assert first["has_more"] is True
    assert len(first["items"]) == 20
    assert first["items"][0] == {
        "id": "page-000",
        "task_type": "classification",
        "split": "train",
        "prompt_preview": ("Classify " + "x" * 250)[:200],
    }
    final = client.get("/api/v1/tasks", params={**IDENTITY, "split": "train", "offset": 20}, headers=HEADERS).json()
    assert [item["id"] for item in final["items"]] == [f"page-{i:03}" for i in range(20, 25)]
    assert final["has_more"] is False
    for params in ({"split": "validation"}, {"offset": 1000}):
        empty = client.get("/api/v1/tasks", params={**IDENTITY, **params}, headers=HEADERS).json()
        assert empty["items"] == []
        assert empty["has_more"] is False
    exact = client.get("/api/v1/tasks", params={**IDENTITY, "split": "train", "limit": 25}, headers=HEADERS).json()
    assert len(exact["items"]) == 25
    assert exact["has_more"] is False


def test_detail_preserves_native_targets_and_references_without_values(client, monkeypatch):
    def no_values(*args, **kwargs):
        pytest.fail("task browsing loaded signal values")

    monkeypatch.setattr(Signal, "to_arrow", no_values)
    monkeypatch.setattr(Signal, "read_steps", no_values)
    response = client.get("/api/v1/task", params={**IDENTITY, "task_id": "rich", "split": "train"}, headers=HEADERS)
    assert response.status_code == 200
    detail = response.json()
    assert detail["split"] == "test"  # Lookup is independent of list filtering.
    assert detail["prompt"] == "Full prompt"
    assert detail["rationale"] == "Because"
    assert detail["input_record_ids"] == ["record-0"]
    assert detail["parent_task_ids"] == ["task-cls-0"]
    assert detail["metadata"] == {"nested": {"items": [1, False]}}
    assert detail["targets"][:4] == [
        {"kind": "bool", "value": True},
        {"kind": "int", "value": 7},
        {"kind": "float", "value": 2.5},
        {"kind": "str", "value": "answer"},
    ]
    assert detail["targets"][4] == {"kind": "record", "id": "record-0"}
    assert detail["targets"][5]["kind"] == "signal"
    assert set(detail["targets"][5]) == {"kind", "id"}
    assert detail["targets"][6]["value"] == {
        "kind": "point",
        "unit": "microseconds",
        "start": 500000,
        "stop": None,
        "signal_ids": None,
    }
    assert detail["scope"]["stop"] == 1000000
    assert detail["annotations"][0]["value_type"] == "bool"
    assert detail["annotations"][0]["value"] is True
    assert detail["annotations"][0]["occurrence_id"]
    answer = client.get("/api/v1/task", params={**IDENTITY, "task_id": "task-answer-0"}, headers=HEADERS).json()
    assert answer["input_annotations"][0]["key"] == "cohort"
    scalar = client.get("/api/v1/task", params={**IDENTITY, "task_id": "task-scalar-0"}, headers=HEADERS).json()
    assert scalar["configuration"]["target_name"] == "mean_rate"
    assert scalar["configuration"]["unit"]
    assert client.get("/api/v1/task", params={**IDENTITY, "task_id": "absent"}, headers=HEADERS).status_code == 404


def test_empty_and_annotation_targets_remain_distinct():
    assert project_task(AnswerTask(targets=None)).targets is None
    assert project_task(AnswerTask(targets=())).targets == []
    assert project_span(StepPoint(time_series_id="ordinal", start=2)).model_dump() == {
        "kind": "point",
        "unit": "steps",
        "start": 2,
        "stop": None,
        "signal_ids": ["ordinal"],
    }
    assert project_span(StepInterval(time_series_id="ordinal", start=2, stop=4)).stop == 4


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("/api/v1/tasks", {"offset": -1}),
        ("/api/v1/tasks", {"limit": 0}),
        ("/api/v1/tasks", {"limit": 101}),
        ("/api/v1/tasks", {"split": "unknown"}),
        ("/api/v1/task", {"task_id": ""}),
        ("/api/v1/task", {}),
    ],
)
def test_invalid_task_arguments_do_not_open_a_version(task_registry, monkeypatch, path, extra):
    opener = MagicMock(side_effect=AssertionError("opened an invalid request"))
    monkeypatch.setattr(task_registry, "open_version", opener)
    client = TestClient(create_app(task_registry, token="secret", origin="http://testserver"))
    assert client.get(path, params={**IDENTITY, **extra}, headers=HEADERS).status_code == 422
    opener.assert_not_called()


@pytest.mark.parametrize("path", ["/api/v1/tasks", "/api/v1/task"])
@pytest.mark.parametrize(
    ("identity", "status"),
    [
        ({"dataset_id": "../escape", "version": "1.0.0"}, 422),
        ({"dataset_id": "timenet/hello-world", "version": "latest"}, 422),
        ({"dataset_id": "timenet/hello-world", "version": "1.1.0"}, 404),
        ({"dataset_id": "other/dataset", "version": "1.0.0"}, 404),
        ({}, 422),
    ],
)
def test_direct_identity_is_rejected_before_opening(task_registry, monkeypatch, path, identity, status):
    manifest = task_registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    opener = MagicMock(side_effect=AssertionError("opened another version"))
    monkeypatch.setattr(task_registry, "open_version", opener)
    client = TestClient(
        create_app(task_registry, token="secret", origin="http://testserver", initial_manifest=manifest)
    )
    assert client.get(path, params={**identity, "task_id": "rich"}, headers=HEADERS).status_code == status
    opener.assert_not_called()


@pytest.mark.parametrize("path", ["/api/v1/tasks", "/api/v1/task"])
def test_task_authentication_precedes_opening(task_registry, monkeypatch, path):
    opener = MagicMock()
    monkeypatch.setattr(task_registry, "open_version", opener)
    client = TestClient(create_app(task_registry, token="secret", origin="http://testserver"))
    assert client.get(path, params={**IDENTITY, "task_id": "rich"}).status_code == 401
    assert client.get(path, params=IDENTITY, headers={**HEADERS, "Origin": "https://foreign.test"}).status_code == 403
    opener.assert_not_called()


@pytest.mark.parametrize(
    ("route", "failure", "expected"),
    [
        (("/api/v1/tasks", {"limit": 1}), None, 200),
        (("/api/v1/task", {"task_id": "page-000"}), None, 200),
        (("/api/v1/task", {"task_id": "missing"}), None, 404),
        (("/api/v1/tasks", {}), "iteration", 500),
        (("/api/v1/task", {"task_id": "page-000"}), "projection", 500),
    ],
)
def test_reader_and_iterator_cleanup_on_same_worker(task_registry, monkeypatch, route, failure, expected):
    path, params = route
    events = []
    yielded = []

    class TrackingReader(TimeFReader):
        def __init__(self, version):
            super().__init__(version)
            events.append(("open", get_ident()))

        def iter_tasks(self, *args, **kwargs):
            try:
                if failure == "iteration":
                    raise TimeFFormatError("private /server/path")
                for task in super().iter_tasks(*args, **kwargs):
                    yielded.append(task.id)
                    events.append(("yield", get_ident()))
                    yield task
            finally:
                events.append(("iterator closed", get_ident()))

        def close(self):
            events.append(("reader closed", get_ident()))
            super().close()

    monkeypatch.setattr(routes, "TimeFReader", TrackingReader)
    if failure == "projection":
        monkeypatch.setattr(routes, "project_task", MagicMock(side_effect=RuntimeError("projection failed")))
    client = TestClient(
        create_app(task_registry, token="secret", origin="http://testserver"), raise_server_exceptions=False
    )
    response = client.get(path, params={**IDENTITY, **params}, headers=HEADERS)
    assert response.status_code == expected
    assert events[-2][0] == "iterator closed"
    assert events[-1][0] == "reader closed"
    assert len({thread for _, thread in events}) == 1
    if params == {"limit": 1}:
        assert len(yielded) == 2
    if params == {"task_id": "page-000"}:
        assert yielded == ["page-000"]
    if failure == "iteration":
        assert "/server/path" not in response.text


def test_catalog_task_requests_use_the_explicit_version_and_direct_mode_can_browse(task_registry):
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    task_registry.store(newer)
    client = TestClient(create_app(task_registry, token="secret", origin="http://testserver"))
    assert client.get("/api/v1/task", params={**IDENTITY, "task_id": "rich"}, headers=HEADERS).status_code == 200
    assert (
        client.get(
            "/api/v1/task",
            params={**IDENTITY, "version": "1.1.0", "task_id": "rich"},
            headers=HEADERS,
        ).status_code
        == 404
    )
    manifest = task_registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    direct = TestClient(
        create_app(task_registry, token="secret", origin="http://testserver", initial_manifest=manifest)
    )
    page = direct.get("/api/v1/tasks", params={**IDENTITY, "split": "test"}, headers=HEADERS)
    assert page.status_code == 200
    assert [item["id"] for item in page.json()["items"]] == ["rich"]
    assert direct.get("/api/v1/task", params={**IDENTITY, "task_id": "rich"}, headers=HEADERS).status_code == 200


@pytest.mark.parametrize("path", ["/api/v1/tasks", "/api/v1/task"])
@pytest.mark.parametrize("direct_launch", [False, True])
def test_task_routes_use_the_canonical_version(task_registry, path, direct_launch):
    manifest = task_registry.get_manifest(IDENTITY["dataset_id"], IDENTITY["version"])
    opener = MagicMock(wraps=task_registry.open_version)
    task_registry.open_version = opener
    app = create_app(
        task_registry, token="secret", origin="http://testserver", initial_manifest=manifest if direct_launch else None
    )
    params = {**IDENTITY, "version": "01.0.0", "task_id": "rich"}
    assert TestClient(app).get(path, params=params, headers=HEADERS).status_code == 200
    opener.assert_called_once_with(IDENTITY["dataset_id"], "1.0.0")


@pytest.mark.parametrize("path", ["/api/v1/tasks", "/api/v1/task"])
def test_task_registry_failures_use_structured_errors(task_registry, monkeypatch, path):
    monkeypatch.setattr(task_registry, "open_version", MagicMock(side_effect=TimeNetDownloadError("/server/path")))
    client = TestClient(create_app(task_registry, token="secret", origin="http://testserver"))
    response = client.get(path, params={**IDENTITY, "task_id": "rich"}, headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "temporarily_unavailable"
    assert response.json()["detail"]["operation"] == path.rsplit("/", 1)[-1]
    assert "/server/path" not in response.text


def test_missing_task_uses_the_structured_error(client):
    response = client.get("/api/v1/task", params={**IDENTITY, "task_id": "absent"}, headers=HEADERS)
    assert response.status_code == 404
    assert response.json()["detail"] == {
        "code": "not_found",
        "message": "Task not found.",
        "operation": "task",
        "limit": None,
    }
