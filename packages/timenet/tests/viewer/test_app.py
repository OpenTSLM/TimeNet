"""HTTP contracts for viewer relationship details."""

from fastapi.testclient import TestClient

from timenet.client import TimeNet
from timenet.testing import make_dataset
from timenet.viewer.app import create_app
from timenet.writer import TimeFWriter


def test_task_detail_and_annotations_expose_public_relationships(tmp_path):  # noqa: PLR0914
    """Expose task references and annotation owners without leaking storage keys."""
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()

    registry_root = tmp_path
    with TimeNet(registry_root).open_reader("timenet/hello-world") as reader:
        app = create_app(reader, token="test-token", port=8000, registry="file:///fixture-registry")
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            headers = {"Authorization": "Bearer test-token"}
            shell = client.get("/")
            plotly = client.get("/static/plotly-basic.min.js")
            session = client.get("/api/v1/session", headers=headers)
            signal_owner = client.post(
                "/api/v1/owners/record",
                json={"object_type": "Signal", "object_id": "ts-shared"},
                headers=headers,
            )
            records_page = client.post("/api/v1/records/query", json={"limit": 1}, headers=headers)
            tasks_page = client.post("/api/v1/tasks/query", json={"limit": 1}, headers=headers)
            annotations_page = client.post("/api/v1/annotations/query", json={"limit": 1}, headers=headers)
            window = client.post(
                "/api/v1/windows",
                json={"record_id": "record-0", "signal_id": "ts-shared", "stop": 16},
                headers=headers,
            )
            task = client.post("/api/v1/tasks/detail", json={"task_id": "task-answer-0"}, headers=headers)
            localized = client.post("/api/v1/tasks/detail", json={"task_id": "task-localize-0"}, headers=headers)
            annotations = client.post("/api/v1/annotations/query", json={}, headers=headers)
            filtered = client.post(
                "/api/v1/annotations/query",
                json={
                    "object_type": "Record",
                    "object_id": "record-0",
                    "span_type": "static",
                    "value_query": "A",
                },
                headers=headers,
            )

    assert shell.status_code == 200
    assert session.json()["registry"] == "file:///fixture-registry"
    assert "plotly-basic.min.js" in shell.text
    assert plotly.status_code == 200
    assert "Plotly" in plotly.text
    assert signal_owner.json()["record_id"] == "record-0"
    assert records_page.json()["next_cursor"]
    assert tasks_page.json()["next_cursor"]
    assert annotations_page.json()["next_cursor"]
    assert window.json()["signal_name"] == "a"
    assert window.json()["spec_name"] == "Sine"
    assert window.json()["unit"] == "dimensionless"
    assert task.status_code == 200
    detail = task.json()
    assert detail["parent_task_ids"] == ["task-cls-0"]
    assert detail["targets"] == [
        {
            "kind": "text",
            "value": "Normal.",
            "record_id": None,
            "signal_id": None,
            "start": None,
            "end": None,
            "signal_ids": [],
        }
    ]
    assert len(detail["annotation_refs"]) == 1
    annotation_ref = detail["annotation_refs"][0]
    assert annotation_ref["occurrence_id"]
    assert annotation_ref | {"occurrence_id": None} == {
        "field": "input_annotations",
        "occurrence_id": None,
        "object_type": "Record",
        "object_id": "record-0",
        "name": "cohort",
        "value": "A",
        "span_type": "static",
        "start_us": None,
        "end_us": None,
    }
    assert localized.status_code == 200
    assert localized.json()["targets"] == [
        {
            "kind": "time_point",
            "value": None,
            "record_id": None,
            "signal_id": None,
            "start": "500000",
            "end": None,
            "signal_ids": [],
        },
        {
            "kind": "time_interval",
            "value": None,
            "record_id": None,
            "signal_id": None,
            "start": "0",
            "end": "250000",
            "signal_ids": ["ts-shared"],
        },
    ]
    assert annotations.status_code == 200
    assert {item["object_id"] for item in annotations.json()["items"]} >= {"record-0", "ts-shared"}
    assert filtered.status_code == 200
    assert [item["name"] for item in filtered.json()["items"]] == ["cohort"]
