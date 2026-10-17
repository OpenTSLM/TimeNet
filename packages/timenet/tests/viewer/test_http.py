"""Authentication, request validation, and the browser API contract."""

import json
from pathlib import Path

from pydantic import TypeAdapter
import pytest

from timenet.viewer.contracts import ENDPOINTS


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"Authorization": ""}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Origin": "https://other.example"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
    ],
)
def test_untrusted_requests_are_rejected_before_body_parsing(client, headers, status):
    response = client.post("/api/v1/records/query", content="not json", headers=headers)
    assert response.status_code == status


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/v1/session"])
def test_host_validation_covers_assets_and_api(client, path):
    response = client.get(path, headers={"Host": "other.example"})
    assert response.status_code == 403
    assert response.headers["Cache-Control"] == "no-store"


def test_oversized_request_is_rejected(client):
    assert client.post("/api/v1/records/query", content=b" " * 65537).status_code == 413


def test_validation_errors_identify_the_invalid_field(client):
    response = client.post("/api/v1/windows", json={"record_id": "record-0"})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert "signal_id" in response.json()["message"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/owners/record", {"object_type": "Signal", "object_id": "absent"}),
        ("/api/v1/tasks/detail", {"task_id": "absent"}),
    ],
)
def test_missing_objects_return_not_found(client, path, body):
    assert client.post(path, json=body).status_code == 404


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/records/detail", {"record_id": "record-0", "signal_id": "ts-shared"}),
        ("/api/v1/tasks/detail", {"task_id": "task-localize-0"}),
        ("/api/v1/windows", {"record_id": "record-0", "signal_id": "ts-shared", "stop": 4}),
    ],
)
def test_success_responses_match_the_client_contract(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    operation = next(item for item in ENDPOINTS if item.path == path)
    TypeAdapter(operation.response).validate_python(response.json(), strict=True)


def test_generated_contract_matches_http_routes(registry_client):
    live_document = registry_client.app.openapi()
    generated_document = json.loads((Path(__file__).resolve().parents[4] / "frontend/schema.json").read_text())
    live_schemas = live_document["components"]["schemas"]
    generated_schemas = generated_document["components"]["schemas"]
    # The client accepts decimal strings to preserve exact microsecond bounds.
    for name in ("WindowQuery", "AnnotationWindowQuery"):
        for field in ("start_us", "end_us"):
            generated_schemas[name]["properties"][field]["anyOf"].remove({"type": "string", "pattern": "^-?[0-9]+$"})
    # FastAPI adds validation errors and expands query models and union aliases inline.
    assert live_schemas.keys() - generated_schemas.keys() == {"HTTPValidationError", "ValidationError"}
    assert generated_schemas.keys() - live_schemas.keys() == {"CatalogQuery", "SessionQuery", "TargetItem", "TaskScope"}
    for name in live_schemas.keys() & generated_schemas.keys():
        assert _without_null_schema_keywords(live_schemas[name]) == _without_null_schema_keywords(
            generated_schemas[name]
        ), name
    task_fields = live_schemas["TaskDetail"]["properties"]
    assert generated_schemas["TargetItem"] == task_fields["targets"]["items"]
    assert generated_schemas["TaskScope"] == {
        key: value for key, value in task_fields["scope"].items() if key != "title"
    }
    for operation in ENDPOINTS:
        if operation.query is not None:
            assert generated_schemas[operation.query.__name__] == TypeAdapter(operation.query).json_schema()
    live = live_document["paths"]
    generated = generated_document["paths"]
    assert set(generated) == {path for path in live if path.startswith("/api/")}
    for path, methods in generated.items():
        assert set(methods) == set(live[path])
        for method, expected in methods.items():
            actual = live[path][method]
            assert actual.get("requestBody") == expected.get("requestBody")
            assert {(p["name"], p["in"]): _without_null_schema_keywords(p) for p in actual.get("parameters", [])} == {
                (p["name"], p["in"]): _without_null_schema_keywords(p) for p in expected.get("parameters", [])
            }
            for status, response in expected["responses"].items():
                assert actual["responses"][status]["content"] == response["content"]


def _without_null_schema_keywords(value):
    # FastAPI omits null defaults and null constants from its OpenAPI document.
    if isinstance(value, dict):
        return {
            key: _without_null_schema_keywords(item)
            for key, item in value.items()
            if not (key in {"default", "const"} and item is None)
        }
    if isinstance(value, list):
        return [_without_null_schema_keywords(item) for item in value]
    return value
