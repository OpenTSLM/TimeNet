"""The viewer serves authenticated catalog and pinned-manifest facts."""

import socket
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner

from timenet.cli.app import app as cli_app
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry, S3Registry
from timenet.testing import make_dataset
from timenet.types import ClassificationTask, Domain, License, Version
from timenet.viewer import launch
from timenet.viewer.app import create_app
from timenet.viewer.launch import open_viewer, serve


@pytest.fixture
def registry(tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(make_dataset())
    return registry


@pytest.fixture
def catalog_registry(registry):
    other = make_dataset()
    other._metadata = other.metadata.model_copy(
        update={
            "dataset_id": "demo/other",
            "dataset_version": Version(2, 0, 0),
            "name": "Other Dataset",
            "domains": (Domain.FINANCE,),
            "license": License.APACHE_2_0,
        }
    )
    registry.store(other)
    other._metadata = other.metadata.model_copy(update={"dataset_version": Version(2, 1, 0)})
    other.add_tasks(tasks=[ClassificationTask(inputs=(other.records[0],), targets=("finance",), id="finance-extra")])
    registry.store(other)
    return registry


def test_open_viewer_can_open_catalog_or_pin_a_dataset(registry):
    original = registry.get_manifest("timenet/hello-world")
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    registry.store(newer)
    backend, selected = open_viewer(registry=str(registry.root))
    assert backend.list_datasets()[0].dataset_id == "timenet/hello-world"
    assert selected is None

    backend, manifest = open_viewer("timenet/hello-world", str(registry.root))
    assert manifest is not None
    assert manifest.metadata.dataset_id == "timenet/hello-world"
    assert str(manifest.metadata.dataset_version) == "1.1.0"
    _, pinned = open_viewer("timenet/hello-world@1.0.0", str(registry.root))
    assert pinned == original


def test_direct_manifest_api_returns_only_the_selected_version(registry):
    manifest = registry.get_manifest("timenet/hello-world")
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    registry.store(newer)
    client = TestClient(create_app(registry, token="secret", origin="http://testserver", initial_manifest=manifest))

    response = client.get(
        "/api/v1/manifest",
        params={"dataset_id": "timenet/hello-world", "version": "1.0.0"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    assert response.json() == manifest.to_dict()
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/api/v1/catalog", headers={"Authorization": "Bearer secret"}).status_code == 404
    assert client.get("/api/v1/manifest", headers={"Authorization": "Bearer secret"}).status_code == 422
    invalid = client.get(
        "/api/v1/manifest",
        params={"dataset_id": "../escape", "version": "1.0.0"},
        headers={"Authorization": "Bearer secret"},
    )
    assert invalid.status_code == 422
    other_version = client.get(
        "/api/v1/manifest",
        params={"dataset_id": "timenet/hello-world", "version": "1.1.0"},
        headers={"Authorization": "Bearer secret"},
    )
    assert other_version.status_code == 404


def test_catalog_pages_and_enriches_latest_manifests(catalog_registry):
    client = TestClient(create_app(catalog_registry, token="secret", origin="http://testserver"))
    headers = {"Authorization": "Bearer secret"}

    first = client.get("/api/v1/catalog", params={"limit": 1}, headers=headers)
    assert first.status_code == 200
    assert first.json()["total"] == 2
    assert len(first.json()["items"]) == 1
    expected = catalog_registry.get_manifest("demo/other", "2.1.0")
    item = first.json()["items"][0]
    assert item["metadata"] == expected.to_dict()["metadata"]
    assert first.json()["domains"] == ["finance", "general"]
    assert first.json()["licenses"] == ["Apache-2.0", "CC-BY-4.0"]
    assert item["counts"] == {
        "records": expected.counts.records,
        "sources": expected.counts.sources,
        "signals": expected.counts.signals,
        "tasks": expected.counts.tasks,
        "annotation_occurrences": expected.counts.annotation_occurrences,
    }
    assert set(item["task_types"]) == {"classification", "answer", "scalar_prediction", "temporal_localization"}
    assert set(item["signal_types"]) == {"sine", "cosine"}
    assert set(item["annotation_types"]) == {"age", "cohort", "stimulus", "artifact"}
    second = client.get("/api/v1/catalog", params={"offset": 1, "limit": 1}, headers=headers)
    assert second.json()["items"][0]["metadata"]["dataset_id"] == "timenet/hello-world"
    assert len(second.json()["items"]) == 1
    assert client.get("/api/v1/catalog", params={"limit": 51}, headers=headers).status_code == 422


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"query": "OTHER"}, ["demo/other"]),
        ({"query": "Hello"}, ["timenet/hello-world"]),
        ({"domain": "finance"}, ["demo/other"]),
        ({"license": "Apache-2.0"}, ["demo/other"]),
        ({"query": "Hello", "domain": "finance"}, []),
    ],
)
def test_catalog_filters_independently_and_intersects_them(catalog_registry, filters, expected_ids):
    client = TestClient(create_app(catalog_registry, token="secret", origin="http://testserver"))
    response = client.get("/api/v1/catalog", params=filters, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["total"] == len(expected_ids)
    assert [item["metadata"]["dataset_id"] for item in response.json()["items"]] == expected_ids


def test_catalog_manifest_api_pins_versions_and_reports_missing_versions(registry):
    original = registry.get_manifest("timenet/hello-world")
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    registry.store(newer)
    client = TestClient(create_app(registry, token="secret", origin="http://testserver"))
    headers = {"Authorization": "Bearer secret"}
    response = client.get(
        "/api/v1/manifest", params={"dataset_id": "timenet/hello-world", "version": "1.0.0"}, headers=headers
    )
    assert response.status_code == 200
    assert response.json() == original.to_dict()
    missing = client.get(
        "/api/v1/manifest", params={"dataset_id": "timenet/hello-world", "version": "9.9.9"}, headers=headers
    )
    assert missing.status_code == 404


@pytest.mark.parametrize("version", [" 1.0.0", "+1.0.0", "01.0.0"])
@pytest.mark.parametrize("direct_launch", [False, True])
def test_manifest_api_uses_the_canonical_version(registry, version, direct_launch):
    manifest = registry.get_manifest("timenet/hello-world")
    get_manifest = MagicMock(wraps=registry.get_manifest)
    registry.get_manifest = get_manifest
    app = create_app(
        registry, token="secret", origin="http://testserver", initial_manifest=manifest if direct_launch else None
    )
    response = TestClient(app).get(
        "/api/v1/manifest",
        params={"dataset_id": "timenet/hello-world", "version": version},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    assert response.json() == manifest.to_dict()
    if not direct_launch:
        get_manifest.assert_called_once_with("timenet/hello-world", "1.0.0")


@pytest.mark.parametrize(
    "authorization",
    # Latin-1 bytes, as a client sends them: a non-ASCII token must be refused, not crash the server.
    [None, "Bearer wrong-token", "Basic secret", "Bearer s\xe9cret".encode("latin-1")],
)
@pytest.mark.parametrize("path", ["/api/v1/catalog", "/api/v1/manifest?dataset_id=timenet/hello-world&version=1.0.0"])
def test_api_rejects_missing_or_incorrect_credentials(registry, path, authorization):
    client = TestClient(create_app(registry, token="secret", origin="http://testserver"))
    headers = {} if authorization is None else {"Authorization": authorization}
    assert client.get(path, headers=headers).status_code == 401


def test_s3_catalog_reports_unsupported_backend():
    # S3 can open a known dataset directly, but it has no catalog index for
    # list_datasets(). The viewer must turn that limitation into a clear 501.
    client = TestClient(create_app(S3Registry("s3://test-bucket"), token="secret", origin="http://testserver"))
    response = client.get("/api/v1/catalog", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 501
    assert "no catalog" in response.json()["detail"]


def test_request_boundary_and_packaged_assets(registry):
    client = TestClient(create_app(registry, token="secret", origin="http://testserver"))
    headers = {"Authorization": "Bearer secret"}

    assert client.get("/api/v1/catalog", headers={**headers, "Origin": "https://elsewhere.test"}).status_code == 403
    assert client.get("/api/v1/catalog", headers={**headers, "Host": "elsewhere.test"}).status_code == 403
    shell = client.get("/")
    assert shell.status_code == 200
    assert "/static/app.js" in shell.text
    assert client.get("/static/app.js").status_code == 200
    assert "MIT License" in client.get("/static/REACT-LICENSE").text
    assert client.get("/static/unknown.js").status_code == 404


@pytest.mark.parametrize("header", ["Host", "Origin"])
def test_request_boundary_rejects_same_host_on_a_different_port(registry, header):
    client = TestClient(
        create_app(registry, token="secret", origin="https://viewer.example.test:443"),
        base_url="https://viewer.example.test",
    )
    foreign = "viewer.example.test:8443" if header == "Host" else "https://viewer.example.test:8443"
    response = client.get("/api/v1/catalog", headers={"Authorization": "Bearer secret", header: foreign})
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("configured_origin", "browser_origin"),
    [
        ("https://viewer.example.test", "https://viewer.example.test"),
        ("https://VIEWER.example.test:443", "https://viewer.example.test"),
        ("http://VIEWER.example.test:80", "http://viewer.example.test"),
        ("https://viewer.example.test:8443", "https://viewer.example.test:8443"),
        ("https://[::1]:443", "https://[::1]"),
    ],
)
def test_browser_normalized_origin_is_accepted(registry, configured_origin, browser_origin):
    client = TestClient(
        create_app(registry, token="secret", origin=configured_origin),
        base_url=browser_origin,
    )
    response = client.get(
        "/api/v1/catalog",
        headers={"Authorization": "Bearer secret", "Origin": browser_origin},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(("direct_launch", "public_origin"), [(False, None), (True, "https://VIEWER.example.test:443")])
def test_serve_prints_bare_authenticated_url(registry, monkeypatch, capsys, direct_launch, public_origin):
    socket_factory = MagicMock()
    socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
    apps = []

    def skip_serving(server, **options):
        apps.append(server.config.app)

    monkeypatch.setattr(launch.uvicorn.Server, "run", skip_serving)
    browser = MagicMock()
    monkeypatch.setattr(launch.webbrowser, "open", browser)
    manifest = registry.get_manifest("timenet/hello-world") if direct_launch else None

    with monkeypatch.context() as socket_patch:
        socket_patch.setattr(launch.socket, "socket", socket_factory)
        serve(registry, initial_manifest=manifest, public_origin=public_origin, open_browser=False)

    output = capsys.readouterr()
    url = output.out.strip()
    parsed = urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}" == (
        "https://viewer.example.test" if public_origin else "http://127.0.0.1:12345"
    )
    assert parsed.path == "/"
    token = parse_qs(parsed.fragment)["token"][0]
    assert parse_qs(parsed.query) == (
        {"dataset": ["timenet/hello-world"], "version": ["1.0.0"]} if direct_launch else {}
    )
    assert output.out == url + "\n"
    assert output.err == ""
    browser.assert_not_called()
    client = TestClient(apps[0], base_url=f"{parsed.scheme}://{parsed.netloc}")
    path = "/api/v1/manifest" if direct_launch else "/api/v1/catalog"
    params = {"dataset_id": "timenet/hello-world", "version": "1.0.0"} if direct_launch else {}
    response = client.get(path, params=params, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    if direct_launch:
        assert manifest is not None
        assert response.json() == manifest.to_dict()
    else:
        assert response.json()["items"][0]["metadata"]["dataset_id"] == "timenet/hello-world"


def test_remote_bind_requires_https_origin(registry):
    with pytest.raises(TimeFValidationError, match="requires --public-origin"):
        serve(registry, host="192.0.2.1", open_browser=False)
    with pytest.raises(TimeFValidationError, match="HTTPS origin"):
        serve(registry, public_origin="http://example.test", open_browser=False)


def test_unbindable_address_reports_a_validation_error(registry):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(TimeFValidationError, match="cannot bind the viewer"):
            serve(registry, port=port, open_browser=False)
    with pytest.raises(TimeFValidationError, match="cannot bind the viewer"):
        serve(registry, host="::1", public_origin="https://viewer.example.test", open_browser=False)


def test_cli_view_passes_registry_and_optional_manifest(registry, monkeypatch):
    pinned = registry.get_manifest("timenet/hello-world")
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    registry.store(newer)
    received = []

    def fake_serve(backend, **options):
        received.append((backend, options))

    monkeypatch.setattr(launch, "serve", fake_serve)
    result = CliRunner().invoke(
        cli_app,
        ["view", "timenet/hello-world@1.0.0", "--registry", str(registry.root), "--no-browser"],
    )
    assert result.exit_code == 0, result.output
    assert received[0][0].root == registry.root
    assert received[0][1]["initial_manifest"] == pinned
    assert received[0][1]["open_browser"] is False
    result = CliRunner().invoke(cli_app, ["view", "--registry", str(registry.root), "--no-browser"])
    assert result.exit_code == 0, result.output
    assert received[1][0].root == registry.root
    assert received[1][1]["initial_manifest"] is None
