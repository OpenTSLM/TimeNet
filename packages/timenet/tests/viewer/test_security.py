"""The viewer's request boundary checks the host, origin, and token, and adds browser headers."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from timenet.errors import TimeFValidationError
from timenet.viewer.security import install_security


ORIGIN = "http://127.0.0.1:8765"
AUTHORIZED = {"Authorization": "Bearer secret"}


def _secured_client(origin: str, base_url: str) -> TestClient:
    app = FastAPI()
    install_security(app, token="secret", origin=origin)

    @app.get("/api/v1/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/")
    def shell() -> str:
        return "shell"

    return TestClient(app, base_url=base_url)


@pytest.fixture
def client():
    return _secured_client(ORIGIN, ORIGIN)


def test_authorized_same_origin_request_passes_with_security_headers(client):
    response = client.get("/api/v1/ping", headers={**AUTHORIZED, "Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_only_api_routes_require_the_token(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/v1/ping").status_code == 401


@pytest.mark.parametrize(
    "authorization",
    # Latin-1 bytes, as a client sends them: a non-ASCII token must be refused, not crash the server.
    ["Bearer wrong-token", "Basic secret", "Bearer", "Bearer s\xe9cret".encode("latin-1")],
)
def test_api_rejects_incorrect_credentials(client, authorization):
    assert client.get("/api/v1/ping", headers={"Authorization": authorization}).status_code == 401


@pytest.mark.parametrize(
    ("header", "foreign", "status"),
    # A browser always sends the Host of the server it connected to, so only the host name is
    # checked; another page on the same machine is caught by its Origin, including the port.
    [
        ("Host", "localhost:8765", 400),
        ("Host", "elsewhere.test", 400),
        ("Origin", "http://127.0.0.1:8766", 403),
        ("Origin", "https://elsewhere.test", 403),
        ("Sec-Fetch-Site", "cross-site", 403),
        ("Sec-Fetch-Site", "same-site", 403),
    ],
)
def test_request_boundary_rejects_foreign_requests(client, header, foreign, status):
    response = client.get("/api/v1/ping", headers={**AUTHORIZED, header: foreign})
    assert response.status_code == status
    # Every rejection, the host check's included, still carries the security headers.
    assert response.headers["X-Frame-Options"] == "DENY"


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
def test_browser_normalized_origin_is_accepted(configured_origin, browser_origin):
    client = _secured_client(configured_origin, browser_origin)
    response = client.get("/api/v1/ping", headers={**AUTHORIZED, "Origin": browser_origin})
    assert response.status_code == 200


def test_public_origin_rejects_the_same_host_on_another_port():
    client = _secured_client("https://viewer.example.test:443", "https://viewer.example.test")
    foreign = {**AUTHORIZED, "Origin": "https://viewer.example.test:8443"}
    assert client.get("/api/v1/ping", headers=foreign).status_code == 403


@pytest.mark.parametrize("origin", ["http://", "ftp://viewer.example.test", "https://viewer.example.test/path"])
def test_origin_must_be_an_http_origin_with_a_host(origin):
    with pytest.raises(TimeFValidationError, match=r"HTTP\(S\) origin"):
        install_security(FastAPI(), token="secret", origin=origin)
