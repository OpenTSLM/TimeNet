"""The viewer's request boundary checks the host, origin, and token, and adds browser headers."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from timenet.errors import TimeFValidationError
from timenet.viewer.security import install_security


ORIGIN = "http://127.0.0.1:8765"
AUTHORIZED = {"Authorization": "Bearer secret"}


@pytest.fixture
def client():
    app = FastAPI()
    install_security(app, token="secret", origin=ORIGIN)

    @app.get("/api/v1/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/")
    def shell() -> str:
        return "shell"

    return TestClient(app, base_url=ORIGIN)


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


def test_origin_without_a_host_is_rejected():
    with pytest.raises(TimeFValidationError, match="no host name"):
        install_security(FastAPI(), token="secret", origin="http://")
