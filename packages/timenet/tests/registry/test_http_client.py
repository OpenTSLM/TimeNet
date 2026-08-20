import io

import httpx
import pytest

from timenet.errors import DatasetNotFoundError, RegistryError
from timenet.registry.remote._http import RegistryHttpClient, timenet_user_agent


def _client(handler, *, token=None):
    return RegistryHttpClient("http://api.local/", token=token, transport=httpx.MockTransport(handler))


def test_user_agent_is_timenet_versioned():
    assert timenet_user_agent().startswith("timenet/")


def test_get_json_sets_prefix_auth_and_user_agent():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={"ok": True})

    with _client(handler, token="tok_abc") as client:
        assert client.get_json("/datasets") == {"ok": True}
    assert seen["path"] == "/api/v1/datasets"
    assert seen["auth"] == "Bearer tok_abc"
    assert seen["ua"].startswith("timenet/")


def test_anonymous_sends_no_authorization():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    with _client(handler) as client:
        client.get_json("/datasets")
    assert seen["auth"] is None


def test_404_maps_to_dataset_not_found():
    def handler(request):
        return httpx.Response(404, json={"detail": "nope"})

    with _client(handler) as client, pytest.raises(DatasetNotFoundError):
        client.get_json("/datasets/o/n")


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_error_statuses_map_to_registry_error(status):
    def handler(request):
        return httpx.Response(status, json={"detail": "x"})

    with _client(handler) as client, pytest.raises(RegistryError):
        client.get_json("/datasets")


def test_resolve_presigned_reads_location_without_following():
    def handler(request):
        assert request.url.path == "/api/v1/datasets/o/n/1.0.0/download/a.parquet"
        return httpx.Response(307, headers={"location": "http://blob.local/obj/a?sig=1"})

    with _client(handler) as client:
        url = client.resolve_presigned("o/n", "1.0.0", "a.parquet")
    assert url == "http://blob.local/obj/a?sig=1"


def test_blob_reads_send_no_authorization():
    seen = {}

    def handler(request):
        if request.url.host == "blob.local":
            seen["auth"] = request.headers.get("authorization")
            seen["ua"] = request.headers.get("user-agent")
            body = b"0123456789"
            rng = request.headers.get("range")
            if rng:
                start, _, end = rng.removeprefix("bytes=").partition("-")
                s, e = int(start), int(end)
                return httpx.Response(206, content=body[s : e + 1])
            return httpx.Response(200, content=body)
        return httpx.Response(307, headers={"location": "http://blob.local/obj/a"})

    with _client(handler, token="tok_abc") as client:
        url = client.resolve_presigned("o/n", "1.0.0", "a.parquet")
        sink = io.BytesIO()
        client.stream_to(url, sink)
        assert sink.getvalue() == b"0123456789"
    assert seen["auth"] is None
    assert not (seen["ua"] or "").startswith("timenet/")


def test_stream_to_maps_error_status_on_unread_body():
    # An iterator body makes the response streaming, so .text raises ResponseNotRead until it is
    # read; the error mapping must still surface as a RegistryError.
    def handler(request):
        return httpx.Response(403, content=iter([b"denied"]))

    with _client(handler) as client, pytest.raises(RegistryError):
        client.stream_to("http://blob.local/obj/a", io.BytesIO())
