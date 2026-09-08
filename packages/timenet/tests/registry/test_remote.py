from pathlib import Path

from _fake_registry import build_fake, build_publish_fake, range_server
import pytest

from timenet.client import TimeNet
from timenet.errors import TimeNetRegistryError
from timenet.manifest import Manifest
from timenet.registry import RemoteRegistry
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Domain
from timenet.writer import TimeFWriter


@pytest.fixture
def version_dir(tmp_path) -> tuple[Path, Manifest]:
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    manifest = Manifest.from_json(next(tmp_path.rglob("manifest.json")).read_text())
    return next(tmp_path.rglob("manifest.json")).parent, manifest


def _remote(version_dir, tmp_path, **kw):
    directory, _ = version_dir
    transport, requests = build_fake(directory, token=kw.pop("token", None))
    registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache", **kw)
    return registry, requests


def test_list_datasets_maps_summaries(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    metadatas = registry.list_datasets()
    assert [m.dataset_id for m in metadatas] == [manifest.metadata.dataset_id]
    assert str(metadatas[0].dataset_version) == str(manifest.metadata.dataset_version)
    assert all(isinstance(d, Domain) for d in metadatas[0].domains)


def test_search_filters_by_domain(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    domain = manifest.metadata.domains[0]
    assert registry.search(domain=domain)
    assert registry.search(domain=Domain.FINANCE) == [] or manifest.metadata.domains == (Domain.FINANCE,)


def test_open_file_streams_bytes(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    directory, manifest = version_dir
    relpath = manifest.files.all_parts()[0]
    with registry.open_file(manifest.metadata.dataset_id, str(manifest.metadata.dataset_version), relpath) as fh:
        assert fh.read() == (directory / relpath).read_bytes()


def test_token_required_when_configured(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path, token="tok_rw")
    # constructed without passing the token -> anonymous -> 401 mapped to TimeNetRegistryError
    with pytest.raises(TimeNetRegistryError):
        registry.list_datasets()


def test_full_download_writes_identical_bytes(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    directory, manifest = version_dir
    dest = tmp_path / "out" / manifest.metadata.dataset_id / str(manifest.metadata.dataset_version)
    registry.download_version(manifest.metadata.dataset_id, str(manifest.metadata.dataset_version), dest)
    for part in manifest.files.all_parts():
        assert (dest / part).read_bytes() == (directory / part).read_bytes()
    assert (dest / "manifest.json").exists()


def test_load_full_round_trips(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    client = TimeNet(registry=registry, storage_path=tmp_path / "storage")
    loaded = client.load(manifest.metadata.dataset_id, download_mode="full")
    assert_datasets_equal(make_dataset(), loaded)


def test_load_on_demand_round_trips(version_dir, tmp_path):
    directory, manifest = version_dir
    # fsspec reads presigned URLs over aiohttp, which a MockTransport can't serve, so point the download
    # redirect at a real range-capable server. The lazy reads run during the assert, so keep it inside.
    with range_server(directory) as blob_base:
        transport, _ = build_fake(directory, blob_base=blob_base)
        registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache")
        client = TimeNet(registry=registry, storage_path=tmp_path / "storage")
        loaded = client.load(manifest.metadata.dataset_id, download_mode="on_demand")
        assert_datasets_equal(make_dataset(), loaded)


def test_on_demand_serves_from_cache_without_network(version_dir, tmp_path):
    registry, requests = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    dataset_id = manifest.metadata.dataset_id
    version = str(manifest.metadata.dataset_version)
    # Prime the cache with a full download.
    registry.download_version(dataset_id, version, registry._cache_dir / dataset_id / version)
    requests.clear()
    handle = registry.open_version(dataset_id, version, mode="on_demand")
    import timenet.reader as reader_mod  # noqa: PLC0415

    _ = reader_mod.TimeFReader(handle).read().samples  # force reads
    assert not any(r.url.host == "blob.local" for r in requests)  # every read came from cache


def test_store_publishes_uploads_and_finalizes(tmp_path):
    transport, state = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", token="tok_rw", transport=transport, cache_dir=tmp_path)
    dataset = make_dataset()
    version = registry.store(dataset)
    assert version == str(dataset.metadata.dataset_version)
    assert state["finalized"] is True
    # every declared file was uploaded
    assert set(state["store"]) == set(state["published"])


def test_store_rejects_a_publish_list_that_drops_a_manifest_file(tmp_path):
    import httpx  # noqa: PLC0415

    def handler(request):
        # The service omits every declared file from its upload list. store() must reject that before
        # uploading, rather than publish an incomplete version.
        if request.url.path.endswith("/publish"):
            return httpx.Response(200, json={"files": []})
        return httpx.Response(200, json={})

    registry = RemoteRegistry(
        "http://api.local", token="tok_rw", transport=httpx.MockTransport(handler), cache_dir=tmp_path
    )
    with pytest.raises(TimeNetRegistryError, match="disagrees with the manifest"):
        registry.store(make_dataset(), force=True)


def test_remote_registry_uses_client_storage_path(tmp_path):
    # A URL-configured remote registry must cache under the client's storage_path, so download() and
    # on-demand load() share one cache-first directory.
    client = TimeNet(registry="https://registry.example.test", storage_path=tmp_path / "store")
    assert isinstance(client._registry, RemoteRegistry)
    assert client._registry._cache_dir == client._storage == tmp_path / "store"


def test_store_without_write_token_is_rejected(tmp_path):
    transport, _ = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path)  # anonymous
    with pytest.raises(TimeNetRegistryError):
        registry.store(make_dataset())
