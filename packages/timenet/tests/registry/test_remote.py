from pathlib import Path

from _fake_registry import build_fake, build_publish_fake
import httpx
import numpy as np
import pytest

from timenet.client import TimeNet
from timenet.control_plane import TimeFWriter
from timenet.errors import TimeNetRegistryError
from timenet.manifest import Manifest
from timenet.registry import RemoteRegistry
from timenet.testing import make_dataset
from timenet.types import Access, Domain, License


@pytest.fixture
def version_dir(tmp_path) -> tuple[Path, Manifest]:
    dataset = make_dataset(n_records=2, n_values=64)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    manifest = Manifest.from_json(next(tmp_path.rglob("manifest.json")).read_text())
    return next(tmp_path.rglob("manifest.json")).parent, manifest


def _remote(version_dir, tmp_path, **kw):
    directory, _ = version_dir
    transport, requests = build_fake(directory, token=kw.pop("token", None))
    registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache", **kw)
    return registry, requests


def test_list_datasets_maps_summaries(version_dir, tmp_path):
    registry, requests = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    metadatas = registry.list_datasets()
    assert [m.dataset_id for m in metadatas] == [manifest.metadata.dataset_id]
    assert str(metadatas[0].dataset_version) == str(manifest.metadata.dataset_version)
    assert all(isinstance(d, Domain) for d in metadatas[0].domains)
    assert metadatas[0].license_url is None
    assert metadatas[0].access is Access.OPEN
    assert metadatas[0].access_url is None
    assert [request.url.path for request in requests] == ["/api/v1/datasets"]


def test_list_datasets_preserves_optional_license_and_access_fields(tmp_path):
    summary = {
        "dataset_id": "demo/restricted",
        "version": "1.0.0",
        "name": "Restricted",
        "description": "A restricted dataset.",
        "license": "other",
        "license_url": "https://example.org/license",
        "domains": ["general"],
        "tags": [],
        "access": "credentialed",
        "access_url": "https://example.org/access",
    }

    def handler(request):
        assert request.url.path == "/api/v1/datasets"
        return httpx.Response(200, json={"datasets": [summary]})

    registry = RemoteRegistry(
        "http://api.local",
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path,
    )
    metadata = registry.list_datasets()[0]
    assert metadata.license is License.OTHER
    assert metadata.license_url == "https://example.org/license"
    assert metadata.access is Access.CREDENTIALED
    assert metadata.access_url == "https://example.org/access"


@pytest.mark.parametrize(
    ("manifest_updates", "summary_updates"),
    [
        (
            {"license": "other", "license_url": "https://example.org/license"},
            {"license": "other"},
        ),
        (
            {"access": "credentialed", "access_url": "https://example.org/credentials"},
            {"access": "credentialed"},
        ),
        (
            {"access": "restricted", "access_url": "https://example.org/dua"},
            {"access": "restricted"},
        ),
    ],
)
def test_list_datasets_fetches_pinned_manifest_for_incomplete_summary(
    version_dir,
    tmp_path,
    manifest_updates,
    summary_updates,
):
    _, manifest = version_dir
    manifest_payload = manifest.to_dict()
    manifest_payload["metadata"].update(manifest_updates)
    expected = Manifest.from_dict(manifest_payload).metadata
    summary = {
        "dataset_id": manifest.dataset_id,
        "version": str(manifest.metadata.dataset_version),
        "name": manifest.metadata.name,
        "description": manifest.metadata.description,
        "license": manifest.metadata.license.value,
        "domains": [domain.value for domain in manifest.metadata.domains],
        "tags": list(manifest.metadata.tags),
    }
    summary.update(summary_updates)
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/v1/datasets":
            return httpx.Response(200, json={"datasets": [summary]})
        return httpx.Response(200, json=manifest_payload)

    registry = RemoteRegistry(
        "http://api.local",
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path,
    )
    assert registry.list_datasets() == [expected]
    assert paths == [
        "/api/v1/datasets",
        f"/api/v1/datasets/{manifest.dataset_id}/{manifest.metadata.dataset_version}/manifest",
    ]


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


def test_open_full_round_trips(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    _, manifest = version_dir
    client = TimeNet(registry=registry, storage_path=tmp_path / "storage")
    source = make_dataset(n_records=2, n_values=64)
    # a remote open materializes the version, then reads the control plane locally
    with client.open(manifest.metadata.dataset_id) as reader:
        assert reader.record_ids() == sorted(record.id for record in source.records)
        assert reader.task_ids() == sorted(task.id for task in source.tasks)
        for record in source.records:
            for signal in record.signals():
                np.testing.assert_array_equal(reader.values(signal.id), signal.values)


def test_store_publishes_uploads_and_finalizes(tmp_path):
    transport, state = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", token="tok_rw", transport=transport, cache_dir=tmp_path)
    dataset = make_dataset(n_records=1, n_values=64)
    version = registry.store(dataset)
    assert version == str(dataset.metadata.dataset_version)
    assert state["finalized"] is True
    # every declared file was uploaded
    assert set(state["store"]) == set(state["published"])


def test_store_rejects_a_publish_list_that_drops_a_manifest_file(tmp_path):
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
        registry.store(make_dataset(n_records=1, n_values=64), force=True)


def test_remote_registry_uses_client_storage_path(tmp_path):
    # A URL-configured remote registry must cache under the client's storage_path, so download() and
    # open() share one cache-first directory.
    client = TimeNet(registry="https://registry.example.test", storage_path=tmp_path / "store")
    assert isinstance(client._registry, RemoteRegistry)
    assert client._registry._cache_dir == client._storage == tmp_path / "store"


def test_store_without_write_token_is_rejected(tmp_path):
    transport, _ = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path)  # anonymous
    with pytest.raises(TimeNetRegistryError):
        registry.store(make_dataset(n_records=1, n_values=64))
