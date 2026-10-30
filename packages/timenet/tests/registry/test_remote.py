import json
from pathlib import Path

from _fake_registry import build_fake, build_publish_fake, registry_server
import httpx
from pydantic import ValidationError
import pytest

from timenet.client import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError, TimeNetFileError, TimeNetRegistryError
from timenet.format.checksums import file_checksum
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Access, DatasetSchema, Domain, License
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


@pytest.fixture
def version_dir(tmp_path) -> tuple[Path, Manifest]:
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    manifest = Manifest.model_validate_json(next(tmp_path.rglob("manifest.json")).read_text())
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


def test_list_datasets_rejects_malformed_payload_with_field_path(tmp_path):
    def handler(request):
        assert request.url.path == "/api/v1/datasets"
        return httpx.Response(200, json={"datasets": [{"dataset_id": 3}]})

    registry = RemoteRegistry("http://api.local", transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with pytest.raises(ValidationError, match=r"datasets\.0\.dataset_id"):
        registry.list_datasets()


def test_list_datasets_exposes_pydantic_json_errors(tmp_path):
    registry = RemoteRegistry(
        "http://api.local",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"not json")),
        cache_dir=tmp_path,
    )
    with pytest.raises(ValidationError, match="Invalid JSON"):
        registry.list_datasets()


def test_list_datasets_preserves_optional_license_and_access_fields(tmp_path):
    summary = {
        "dataset_id": "demo/restricted",
        "version": "1.0.0",
        "name": "Restricted",
        "description": "A restricted dataset.",
        "license": "OTHER",
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
            {"license": "OTHER", "license_url": "https://example.org/license"},
            {"license": "OTHER"},
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
    expected = Manifest.model_validate(manifest_payload).metadata
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


def test_a_corrupted_remote_download_names_the_file_and_its_checksums(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    directory, manifest = version_dir
    part = manifest.files.control[0]
    original = (directory / part.path).read_bytes()
    # The same size, so only the checksum shows the change.
    (directory / part.path).write_bytes(bytes([original[0] ^ 1]) + original[1:])
    release = DatasetRef(dataset_id=manifest.metadata.dataset_id, version=manifest.metadata.dataset_version)
    dest = tmp_path / "out" / release.dataset_id / str(release.version)

    with pytest.raises(TimeNetFileError) as caught:
        registry.download_version(release.dataset_id, str(release.version), dest)

    error = caught.value
    # A caller that handles format errors still catches it.
    assert isinstance(error, TimeFFormatError)
    assert (error.release, error.path, error.expected) == (release, part.path, part.checksum)
    assert error.actual == file_checksum(directory / part.path)
    assert not dest.exists()


def test_a_remote_download_whose_size_differs_from_its_record_names_both_sizes(version_dir, tmp_path):
    registry, _ = _remote(version_dir, tmp_path)
    directory, manifest = version_dir
    payload = json.loads((directory / "manifest.json").read_text())
    record = payload["files"]["control"][0]
    # The manifest records one byte more. The checksum still matches, so only the size check sees it.
    record["size"] += 1
    (directory / "manifest.json").write_text(json.dumps(payload))
    release = DatasetRef(dataset_id=manifest.metadata.dataset_id, version=manifest.metadata.dataset_version)
    dest = tmp_path / "out" / release.dataset_id / str(release.version)

    with pytest.raises(TimeNetFileError) as caught:
        registry.download_version(release.dataset_id, str(release.version), dest)

    error = caught.value
    size = record["size"]
    assert (error.release, error.owner, error.path) == (release, f"download of {release}", record["path"])
    assert (error.expected, error.actual) == (f"{size:,} bytes", f"{size - 1:,} bytes")
    assert str(error) == (
        f"download of {release}: file '{record['path']}' has {size - 1:,} bytes, expected {size:,} bytes."
    )
    assert not dest.exists()


@pytest.mark.parametrize("entry_point", ["open", "load"])
def test_load_full_round_trips(version_dir, tmp_path, entry_point):
    directory, manifest = version_dir
    with registry_server(directory) as server:
        registry = RemoteRegistry(server.url, cache_dir=tmp_path / "cache")
        client = TimeNet(registry=registry, storage_path=tmp_path / "storage")
        if entry_point == "open":
            with client.open(manifest.metadata.dataset_id) as reader:
                assert_datasets_equal(make_dataset(), reader.read())
        else:
            loaded = client.load(manifest.metadata.dataset_id)
    if entry_point == "load":
        # Every value was loaded, so the dataset reads after the registry is gone.
        assert_datasets_equal(make_dataset(), loaded)


def test_store_publishes_uploads_and_finalizes(tmp_path):
    transport, state = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", token="tok_rw", transport=transport, cache_dir=tmp_path)
    dataset = make_dataset()
    version = registry.store(dataset)
    assert version == str(dataset.metadata.dataset_version)
    assert state["finalized"] is True
    # every declared file was uploaded
    assert set(state["store"]) == set(state["published"])


def test_store_refuses_a_payload_source_the_service_does_not_hold_before_publishing(tmp_path):
    local = LocalRegistry(tmp_path / "local")
    local.store(make_dataset())
    transport, state = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", token="tok_rw", transport=transport, cache_dir=tmp_path / "cache")
    with TimeFReader(local.open_version(make_dataset().metadata.dataset_id)) as reader:
        parent = reader.read()
        child = TimeFDataset.from_parts(
            metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}),
            records=parent.records,
            tasks=(),
            schema=DatasetSchema(),
            last_ids=parent.last_ids,
        )
        with pytest.raises(
            TimeNetDatasetNotFoundError, match=r"of test/child@1\.0\.0 is not in registry http://api\.local"
        ):
            registry.store(child)
    assert state["manifest"] is None
    assert not state["store"]


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


def test_store_rejects_unsafe_publish_paths(tmp_path):
    def handler(request):
        if request.url.path.endswith("/publish"):
            return httpx.Response(200, json={"files": ["../escape"]})
        return httpx.Response(200, json={})

    registry = RemoteRegistry(
        "http://api.local", token="tok_rw", transport=httpx.MockTransport(handler), cache_dir=tmp_path
    )
    with pytest.raises(ValidationError, match="stay within the dataset root"):
        registry.store(make_dataset(), force=True)


def test_store_rejects_malformed_upload_grant(tmp_path):
    def handler(request):
        if request.url.path.endswith("/publish"):
            manifest = json.loads(request.content)
            files = [part["path"] for group in manifest["files"].values() for part in group]
            return httpx.Response(200, json={"files": files})
        if request.url.path.endswith("/publish/upload-url"):
            return httpx.Response(200, json={"url": "not-a-url", "headers": []})
        return httpx.Response(200, json={})

    registry = RemoteRegistry(
        "http://api.local", token="tok_rw", transport=httpx.MockTransport(handler), cache_dir=tmp_path
    )
    with pytest.raises(ValidationError, match="headers"):
        registry.store(make_dataset(), force=True)


def test_latest_detail_rejects_non_string_version(tmp_path):
    def handler(request):
        assert request.url.path == "/api/v1/datasets/demo/example"
        return httpx.Response(200, json={"version": 3})

    registry = RemoteRegistry("http://api.local", transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with pytest.raises(ValidationError, match="version"):
        registry.get_manifest("demo/example")


def test_remote_registry_uses_client_storage_path(tmp_path):
    # A URL-configured remote registry must cache under the client's storage_path, so download() and
    # load() share one cache-first directory.
    client = TimeNet(registry="https://registry.example.test", storage_path=tmp_path / "store")
    assert isinstance(client._registry, RemoteRegistry)
    assert client._registry._cache_dir == client._storage == tmp_path / "store"


def test_store_without_write_token_is_rejected(tmp_path):
    transport, _ = build_publish_fake(token="tok_rw")
    registry = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path)  # anonymous
    with pytest.raises(TimeNetRegistryError):
        registry.store(make_dataset())
