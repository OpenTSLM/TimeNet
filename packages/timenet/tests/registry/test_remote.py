import json
from pathlib import Path
import pickle

from _fake_registry import build_fake, build_publish_fake, range_server, read_file, registry_server
import httpx
import numpy as np
from pydantic import ValidationError
import pytest

from timenet.client import TimeNet
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetRegistryError
from timenet.manifest import Manifest
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry, open_local_version
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import Access, DatasetSchema, Domain, License
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import InputAlias
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


@pytest.mark.parametrize("expire_urls", [False, True])
def test_load_full_round_trips_after_pickling_and_url_expiry(version_dir, tmp_path, expire_urls):
    directory, manifest = version_dir
    used_urls: set[str] = set()

    def single_use_download(request):
        url = str(request.url)
        if url in used_urls:
            return httpx.Response(403, text="expired signed URL")
        used_urls.add(url)
        return read_file(directory, request)

    with registry_server(directory, serve_file=single_use_download if expire_urls else None) as server:
        registry = RemoteRegistry(server.url, cache_dir=tmp_path / "cache")
        client = TimeNet(registry=registry, storage_path=tmp_path / "storage")
        with client.open(manifest.metadata.dataset_id) as reader:
            # A worker must reopen its own clients, including after the main process has read values.
            reader.get(Signal, make_dataset().records[0].signals[0].id).to_arrow()
            with pickle.loads(pickle.dumps(reader)) as restored:
                loaded = restored.read(load=LoadPlan.ALL_EAGER)
    assert_datasets_equal(make_dataset(), loaded)


@pytest.mark.parametrize("support_ranges", [True, False])
def test_small_signal_read_fetches_ranges_and_keeps_registry_token_private(tmp_path, support_ranges):
    values = np.random.default_rng(7).normal(size=2**21).astype("float32")
    template = make_dataset()
    signal = Signal.from_values(
        values,
        spec=template.records[0].signals[0].spec,
        name="large",
        time_axis=RegularAxis.from_rate_hz(100),
    )
    dataset = TimeFDataset(metadata=template.metadata)
    dataset.add_record(record=Record(sources=(Source(name="Monitor", signals=(signal,)),)))
    with TimeFWriter(
        tmp_path / "origin", dataset, chunk_max_bytes=2**18, row_group_target_bytes=2**18, compression="none"
    ) as writer:
        writer.write()
    directory = tmp_path / "origin" / dataset.metadata.dataset_id / "1.0.0"

    def serve_file(request):
        if support_ranges:
            return read_file(directory, request)
        return httpx.Response(200, content=(directory / request.url.path.lstrip("/")).read_bytes())

    with registry_server(directory, serve_file=serve_file) as server:
        remote = RemoteRegistry(server.url, token="private-token", cache_dir=tmp_path / "cache")
        with TimeNet(remote).open(dataset.metadata.dataset_id, "1.0.0") as reader:
            assert all(response.request.url.path.startswith("/api/v1/") for response in server.responses)
            if not support_ranges:
                # The get reads the catalog, which already needs range requests.
                with pytest.raises(TimeNetRegistryError, match="range requests"):
                    reader.get(Signal, signal.id).read_steps(100, 120)
                return
            selected = reader.get(Signal, signal.id)
            np.testing.assert_array_equal(selected.read_steps(100, 120).to_numpy(), values[100:120])
            downloads = [
                response for response in server.responses if not response.request.url.path.startswith("/api/v1/")
            ]
            payloads = [response for response in downloads if response.request.url.path.endswith(".parquet")]
            assert payloads
            assert sum(len(response.content) for response in payloads) < values.nbytes // 4
            assert all("Range" in response.request.headers for response in payloads)
            assert all("Authorization" not in response.request.headers for response in downloads)
            transferred = len(server.responses)
            np.testing.assert_array_equal(selected.read_steps(100, 120).to_numpy(), values[100:120])
            assert len(server.responses) == transferred


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


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_child_streams_physical_sources_and_downloads_for_offline_reads(tmp_path, backend):
    origin = tmp_path / "origin"
    local = LocalRegistry(origin)
    parent = make_dataset()
    build_only = parent.metadata.model_copy(update={"dataset_id": "test/build-only"})
    parent.parent_cards[InputAlias("build-only")] = ParentDatasetCard(metadata=build_only)
    local.store(parent, values_backend=backend)
    parent_id = parent.metadata.dataset_id
    with TimeFReader(local.open_version(parent_id, "1.0.0")) as reader:
        reference = DatasetRef(dataset_id=parent_id, version=parent.metadata.dataset_version)
        child = TimeFDataset(
            metadata=parent.metadata.model_copy(
                update={"dataset_id": "test/child", "inputs": {InputAlias("parent"): reference}}
            )
        )
        imported = child.import_from(reader, alias=InputAlias("parent"), records=[parent.records[0].id])
        expected = parent.records[0].signals[0].to_arrow()
        signal_id = imported.records[parent.records[0].id].signals[0].id
        local.store(child)
    requests = []

    def handler(request):
        path = request.url.path
        requests.append(path)
        if request.url.host == "127.0.0.1":
            return httpx.Response(200, content=(origin / path.lstrip("/")).read_bytes())
        relative = path.removeprefix("/api/v1/datasets/")
        if relative.endswith("/manifest"):
            return httpx.Response(
                200, content=(origin / relative.removesuffix("manifest") / "manifest.json").read_bytes()
            )
        if "/download/" in relative:
            return httpx.Response(307, headers={"location": blob_url + "/" + relative.replace("/download/", "/")})
        return httpx.Response(404)

    cache = tmp_path / "cache"
    remote = RemoteRegistry("http://api.local", transport=httpx.MockTransport(handler), cache_dir=cache)
    with range_server(origin) as blob_url:
        with TimeNet(remote).open("test/child", "1.0.0") as reader:
            assert all("/download/" not in path for path in requests)
            assert reader.get(Signal, signal_id).to_arrow().equals(expected)
            assert any(parent_id + "/1.0.0/download/" in path for path in requests)
            assert not any(parent_id + "/1.0.0/download/control.duckdb" in path for path in requests)
            assert not (cache / parent_id / "1.0.0").exists()
        remote.download_version("test/child", "1.0.0", cache / "test/child/1.0.0")
    with TimeNet(remote).open("test/child", "1.0.0", build_parents=True) as reader:
        assert reader.get(Signal, signal_id).to_arrow().equals(expected)
    assert not any("test/build-only" in path for path in requests)
    assert (cache / parent_id / "1.0.0/control.duckdb").exists()
    with TimeFReader(open_local_version(cache / "test/child/1.0.0")) as reader:
        reader.verify()
        assert reader.get(Signal, signal_id).to_arrow().equals(expected)
    # The parent is already complete: preparing it again requires no network requests.
    prepared_requests = list(requests)
    remote.download_version(parent_id, "1.0.0", cache / parent_id / "1.0.0")
    assert (cache / parent_id / "1.0.0/control.duckdb").is_file()
    assert requests == prepared_requests
