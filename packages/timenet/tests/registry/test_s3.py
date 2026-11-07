import pytest


boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto")

import botocore.exceptions  # ruff: ignore[module-import-not-at-top-of-file]
from moto.server import ThreadedMotoServer  # ruff: ignore[module-import-not-at-top-of-file]
import pyarrow.fs as pafs  # ruff: ignore[module-import-not-at-top-of-file]

from timenet.client import TimeNet  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.dataset import TimeFDataset  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.errors import (  # ruff: ignore[module-import-not-at-top-of-file]
    TimeNetDatasetNotFoundError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.registry import LocalRegistry, S3Registry, open_local_version  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.testing import assert_datasets_equal, make_dataset  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.types import DatasetSchema, Version  # ruff: ignore[module-import-not-at-top-of-file]
from timenet.types.references import DatasetRef  # ruff: ignore[module-import-not-at-top-of-file]


@pytest.fixture
def s3_root(monkeypatch):
    """Start a local moto S3 endpoint, point boto3 + pyarrow at it, and yield the registry URI."""
    server = ThreadedMotoServer(port=0)
    try:
        server.start()
    except Exception as exc:
        pytest.skip(f"cannot start a local S3 endpoint: {exc}")
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    for key, value in {
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_ENDPOINT_URL_S3": endpoint,
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(key, value)
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket="tn-test")
    yield "s3://tn-test/registry"
    server.stop()


@pytest.mark.parametrize("entry_point", ["open", "load"])
@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_put_dataset_and_remote_load_round_trip(s3_root, tmp_path, backend, entry_point):
    registry = S3Registry(s3_root, local_registry=tmp_path / "cache")
    reference = registry.put_dataset(make_dataset(), values_backend=backend)
    assert reference == DatasetRef.parse(f"{make_dataset().metadata.dataset_id}@1.0.0")
    client = TimeNet(sources=[registry])
    if entry_point == "open":
        with client.open(make_dataset().metadata.dataset_id) as reader:
            # The reader reads the release from S3, not from a downloaded copy.
            assert isinstance(reader._version.filesystem, pafs.S3FileSystem)
            assert_datasets_equal(make_dataset(), reader.read())
    else:
        assert_datasets_equal(make_dataset(), client.load(make_dataset().metadata.dataset_id))
    assert isinstance(
        registry.open_release(DatasetRef.parse("timenet/hello-world@1.0.0")).filesystem, pafs.S3FileSystem
    )
    assert not (tmp_path / "cache" / make_dataset().metadata.dataset_id).exists()


def test_put_dataset_checks_payload_sources_against_the_bucket(s3_root, tmp_path):
    local = LocalRegistry(tmp_path / "local")
    local.put_dataset(make_dataset())
    # moto keeps one bucket state per process, so this test uses a prefix no other test writes.
    registry = S3Registry("s3://tn-test/payload-sources", local_registry=tmp_path / "cache")
    with TimeFReader(open_local_version(local.root / "timenet/hello-world/1.0.0")) as reader:
        parent = reader.read()
        child = TimeFDataset.from_parts(
            metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}),
            records=parent.records,
            tasks=(),
            schema=DatasetSchema(),
            last_ids=parent.last_ids,
        )
        with pytest.raises(
            TimeNetDatasetNotFoundError,
            match=r"of test/child@1\.0\.0 is not in S3 registry s3://tn-test/payload-sources",
        ):
            registry.put_dataset(child)
    assert not registry.exists("test/child", "1.0.0")
    keys = [obj["Key"] for obj in boto3.client("s3").list_objects_v2(Bucket="tn-test").get("Contents", [])]
    assert not [key for key in keys if "test/child" in key]


def test_objects_live_under_a_datasets_prefix(s3_root, tmp_path):
    # Keys sit under <prefix>/datasets/... so the bucket matches the hosted registry's layout and can
    # hold other top-level prefixes beside the datasets.
    registry = S3Registry(s3_root, local_registry=tmp_path / "cache")
    registry.put_dataset(make_dataset())
    keys = [obj["Key"] for obj in boto3.client("s3").list_objects_v2(Bucket="tn-test")["Contents"]]
    assert keys  # something was stored
    assert all(key.startswith("registry/datasets/") for key in keys)


def test_download_then_load_serves_from_cache(s3_root, tmp_path):
    registry = S3Registry(s3_root, local_registry=tmp_path / "cache")
    dataset_id = make_dataset().metadata.dataset_id
    registry.put_dataset(make_dataset())
    # The store is the cache directory of the registry, so the registry reuses each download.
    client = TimeNet(sources=[LocalRegistry(tmp_path / "cache", store=True), registry])
    path = client.download(dataset_id)  # boto3 GetObject per file, into the store
    assert path == tmp_path / "cache" / dataset_id / "1.0.0"
    assert (path / "manifest.json").exists()
    handle = registry.open_release(DatasetRef.parse(f"{dataset_id}@1.0.0"))
    assert isinstance(handle.filesystem, pafs.LocalFileSystem)  # the files come from the store, not S3
    assert_datasets_equal(make_dataset(), client.load(dataset_id))


def test_get_bytes_reraises_non_404_client_errors(tmp_path, monkeypatch):
    # A missing object returns None, but any other S3 error (auth, throttling) must surface, not hide.
    import botocore.exceptions  # ruff: ignore[import-outside-top-level]

    registry = S3Registry("s3://tn-test/registry", local_registry=tmp_path)
    denied = botocore.exceptions.ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")

    class _Denying:
        def get_object(self, **kwargs):
            raise denied

    monkeypatch.setattr(registry, "_client", _Denying)
    with pytest.raises(TimeNetRegistryError, match="cannot read s3://"):
        registry._get_bytes("datasets/o/n/1.0.0/manifest.json")


def test_list_and_search_are_unsupported(tmp_path):
    registry = S3Registry("s3://tn-test/registry", local_registry=tmp_path)
    with pytest.raises(NotImplementedError):
        registry.list_datasets()
    with pytest.raises(NotImplementedError):
        registry.search()


def test_rejects_a_non_s3_uri(tmp_path):
    with pytest.raises(TimeNetRegistryError, match="s3://"):
        S3Registry("https://not-s3", local_registry=tmp_path)


class _Unreachable:
    """An S3 client whose every request fails to reach the endpoint."""

    def __getattr__(self, name):
        def request(*args, **kwargs):
            raise botocore.exceptions.EndpointConnectionError(endpoint_url="https://s3.example")

        return request


def test_an_unreachable_bucket_raises_a_registry_error_that_names_it(monkeypatch):
    registry = S3Registry("s3://tn-test/registry")
    monkeypatch.setattr(registry, "_client", _Unreachable)
    with pytest.raises(TimeNetRegistryError, match=r"^cannot reach s3://tn-test/registry: Could not connect"):
        registry.resolve("timenet/hello-world", Version(1, 0, 0))
