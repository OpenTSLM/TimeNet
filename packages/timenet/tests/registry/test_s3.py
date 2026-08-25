import pytest


boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto")

from moto.server import ThreadedMotoServer  # noqa: E402
import pyarrow.fs as pafs  # noqa: E402

from timenet.client import TimeNet  # noqa: E402
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetRegistryError  # noqa: E402
from timenet.registry import S3Registry  # noqa: E402
from timenet.testing import assert_datasets_equal, make_dataset  # noqa: E402


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


def test_store_and_lazy_load_round_trip(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    version = registry.store(make_dataset())
    assert version == str(make_dataset().metadata.dataset_version)
    client = TimeNet(registry=registry, storage_path=tmp_path / "cache")
    loaded = client.load(make_dataset().metadata.dataset_id)  # lazy S3 range reads
    assert_datasets_equal(make_dataset(), loaded)


def test_store_skips_committed_unless_forced(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    version = str(make_dataset().metadata.dataset_version)
    registry.store(make_dataset())
    assert registry.exists(make_dataset().metadata.dataset_id, version)
    assert registry.store(make_dataset()) == version  # second store is a no-op


def test_get_manifest_latest_resolves(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    registry.store(make_dataset())
    manifest = registry.get_manifest(make_dataset().metadata.dataset_id)
    assert manifest.metadata.dataset_id == make_dataset().metadata.dataset_id


def test_download_then_load_serves_from_cache(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    dataset_id = make_dataset().metadata.dataset_id
    registry.store(make_dataset())
    client = TimeNet(registry=registry, storage_path=tmp_path / "cache")
    path = client.download(dataset_id)  # boto3 GetObject per file, into the cache
    assert (path / "manifest.json").exists()
    handle = registry.open_version(dataset_id)
    assert isinstance(handle.filesystem, pafs.LocalFileSystem)  # cache-first: no S3 reads
    assert_datasets_equal(make_dataset(), client.load(dataset_id))


def test_missing_dataset_raises(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    with pytest.raises(TimeNetDatasetNotFoundError):
        registry.get_manifest("no/such", "1.0.0")


def test_get_bytes_reraises_non_404_client_errors(tmp_path, monkeypatch):
    # A missing object returns None, but any other S3 error (auth, throttling) must surface, not hide.
    import botocore.exceptions  # noqa: PLC0415

    registry = S3Registry("s3://tn-test/registry", cache_dir=tmp_path)
    denied = botocore.exceptions.ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")

    class _Denying:
        def get_object(self, **kwargs):
            raise denied

    monkeypatch.setattr(registry, "_client", _Denying)
    with pytest.raises(TimeNetRegistryError, match="cannot read s3://"):
        registry._get_bytes("datasets/o/n/1.0.0/manifest.json")


def test_list_and_search_are_unsupported(tmp_path):
    registry = S3Registry("s3://tn-test/registry", cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        registry.list_datasets()
    with pytest.raises(NotImplementedError):
        registry.search()


def test_rejects_a_non_s3_uri(tmp_path):
    with pytest.raises(TimeNetRegistryError, match="s3://"):
        S3Registry("https://not-s3", cache_dir=tmp_path)
