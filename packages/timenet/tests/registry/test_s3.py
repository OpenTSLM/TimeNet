import numpy as np
import pytest


boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto")

from moto.server import ThreadedMotoServer  # noqa: E402
import pyarrow.fs as pafs  # noqa: E402

from timenet.client import TimeNet  # noqa: E402
from timenet.errors import TimeNetRegistryError  # noqa: E402
from timenet.registry import S3Registry  # noqa: E402
from timenet.testing import make_dataset  # noqa: E402


def _source():
    return make_dataset(n_records=1, n_values=64)


def _assert_round_trip(reader, source) -> None:
    """The control plane gave back every record, task and signal that was stored."""
    assert reader.record_ids() == sorted(record.id for record in source.records)
    assert reader.task_ids() == sorted(task.id for task in source.tasks)
    for record in source.records:
        view = reader.record(record.id)
        assert [signal.signal_id for signal in view.signals()] == [signal.id for signal in record.signals()]
        for signal in record.signals():
            np.testing.assert_array_equal(reader.values(signal.id), signal.values)


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


def test_store_and_round_trip(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    source = _source()
    assert registry.store(source) == str(source.metadata.dataset_version)
    client = TimeNet(registry=registry, storage_path=tmp_path / "cache")
    client.download(source.metadata.dataset_id)
    with client.open(source.metadata.dataset_id) as reader:
        _assert_round_trip(reader, source)


def test_an_uncached_version_opens_straight_from_s3(s3_root, tmp_path):
    # Nothing is in the cache yet, so the handle points at the objects themselves rather than a
    # downloaded copy: pyarrow serves it with range reads, no whole-version fetch.
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    source = _source()
    registry.store(source)
    handle = registry.open_version(source.metadata.dataset_id)
    assert isinstance(handle.filesystem, pafs.S3FileSystem)
    assert handle.root.startswith("tn-test/registry/datasets/")


def test_objects_live_under_a_datasets_prefix(s3_root, tmp_path):
    # Keys sit under <prefix>/datasets/... so the bucket matches the hosted registry's layout and can
    # hold other top-level prefixes beside the datasets.
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    registry.store(_source())
    keys = [obj["Key"] for obj in boto3.client("s3").list_objects_v2(Bucket="tn-test")["Contents"]]
    assert keys  # something was stored
    assert all(key.startswith("registry/datasets/") for key in keys)


def test_download_then_open_serves_from_cache(s3_root, tmp_path):
    registry = S3Registry(s3_root, cache_dir=tmp_path / "cache")
    source = _source()
    dataset_id = source.metadata.dataset_id
    registry.store(source)
    client = TimeNet(registry=registry, storage_path=tmp_path / "cache")
    path = client.download(dataset_id)  # boto3 GetObject per file, into the cache
    assert (path / "manifest.json").exists()
    handle = registry.open_version(dataset_id)
    assert isinstance(handle.filesystem, pafs.LocalFileSystem)  # cache-first: no S3 reads
    with client.open(dataset_id) as reader:
        _assert_round_trip(reader, source)


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
