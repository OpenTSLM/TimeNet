"""One contract, three backends. Every writable registry must behave the same way from a caller's seat.

The ``registry`` fixture parametrizes over ``local``, ``s3``, and ``remote``. Each yields the same two
datasets, published by calling ``store``, so a single test body runs against all three. Transport-specific
behaviour (S3 key prefixes, remote 429/auth/presign, factory dispatch) stays in the per-backend files;
this file only asserts the shared read/write contract.
"""

import dataclasses
from typing import NamedTuple
import uuid

from _fake_registry import build_service_fake
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import DatasetNotFoundError
from timenet.registry import LocalRegistry, RemoteRegistry, S3Registry, WritableRegistry
from timenet.testing import make_dataset, sine_loader
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)


def _second_dataset() -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/other",
            dataset_version=Version(1, 1, 0),
            name="Other Dataset",
            description="A second small dataset.",
            license=License.MIT,
            domains=(Domain.CARDIOLOGY,),
            tags=("clinical",),
        )
    )
    spec = TimeSeriesSpec(spec_type="ecg_lead", name="ECG Lead", unit_value=ureg.millivolt)
    series = TimeSeries(
        spec=spec,
        channel="II",
        time_axis=RegularAxis.from_rate_hz(16),
        loader=sine_loader(n=16, sampling_rate_hz=16.0),
        time_series_id="other-ts-0",
        n_values=16,
    )
    sample = dataset.add_sample(time_series=(series,), sample_id="other-sample-0")
    sample.add_annotation(Annotation(key="age", value=70, unit="years", id="other-age-0"))
    dataset.add_task(sample, ClassificationTask(target="afib", id="other-task-0"))
    return dataset


class Backend(NamedTuple):
    registry: WritableRegistry
    supports_listing: bool


def _s3_registry(request, tmp_path, monkeypatch) -> S3Registry:
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer  # noqa: PLC0415

    server = ThreadedMotoServer(port=0)
    try:
        server.start()
    except Exception as exc:
        pytest.skip(f"cannot start a local S3 endpoint: {exc}")
    request.addfinalizer(server.stop)
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
    # A unique bucket per fixture: moto's ThreadedMotoServer shares one process-global backend across
    # server instances, so a fixed name would leak these datasets into test_s3.py's own tn-test bucket.
    bucket = f"tn-conformance-{uuid.uuid4().hex[:12]}"
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket=bucket)
    return S3Registry(f"s3://{bucket}/registry", cache_dir=tmp_path / "cache")


@pytest.fixture(params=["local", "s3", "remote"])
def backend(request, tmp_path, monkeypatch) -> Backend:
    """A writable registry of the requested kind, seeded via ``store``.

    Holds timenet/hello-world at 1.0.0 and 1.1.0 (so latest-resolution has a real choice) plus a
    second dataset demo/other, so listing has more than one id to sort.
    """
    kind = request.param
    if kind == "local":
        registry: WritableRegistry = LocalRegistry(tmp_path / "reg")
        supports_listing = True
    elif kind == "remote":
        service_root = tmp_path / "service"
        service_root.mkdir()
        transport, _ = build_service_fake(service_root, token="tok_rw")
        registry = RemoteRegistry("http://svc.local", token="tok_rw", cache_dir=tmp_path / "cache", transport=transport)
        supports_listing = True
    else:
        registry = _s3_registry(request, tmp_path, monkeypatch)
        supports_listing = False
    newer = make_dataset()
    newer._metadata = dataclasses.replace(newer.metadata, dataset_version=Version(1, 1, 0))
    for dataset in (make_dataset(), newer, _second_dataset()):
        registry.store(dataset)
    return Backend(registry=registry, supports_listing=supports_listing)


# ---- get_manifest -----------------------------------------------------------------------------


@pytest.mark.parametrize("version", [None, "", "latest"])
def test_get_manifest_resolves_latest(backend, version):
    # hello-world has 1.0.0 and 1.1.0; the sentinels must resolve to the newest.
    manifest = backend.registry.get_manifest("timenet/hello-world", version)
    assert manifest.metadata.dataset_id == "timenet/hello-world"
    assert str(manifest.metadata.dataset_version) == "1.1.0"


def test_get_manifest_pins_an_explicit_version(backend):
    manifest = backend.registry.get_manifest("timenet/hello-world", "1.0.0")
    assert str(manifest.metadata.dataset_version) == "1.0.0"  # the older version is still addressable


def test_get_manifest_unknown_id_raises(backend):
    with pytest.raises(DatasetNotFoundError):
        backend.registry.get_manifest("no/such")


def test_get_manifest_unknown_version_raises(backend):
    with pytest.raises(DatasetNotFoundError):
        backend.registry.get_manifest("timenet/hello-world", "9.9.9")


# ---- open_file --------------------------------------------------------------------------------


def test_open_file_reads_a_manifest_part(backend):
    manifest = backend.registry.get_manifest("timenet/hello-world")
    relpath = manifest.files.all_parts()[0]
    with backend.registry.open_file("timenet/hello-world", "1.0.0", relpath) as handle:
        assert handle.read()  # a real, non-empty binary stream


def test_open_file_rejects_traversal(backend):
    # LocalRegistry guards up front (ValueError "escapes"); S3 and remote reject implicitly because the
    # escaped key/route does not exist (DatasetNotFoundError). Either way a caller cannot read out of tree.
    with pytest.raises((ValueError, DatasetNotFoundError)):
        backend.registry.open_file("timenet/hello-world", "1.0.0", "../../escape")


# ---- open_version -----------------------------------------------------------------------------


def test_open_version_manifest_matches(backend):
    version = backend.registry.open_version("timenet/hello-world")  # no pin -> latest
    assert version.manifest.metadata.dataset_id == "timenet/hello-world"
    assert str(version.manifest.metadata.dataset_version) == "1.1.0"


# ---- exists -----------------------------------------------------------------------------------


def test_exists_true_for_committed(backend):
    assert backend.registry.exists("timenet/hello-world", "1.0.0")


def test_exists_false_for_unknown_version(backend):
    assert not backend.registry.exists("timenet/hello-world", "9.9.9")


# ---- store ------------------------------------------------------------------------------------


def test_store_is_idempotent(backend):
    # The version is already committed by the fixture; a second store must skip, not raise.
    assert backend.registry.store(make_dataset()) == "1.0.0"


def test_store_force_republishes(backend):
    assert backend.registry.store(make_dataset(), force=True) == "1.0.0"


# ---- list_datasets (catalog-only) -------------------------------------------------------------


def test_list_datasets_returns_latest_metadata_sorted(backend):
    if not backend.supports_listing:
        pytest.skip("backend has no catalog")
    metadatas = backend.registry.list_datasets()
    assert [m.dataset_id for m in metadatas] == ["demo/other", "timenet/hello-world"]
    hello = next(m for m in metadatas if m.dataset_id == "timenet/hello-world")
    assert str(hello.dataset_version) == "1.1.0"  # latest version only, not 1.0.0
