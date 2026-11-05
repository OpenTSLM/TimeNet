"""One contract, three backends. Every writable registry must behave the same way from a caller's seat.

The ``registry`` fixture parametrizes over ``local``, ``s3``, and ``remote``. Each yields the same two
datasets, published by calling ``put_dataset``, so a single test body runs against all three. Transport-specific
behaviour (S3 key prefixes, remote 429/auth/presign, factory dispatch) stays in the per-backend files;
this file only asserts the shared read/write contract.
"""

from typing import NamedTuple
import uuid

from _demo_child import _child_of_the_demo
from _fake_registry import build_service_fake
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetFileError
from timenet.registry import LocalRegistry, RemoteRegistry, S3Registry, WritableRegistry
from timenet.registry._paths import release_dir
from timenet.sources import Found, NotFound
from timenet.testing import make_dataset, sine_loader
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    Label,
    LabelSet,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.types.references import DatasetRef


RHYTHM = LabelSet(
    name="rhythm",
    labels=(
        Label(name="afib"),
        Label(name="normal"),
    ),
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
    series = Signal.from_loader(
        spec=spec,
        name="II",
        time_axis=RegularAxis.from_rate_hz(16),
        loader=sine_loader(n=16, sampling_rate_hz=16.0),
        n_values=16,
    )
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(series,)),)))
    record.annotate(Annotation(key="age", value=70, unit="years"))
    dataset.add_task(task=ClassificationTask(label_set=RHYTHM, inputs=(record,), targets=("afib",)))
    return dataset


class Backend(NamedTuple):
    registry: WritableRegistry
    supports_listing: bool


def _s3_registry(request, tmp_path, monkeypatch) -> S3Registry:
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer  # ruff: ignore[import-outside-top-level]

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
    return S3Registry(f"s3://{bucket}/registry", store=tmp_path / "cache")


@pytest.fixture(params=["local", "s3", "remote"])
def backend(request, tmp_path, monkeypatch) -> Backend:
    """A writable registry of the requested kind, seeded via ``put_dataset``.

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
        registry = RemoteRegistry("http://svc.local", token="tok_rw", store=tmp_path / "cache", transport=transport)
        supports_listing = True
    else:
        registry = _s3_registry(request, tmp_path, monkeypatch)
        supports_listing = False
    newer = make_dataset()
    newer._metadata = newer.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)})
    for dataset in (make_dataset(), newer, _second_dataset()):
        registry.put_dataset(dataset)
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
    with pytest.raises(TimeNetDatasetNotFoundError):
        backend.registry.get_manifest("no/such")


def test_get_manifest_unknown_version_raises(backend):
    with pytest.raises(TimeNetDatasetNotFoundError):
        backend.registry.get_manifest("timenet/hello-world", "9.9.9")


# ---- resolve ----------------------------------------------------------------------------------


def test_a_registry_resolves_a_held_release_and_not_an_unknown_one(backend):
    registry = backend.registry
    assert registry.resolve("timenet/hello-world", None) == Found(Version(1, 1, 0), registry)
    assert registry.resolve("timenet/hello-world", "1.0.0") == Found(Version(1, 0, 0), registry)
    assert registry.resolve("no/such", None) == NotFound()
    assert registry.resolve("no/such", "1.0.0") == NotFound()
    assert registry.resolve("timenet/hello-world", "9.9.9") == NotFound()


# ---- open_file --------------------------------------------------------------------------------


def test_open_file_reads_a_manifest_part(backend):
    manifest = backend.registry.get_manifest("timenet/hello-world")
    relpath = manifest.files.all_parts()[0]
    with backend.registry.open_file("timenet/hello-world", "1.0.0", relpath) as handle:
        assert handle.read()  # a real, non-empty binary stream


def test_open_file_rejects_traversal(backend):
    # LocalRegistry guards up front (ValueError "escapes"); S3 and remote reject implicitly because the
    # escaped key/route does not exist (TimeNetDatasetNotFoundError). Either way a caller cannot read out of tree.
    with pytest.raises((ValueError, TimeNetDatasetNotFoundError)):
        backend.registry.open_file("timenet/hello-world", "1.0.0", "../../escape")


# ---- open_release -----------------------------------------------------------------------------


def test_open_release_opens_one_release_without_its_payload_sources(backend, tmp_path):
    source, _ = _child_of_the_demo(tmp_path / "source")
    child = DatasetRef.parse("test/child@1.0.0")
    manifest = source.get_manifest(child.dataset_id)
    backend.registry.put_release(release_dir(source.root, child))
    version = backend.registry.open_release(child)
    # The child reads files of the demo release, but the handle opens none of them.
    assert manifest.payload_sources
    assert version.payload_sources == {}
    assert version.manifest == backend.registry.get_manifest(child.dataset_id, str(child.version))


def test_open_release_uses_the_manifest_that_the_caller_passes(backend, tmp_path, monkeypatch):
    source, _ = _child_of_the_demo(tmp_path / "source")
    child = DatasetRef.parse("test/child@1.0.0")
    manifest = source.get_manifest(child.dataset_id)
    backend.registry.put_release(release_dir(source.root, child))

    def no_read(*args, **kwargs):
        raise AssertionError("the registry reads the manifest that the caller passed")

    monkeypatch.setattr(backend.registry, "get_manifest", no_read)
    # No cache holds a copy of the child, so S3 and the remote registry open their remote files.
    version = backend.registry.open_release(child, manifest=manifest)
    assert version.manifest == manifest
    assert version.payload_sources == {}


# ---- exists -----------------------------------------------------------------------------------


def test_exists_true_for_committed(backend):
    assert backend.registry.exists("timenet/hello-world", "1.0.0")


def test_exists_false_for_unknown_version(backend):
    assert not backend.registry.exists("timenet/hello-world", "9.9.9")


# ---- put_dataset ------------------------------------------------------------------------------


def test_put_dataset_is_idempotent(backend):
    # The version is already committed by the fixture; a second put_dataset() call must skip, not raise.
    assert backend.registry.put_dataset(make_dataset()) == DatasetRef.parse("timenet/hello-world@1.0.0")


def _local_release(tmp_path):
    """Build a complete local release of timenet/hello-world 2.0.0; return its directory and manifest."""
    source = LocalRegistry(tmp_path / "src")
    dataset = make_dataset()
    dataset._metadata = dataset.metadata.model_copy(update={"dataset_version": Version(2, 0, 0)})
    source.put_dataset(dataset)
    return source.root / "timenet/hello-world/2.0.0", source.get_manifest("timenet/hello-world", "2.0.0")


def test_put_release_publishes_a_complete_release(backend, tmp_path):
    directory, manifest = _local_release(tmp_path)
    assert backend.registry.put_release(directory) == DatasetRef.parse("timenet/hello-world@2.0.0")
    assert backend.registry.get_manifest("timenet/hello-world", "2.0.0") == manifest


def test_put_release_publishes_only_the_declared_files(backend, tmp_path):
    directory, manifest = _local_release(tmp_path)
    (directory / "stray.txt").write_text("not part of the release")
    (directory / "time_series" / "stray.parquet").write_text("not part of the release")
    backend.registry.put_release(directory)
    assert backend.registry.get_manifest("timenet/hello-world", "2.0.0") == manifest
    for relpath in ("stray.txt", "time_series/stray.parquet"):
        with pytest.raises((FileNotFoundError, TimeNetDatasetNotFoundError)):
            backend.registry.open_file("timenet/hello-world", "2.0.0", relpath).close()


def test_put_release_refuses_an_incomplete_directory(backend, tmp_path):
    directory, manifest = _local_release(tmp_path)
    declared = sorted(part.path for part in manifest.files.all_files())
    (directory / declared[0]).unlink()
    with pytest.raises(TimeNetFileError):
        backend.registry.put_release(directory)
    assert not backend.registry.exists("timenet/hello-world", "2.0.0")


# ---- list_datasets (catalog-only) -------------------------------------------------------------


def test_list_datasets_returns_latest_metadata_sorted(backend):
    if not backend.supports_listing:
        pytest.skip("backend has no catalog")
    metadatas = backend.registry.list_datasets()
    assert [m.dataset_id for m in metadatas] == ["demo/other", "timenet/hello-world"]
    hello = next(m for m in metadatas if m.dataset_id == "timenet/hello-world")
    assert str(hello.dataset_version) == "1.1.0"  # latest version only, not 1.0.0
