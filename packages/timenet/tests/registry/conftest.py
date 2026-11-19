from collections.abc import Iterator
from pathlib import Path
import uuid

import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.testing import make_dataset, sine_loader
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    LabelSet,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


RHYTHM = LabelSet(["afib", "normal"])


def _ecg_dataset() -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/ecg",
            dataset_version=Version(2, 0, 0),
            name="ECG Dataset",
            description="A clinical ECG dataset.",
            license=License.MIT,
            domains=(Domain.CARDIOLOGY,),
            tags=("clinical",),
        )
    )
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )
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


@pytest.fixture
def registry_root(tmp_path) -> Path:
    """A local registry directory holding two datasets: timenet/hello-world and demo/ecg."""
    for dataset in (make_dataset(), _ecg_dataset()):
        dataset.derive_schema()
        with TimeFWriter(tmp_path, dataset) as writer:
            writer.write()
    return tmp_path


@pytest.fixture(scope="session")
def _moto_server() -> Iterator[str]:
    """Start one local moto S3 endpoint for the whole session and yield its URL.

    Stopping a ``ThreadedMotoServer`` blocks for about half a second, so one server per test costs
    seconds across the suite. Sharing the server changes no state: moto keeps one process-global
    backend however many servers run. Each test keeps its objects apart with ``s3_bucket``.
    """
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer  # ruff: ignore[import-outside-top-level]

    server = ThreadedMotoServer(port=0)
    try:
        server.start()
    except Exception as exc:
        pytest.skip(f"cannot start a local S3 endpoint: {exc}")
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3_endpoint(_moto_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point boto3 and pyarrow at the shared moto endpoint and return its URL."""
    for key, value in {
        "AWS_ENDPOINT_URL": _moto_server,
        "AWS_ENDPOINT_URL_S3": _moto_server,
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(key, value)
    return _moto_server


@pytest.fixture
def s3_bucket(s3_endpoint: str) -> str:
    """Create an empty bucket with a unique name on the shared moto endpoint and return its name.

    The shared server keeps every object until the session ends, so a fixed bucket name would let
    one test read what an earlier test wrote.
    """
    boto3 = pytest.importorskip("boto3")
    bucket = f"tn-test-{uuid.uuid4().hex[:12]}"
    boto3.client("s3", endpoint_url=s3_endpoint).create_bucket(Bucket=bucket)
    return bucket
