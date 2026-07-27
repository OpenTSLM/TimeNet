"""uuid7 default ids are stored as binary(16) and round-trip back to canonical strings."""

import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal
from timenet.types import (
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    License,
    StaticAnnotation,
    TimeSeriesSpec,
    Version,
    View,
    ureg,
)
from timenet.writer import TimeFWriter


def _spec():
    return TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.dimensionless,
    )


def _series():
    return TimeSeries(
        spec=_spec(),
        channel="c",
        sampling_rate_hz=1.0,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )


def _uuid_dataset(*, sample_id=None):
    """Build a dataset whose ids default to uuid7 (unless an explicit sample_id is passed)."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    sample = dataset.add_sample(time_series=(_series(),), view=View.FULL, sample_id=sample_id)
    sample.add_annotation(StaticAnnotation(key="k", value=1))
    dataset.add_task(sample, ClassificationTask(target="x"))
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset):
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_default_ids_are_uuid7():
    dataset = _uuid_dataset()
    sid = dataset.samples[0].sample_id
    assert uuid.UUID(sid).version == 7
    assert str(uuid.UUID(sid)) == sid  # canonical


def test_uuid_ids_marked_uuid16_in_manifest(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset())
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    for logical in ("sample_id", "time_series_id", "annotation_id", "task_id"):
        assert manifest.id_encoding.get(logical) == "uuid16", logical


def test_uuid_id_columns_are_binary16_on_disk(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset())
    samples = pq.read_table(version_dir / "samples.parquet").schema
    assert samples.field("sample_id").type == pa.binary(16)
    index = pq.read_table(version_dir / "time_series_index.parquet").schema
    assert index.field("sample_id").type == pa.binary(16)
    assert index.field("time_series_id").type == pa.binary(16)
    shard = next(version_dir.glob("time_series/shard-*.parquet"))
    assert pq.read_table(shard).schema.field("time_series_id").type == pa.binary(16)
    annotations = pq.read_table(version_dir / "annotations.parquet").schema
    assert annotations.field("id").type == pa.binary(16)


def test_uuid_ids_round_trip_as_canonical_strings(tmp_path):
    dataset = _uuid_dataset()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(dataset, restored)
    sid = restored.samples[0].sample_id
    assert str(uuid.UUID(sid)) == sid
    assert uuid.UUID(sid).version == 7


def test_forecasting_scalar_id_round_trips(tmp_path):
    """target_sample_id is a scalar id column; exercise its binary(16) encode/decode."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    context = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    target = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    dataset.add_task(
        target,
        ForecastingTask(context_sample_ids=(context.sample_id,), target_sample_id=target.sample_id),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(version_dir) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.target_sample_id == target.sample_id
    assert task.context_sample_ids == (context.sample_id,)


def test_non_uuid_ids_stay_string(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset(sample_id="sample-0"))
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert "sample_id" not in manifest.id_encoding  # not all canonical UUIDs -> string
    samples = pq.read_table(version_dir / "samples.parquet").schema
    assert samples.field("sample_id").type == pa.string()
    # a sibling id space that is all-uuid still packs to binary(16)
    assert manifest.id_encoding.get("time_series_id") == "uuid16"
