"""uuid7 default ids are stored as binary(16) and round-trip back to canonical strings."""

import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    IntervalSpan,
    License,
    PointSpan,
    TemporalLocalizationTask,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
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
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=3,
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
    sample.add_annotation(Annotation(key="k", value=1))
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


def test_span_series_ids_round_trip_as_binary16(tmp_path):
    """A span nests time_series ids inside a struct column; they encode like any other id column."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    series = _series()
    sample = dataset.add_sample(time_series=(series,), view=View.FULL)
    scope = IntervalSpan.seconds(0.0, 1.0, time_series_ids=(series.time_series_id,))
    dataset.add_task(sample, ClassificationTask(target="x"), scope=scope)
    dataset.add_task(
        sample,
        TemporalLocalizationTask(
            prompt="Locate the onsets.",
            target=(PointSpan.seconds(1.0, time_series_ids=(series.time_series_id,)),),
        ),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)

    partition = version_dir / "tasks/task=classification/part-0.parquet"
    scope_type = pq.read_table(partition).schema.field("scope").type
    assert scope_type.field("time_series_ids").type == pa.list_(pa.binary(16))

    with TimeFReader(version_dir) as reader:
        tasks = {type(t): t for t in reader.tasks}
    assert tasks[ClassificationTask].scope == scope
    localization = tasks[TemporalLocalizationTask]
    assert isinstance(localization, TemporalLocalizationTask)
    assert localization.target == (PointSpan.seconds(1.0, time_series_ids=(series.time_series_id,)),)


def test_correspondence_target_ids_round_trip(tmp_path):
    """The correspondence answer is itself a tuple of sample ids, so it encodes as an id column."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    query = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    match = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    other = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    dataset.add_task(
        query,
        TSCorrespondenceTask(
            prompt="Which trace is most similar?",
            candidate_sample_ids=(match.sample_id, other.sample_id),
            target=(match.sample_id,),
        ),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(version_dir) as reader:
        task = reader.tasks[0]
    assert isinstance(task, TSCorrespondenceTask)
    assert task.target == (match.sample_id,)
    assert task.candidate_sample_ids == (match.sample_id, other.sample_id)


def test_editing_and_generation_sample_ids_round_trip(tmp_path):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    source = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    edited = dataset.add_sample(time_series=(_series(),), view=View.FULL)
    dataset.add_task(
        source,
        TSEditingTask(
            prompt="Remove the baseline wander.",
            source_sample_id=source.sample_id,
            target_sample_id=edited.sample_id,
        ),
    )
    dataset.add_task(edited, TSGenerationTask(prompt="10 s of sinus rhythm.", target_sample_id=edited.sample_id))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(version_dir) as reader:
        tasks = {type(t): t for t in reader.tasks}
    edit = tasks[TSEditingTask]
    assert isinstance(edit, TSEditingTask)
    assert (edit.source_sample_id, edit.target_sample_id) == (source.sample_id, edited.sample_id)
    assert edit.prompt == "Remove the baseline wander."
    generation = tasks[TSGenerationTask]
    assert isinstance(generation, TSGenerationTask)
    assert generation.target_sample_id == edited.sample_id
