"""IDs use canonical strings in DuckDB and round-trip without changing their values."""

import uuid

import duckdb
import pyarrow as pa

from timenet.dataset import Record, Source, TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    License,
    StepInterval,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


def _spec():
    return TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )


def _series():
    return TimeSeries.from_loader(
        spec=_spec(),
        name="c",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=3,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )


def _uuid_dataset(*, record_id=None):
    """Build a dataset whose ids default to uuid7 (unless an explicit record_id is passed)."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    series = (_series(),)
    sources = (Source(name="Source", signals=series),)
    record = Record(sources=sources) if record_id is None else Record(sources=sources, record_id=record_id)
    dataset.add_record(record=record)
    record.add_annotation(Annotation(key="k", value=1))
    dataset.add_task(record, ClassificationTask(target="x"))
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset):
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_default_ids_are_uuid7():
    dataset = _uuid_dataset()
    sid = dataset.records[0].record_id
    assert uuid.UUID(sid).version == 7
    assert str(uuid.UUID(sid)) == sid  # canonical


def test_manifest_has_no_id_encoding(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset())
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert not hasattr(manifest, "id_encoding") or "id_encoding" not in manifest.to_dict()


def test_uuid_id_columns_are_varchar_in_control_database(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset())
    with duckdb.connect(str(version_dir / "control.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT typeof(record_id) FROM records").fetchone() == ("VARCHAR",)
        assert connection.execute("SELECT typeof(signal_id) FROM signals").fetchone() == ("VARCHAR",)
        assert connection.execute("SELECT typeof(content_id) FROM annotation_contents").fetchone() == ("VARCHAR",)


def test_uuid_ids_round_trip_as_canonical_strings(tmp_path):
    dataset = _uuid_dataset()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(dataset, restored)
    sid = restored.records[0].record_id
    assert str(uuid.UUID(sid)) == sid
    assert uuid.UUID(sid).version == 7


def test_forecasting_scalar_id_round_trips(tmp_path):
    """target_record_id is a scalar id column; exercise its binary(16) encode/decode."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    context = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    target = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    dataset.add_task(
        target,
        ForecastingTask(context_record_ids=(context.record_id,), target_record_id=target.record_id),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.target_record_id == target.record_id
    assert task.context_record_ids == (context.record_id,)


def test_forecasting_target_span_round_trips(tmp_path):
    """target_span is the only single-Span payload column; exercise its encode/decode branch."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    series_id = record.signals[0].time_series_id
    span = TimeInterval.seconds(1.0, 3.0, time_series_ids=(series_id,))
    scope = TimeInterval.seconds(0.0, 1.0, time_series_ids=(series_id,))
    dataset.add_task(record, ForecastingTask(target_span=span, scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.target_span == span
    assert task.target_record_id is None
    assert task.scope == scope


def test_forecasting_step_horizon_round_trips(tmp_path):
    """A steps target_span carries its frame and bounds through the writer and reader."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    ordinal = TimeSeries.from_values([float(i) for i in range(6)], spec=_spec(), name="c", time_axis=OrdinalAxis())
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(ordinal,)),)))
    series_id = record.signals[0].time_series_id
    span = StepInterval(time_series_id=series_id, start=4, stop=6)
    scope = StepInterval(time_series_id=series_id, start=0, stop=4)
    dataset.add_task(record, ForecastingTask(target_span=span, scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.target_span == span
    assert task.scope == scope


def test_non_uuid_ids_round_trip_without_conversion(tmp_path):
    version_dir = _write(tmp_path, _uuid_dataset(record_id="record-0"))
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader.read().records[0].record_id == "record-0"


def test_span_series_ids_round_trip(tmp_path):
    """A span keeps nested Signal IDs through the normalized task representation."""
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
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(series,)),)))
    scope = TimeInterval.seconds(0.0, 1.0, time_series_ids=(series.time_series_id,))
    dataset.add_task(record, ClassificationTask(target="x", scope=scope))
    dataset.add_task(
        record,
        TemporalLocalizationTask(
            prompt="Locate the onsets.",
            target=(TimePoint.seconds(1.0, time_series_ids=(series.time_series_id,)),),
        ),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        tasks = {type(t): t for t in reader.tasks}
    assert tasks[ClassificationTask].scope == scope
    localization = tasks[TemporalLocalizationTask]
    assert isinstance(localization, TemporalLocalizationTask)
    assert localization.target == (TimePoint.seconds(1.0, time_series_ids=(series.time_series_id,)),)


def test_correspondence_target_ids_round_trip(tmp_path):
    """The correspondence answer is itself a tuple of record ids, so it encodes as an id column."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    query = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    match = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    other = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    dataset.add_task(
        query,
        TSCorrespondenceTask(
            prompt="Which trace is most similar?",
            candidate_record_ids=(match.record_id, other.record_id),
            target=(match.record_id,),
        ),
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, TSCorrespondenceTask)
    assert task.target == (match.record_id,)
    assert task.candidate_record_ids == (match.record_id, other.record_id)


def test_editing_and_generation_record_ids_round_trip(tmp_path):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    source = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    edited = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    dataset.add_task(
        source,
        TSEditingTask(
            prompt="Remove the baseline wander.",
            source_record_id=source.record_id,
            target_record_id=edited.record_id,
        ),
    )
    dataset.add_task(edited, TSGenerationTask(prompt="10 s of sinus rhythm.", target_record_id=edited.record_id))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        tasks = {type(t): t for t in reader.tasks}
    edit = tasks[TSEditingTask]
    assert isinstance(edit, TSEditingTask)
    assert (edit.source_record_id, edit.target_record_id) == (source.record_id, edited.record_id)
    assert edit.prompt == "Remove the baseline wander."
    generation = tasks[TSGenerationTask]
    assert isinstance(generation, TSGenerationTask)
    assert generation.target_record_id == edited.record_id


def test_time_span_round_trips(tmp_path):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/uuid-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    time_span = TimeInterval.seconds(0.0, 5.0)  # contains the series' [0, 3) s window
    dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),), time_span=time_span))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(reader.iter_records())
    assert record.time_span == time_span
    assert isinstance(record.time_span, TimeInterval)
