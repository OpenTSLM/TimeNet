"""IDs use canonical strings in DuckDB and round-trip without changing their values."""

import uuid

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
    TimeInterval,
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
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("x",)))
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


def test_uuid_ids_round_trip_as_canonical_strings(tmp_path):
    dataset = _uuid_dataset()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(dataset, restored)
    sid = restored.records[0].record_id
    assert str(uuid.UUID(sid)) == sid
    assert uuid.UUID(sid).version == 7


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
    dataset.add_task(task=ForecastingTask(inputs=(record,), targets=(span,), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.targets == (span,)
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
    dataset.add_task(task=ForecastingTask(inputs=(record,), targets=(span,), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    assert task.targets == (span,)
    assert task.scope == scope


def test_span_series_ids_round_trip(tmp_path):
    """A task scope keeps its Signal ID scope through the integer-key scope columns."""
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
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("x",), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (task,) = reader.tasks
    assert task.scope == scope


def test_correspondence_record_targets_round_trip(tmp_path):
    """Correspondence Record targets and candidates return as hydrated objects."""
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
        task=TSCorrespondenceTask(
            inputs=(query,),
            prompt="Which trace is most similar?",
            candidate_records=(match, other),
            targets=(match,),
        )
    )
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, TSCorrespondenceTask)
    assert task.targets is not None and isinstance(task.targets[0], Record)
    assert task.targets[0].id == match.id
    assert tuple(record.id for record in task.candidate_records) == (match.id, other.id)


def test_editing_and_generation_record_targets_round_trip(tmp_path):
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
        task=TSEditingTask(
            inputs=(source,),
            prompt="Remove the baseline wander.",
            targets=(edited,),
        )
    )
    dataset.add_task(task=TSGenerationTask(prompt="10 s of sinus rhythm.", targets=(edited,)))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        tasks = {type(t): t for t in reader.tasks}
    edit = tasks[TSEditingTask]
    assert isinstance(edit, TSEditingTask)
    assert tuple(record.id for record in edit.inputs) == (source.id,)
    assert edit.targets is not None and isinstance(edit.targets[0], Record)
    assert edit.targets[0].id == edited.id
    assert edit.prompt == "Remove the baseline wander."
    generation = tasks[TSGenerationTask]
    assert isinstance(generation, TSGenerationTask)
    assert generation.targets is not None and isinstance(generation.targets[0], Record)
    assert generation.targets[0].id == edited.id


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
