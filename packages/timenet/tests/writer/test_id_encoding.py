"""Task spans and record references keep their entity ids through a write and read."""

from dataclasses import replace

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import (
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
    return Signal.from_loader(
        spec=_spec(),
        name="c",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=3,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )


def _write(tmp_path, dataset):
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_forecasting_target_span_round_trips(tmp_path):
    """target_span is the only single-Span payload column; exercise its encode/decode branch."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/id-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(_series(),)),)))
    span = TimeInterval.seconds(1.0, 3.0, time_series=record.signals)
    scope = TimeInterval.seconds(0.0, 1.0, time_series=record.signals)
    dataset.add_task(task=ForecastingTask(inputs=(record,), targets=(span,), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    restored = task.inputs[0].signals
    assert task.targets == (replace(span, time_series=restored),)
    assert task.scope == replace(scope, time_series=restored)


def test_forecasting_step_horizon_round_trips(tmp_path):
    """A steps target_span carries its frame and bounds through the writer and reader."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/id-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    ordinal = Signal.from_values([float(i) for i in range(6)], spec=_spec(), name="c", time_axis=OrdinalAxis())
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(ordinal,)),)))
    span = StepInterval(time_series=ordinal, start=4, stop=6)
    scope = StepInterval(time_series=ordinal, start=0, stop=4)
    dataset.add_task(task=ForecastingTask(inputs=(record,), targets=(span,), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = reader.tasks[0]
    assert isinstance(task, ForecastingTask)
    (restored,) = task.inputs[0].signals
    assert task.targets == (replace(span, time_series=restored),)
    assert task.scope == replace(scope, time_series=restored)


def test_a_span_reads_back_the_signals_of_its_record(tmp_path):
    """A task scope refers to the Signal objects that the reader returns for its input, not to copies."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/id-test",
            dataset_version=Version(1, 0, 0),
            name="U",
            description="d",
            license=License.MIT,
        )
    )
    series = _series()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(series,)),)))
    other = _series()
    dataset.add_record(record=Record(sources=(Source(name="Other", signals=(other,)),)))
    scope = TimeInterval.seconds(0.0, 1.0, time_series=(series,))
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("x",), scope=scope))
    dataset.derive_schema()
    version_dir = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (task,) = reader.tasks
    assert isinstance(task.scope, TimeInterval)
    assert task.scope.time_series is not None
    assert task.scope.time_series[0] is task.inputs[0].signals[0]
    assert task.scope.time_series[0].id == series.id


def test_correspondence_record_targets_round_trip(tmp_path):
    """Correspondence Record targets and candidates return as hydrated objects."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/id-test",
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
            dataset_id="timenet/id-test",
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
            dataset_id="timenet/id-test",
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
