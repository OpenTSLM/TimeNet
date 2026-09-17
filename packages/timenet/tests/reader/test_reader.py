from fractions import Fraction
import json
from pathlib import Path
import pickle

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


def _write(tmp_path, dataset=None, **kwargs) -> Path:
    dataset = dataset if dataset is not None else make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _read(version_dir: Path) -> TimeFDataset:
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read()


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_full_round_trip(tmp_path, backend):
    original = make_dataset()
    restored = _read(_write(tmp_path, dataset=make_dataset(), values_backend=backend))
    assert_datasets_equal(original, restored)


def _referenced_annotation_dataset() -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/referenced-annotation",
            dataset_version=Version(1, 0, 0),
            name="Referenced annotation",
            description="A task references an annotation occurrence on its input record.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    signal = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        data=[0.0, 1.0, 2.0],
        time_series_id="ecg-rec-0-I",
    )
    record = dataset.add_record(time_series=(signal,), record_id="rec-0")
    options = record.add_annotation(Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno"))
    dataset.add_task(
        task=AnswerTask(
            prompt="Rhythm?",
            targets=("yes",),
            inputs=(record,),
            input_annotations=(options,),
            id="qa-0",
        )
    )
    return dataset


def test_annotation_occurrence_and_task_object_references_round_trip(tmp_path):
    original = _referenced_annotation_dataset()
    restored = _read(_write(tmp_path, dataset=_referenced_annotation_dataset()))
    assert_datasets_equal(original, restored)
    assert restored.tasks[0].inputs == (restored.records[0],)
    assert restored.tasks[0].input_annotations == restored.records[0].annotations


def _tasks_dataset(*, streaming: bool) -> TimeFDataset:
    dataset = _referenced_annotation_dataset()
    dataset._tasks.clear()
    dataset.records[0].task_ids = ()
    record = dataset.records[0]
    options = record.annotations[0]
    tasks = [
        AnswerTask(
            prompt=f"Question {index}?",
            targets=("yes",),
            rationale=f"reason {index}",
            inputs=(record,),
            input_annotations=(options,),
            id=f"qa-{index}",
        )
        for index in range(5)
    ]
    if streaming:
        dataset.set_task_stream([AnswerTask], lambda: iter(tasks))
    else:
        dataset.add_tasks(tasks=tasks)
    return dataset


def test_streamed_tasks_round_trip_and_back_populate_records(tmp_path):
    restored = _read(_write(tmp_path, dataset=_tasks_dataset(streaming=True)))
    task = next(task for task in restored.tasks if task.id == "qa-3")
    assert task.prompt == "Question 3?"
    assert task.rationale == "reason 3"
    assert task.inputs == (restored.records[0],)
    assert {task.id for task in restored.tasks_for(restored.records[0])} == {task.id for task in restored.tasks}


def test_streamed_and_batched_tasks_read_back_equally(tmp_path):
    batched = _read(_write(tmp_path / "batch", dataset=_tasks_dataset(streaming=False)))
    streamed = _read(_write(tmp_path / "stream", dataset=_tasks_dataset(streaming=True)))
    assert {task.id: task for task in batched.tasks} == {task.id: task for task in streamed.tasks}


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_split_value_chunks_round_trip_and_support_range_reads(tmp_path, backend):
    version_dir = _write(tmp_path, values_backend=backend, chunk_max_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        signal = next(iter(reader.iter_records())).signals[0]
        expected = signal.to_arrow().slice(3, 7)
        assert signal.read_steps(3, 10).equals(expected)


def test_metadata_and_schema_come_from_the_manifest_without_opening_control(tmp_path):
    reader = TimeFReader(DatasetVersion.open_local(_write(tmp_path)))
    assert reader._control is None
    assert reader.metadata.dataset_id == "timenet/hello-world"
    assert {spec.spec_type for spec in reader.schema.time_series_specs} == {"sine", "cosine"}
    assert ClassificationTask in reader.schema.tasks
    assert reader._control is None


def test_annotation_shapes_and_value_types_round_trip(tmp_path):
    restored = _read(_write(tmp_path))
    annotations = {annotation.key: annotation for annotation in restored.records[0].annotations}
    assert annotations["age"].value == 64
    assert isinstance(annotations["age"].value, int)
    assert isinstance(annotations["stimulus"].span, TimePoint)
    assert isinstance(annotations["artifact"].span, TimeInterval)


def test_task_derivation_round_trips_as_object_references(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        tasks = {task.id: task for task in reader.tasks}
        parent = tasks["task-cls-0"]
        child = tasks["task-answer-0"]
        assert child.from_tasks == (parent,)
        assert child.inputs[0] is parent.inputs[0]


def test_iter_records_matches_full_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        streamed = {record.record_id for record in reader.iter_records()}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        materialized = {record.record_id for record in reader.read().records}
    assert streamed == materialized


def test_signal_values_remain_lazy_until_access(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original_open = pq.ParquetFile
    opens = 0

    def counting_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        dataset = reader.read()
        monkeypatch.setattr(pq, "ParquetFile", counting_open)
        signal = dataset.records[0].signals[0]
        assert opens == 0
        signal.to_arrow()
        assert opens >= 1


def test_signal_values_batch_lazy_chunk_locator_queries(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original = DuckDBControlReader.chunk_rows_by_keys
    calls = []

    def counting_batch(reader, signals):
        calls.append(tuple(signals))
        return original(reader, signals)

    monkeypatch.setattr(DuckDBControlReader, "chunk_rows_by_keys", counting_batch)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        records = tuple(reader.iter_records())
        signals = tuple(signal for record in records for signal in record.signals)
        assert calls == []

        for signal in signals:
            signal.to_arrow()

    assert len(calls) == 1
    assert len(calls[0]) == len(signals)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_dataset_and_lazy_loaders_are_picklable(tmp_path, backend):
    dataset = TimeFReader(DatasetVersion.open_local(_write(tmp_path, values_backend=backend))).read()
    expected = dataset.records[0].signals[0].to_arrow()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.records[0].signals[0].to_arrow().equals(expected)


def test_tasks_open_control_on_first_access_and_are_cached(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        assert reader._control is None
        assert reader._tasks is None
        first = reader.tasks
        assert reader._control is not None
        assert reader.tasks is first


def test_pickling_drops_connections_and_decoded_caches(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        _ = reader.tasks
        reader.read().records[0].signals[0].to_arrow()
        state = reader.__getstate__()
    assert state["_tasks"] is None
    assert state["_control"] is None
    assert state["_records"] is None
    assert state["_values"] is None


def test_iter_records_preserves_requested_order(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        selected = tuple(reader.iter_records(record_ids=("record-2", "record-0")))
    assert [record.record_id for record in selected] == ["record-2", "record-0"]


def test_iter_records_rejects_an_unknown_id_before_yielding(tmp_path):
    with (
        TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader,
        pytest.raises(TimeFValidationError, match="no such record"),
    ):
        next(reader.iter_records(record_ids=("record-0", "record-nope")))


def test_iter_records_accepts_an_empty_selection(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        assert tuple(reader.iter_records(record_ids=())) == ()


def test_iter_records_can_skip_annotations_without_losing_values(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        record = next(reader.iter_records(with_annotations=False))
        assert record.annotations == ()
        assert all(source.annotations == () for source in record.walk_sources())
        assert all(signal.annotations == () for signal in record.signals)
        assert record.signals[0].to_arrow()


def test_missing_root_or_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(empty)


@pytest.mark.parametrize("format_version", [1, 3, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TimeFFormatError):
        DatasetVersion.open_local(version_dir)


def test_verify_accepts_an_intact_dataset_and_detects_a_changed_artifact(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        reader.verify()

    control_path = version_dir / "control.duckdb"
    payload = bytearray(control_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    control_path.write_bytes(payload)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="checksum mismatch"),
    ):
        reader.verify()
