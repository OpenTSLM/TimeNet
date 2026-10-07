from fractions import Fraction
import json
from pathlib import Path
import pickle

import pyarrow.parquet as pq
from pydantic import ValidationError
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.control_reader import DuckDBControlReader
from timenet.manifest import LockedDependency
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion, LocalRegistry
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DatasetRef,
    Domain,
    License,
    Split,
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
    signal = Signal(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        name="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        data=[0.0, 1.0, 2.0],
        id="ecg-rec-0-I",
    )
    record = dataset.add_record(
        record=Record(
            sources=(Source(id="rec-0-source", name="Source", signals=(signal,)),),
            record_id="rec-0",
        )
    )
    options = record.annotate(Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno"))
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
    dataset._task_ids.clear()
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


def test_iter_records_matches_full_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        streamed = {record.record_id for record in reader.iter_records()}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        materialized = {record.record_id for record in reader.read().records}
    assert streamed == materialized


def _child_metadata(parent_ref: DatasetRef) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="test/composed-child",
        dataset_version=Version(1, 0, 0),
        name="Composed child",
        description="Tasks over parent records.",
        license=License.CC_BY_4_0,
        parents=(DatasetRef(dataset_id=parent_ref.dataset_id, version=parent_ref.version),),
    )


def _lock(parent_dir: Path, parent_ref: DatasetRef) -> LockedDependency:
    return LockedDependency(
        dataset_id=parent_ref.dataset_id,
        version=parent_ref.version,
        manifest_checksum=file_checksum(parent_dir / "manifest.json"),
    )


def test_composed_child_reuses_parent_record_values_and_adds_tasks(tmp_path):
    parent_dataset = make_dataset()
    parent_dir = _write(tmp_path, dataset=parent_dataset)
    parent_ref = DatasetRef(
        dataset_id=parent_dataset.metadata.dataset_id,
        version=parent_dataset.metadata.dataset_version,
    )
    with TimeFReader(DatasetVersion.open_local(parent_dir)) as parent_reader:
        imported = next(parent_reader.iter_records())
        child = TimeFDataset(metadata=_child_metadata(parent_ref))
        with pytest.raises(TimeFValidationError, match="no parent"):
            child.import_record(imported, parent="other")
        child.import_record(imported, parent="timenet/hello-world")
        child.set_dependencies((_lock(parent_dir, parent_ref),))
        imported.annotate(Annotation(key="reviewed", value=True, id="reviewed"))
        imported.annotate(Annotation(key="reviewed_at", span=TimePoint.micros(0), id="reviewed-at"))
        child.add_task(
            task=AnswerTask(id="child-question", prompt="Is this imported?", targets=("yes",), inputs=(imported,)),
        )
        child_dir = _write(tmp_path, dataset=child)

    child_version = DatasetVersion.open_local(child_dir)
    assert child_version.manifest.files.time_series == ()
    with LocalRegistry(tmp_path).open_reader("test/composed-child", "1.0.0") as child_reader:
        restored = child_reader.read()
        record = restored.records[0]
        assert restored.tasks[0].inputs == (record,)
        assert record.task_ids == ("child-question",)
        assert {"reviewed", "reviewed-at"} <= {annotation.id for annotation in record.annotations}
        assert record.signals[0].to_arrow().equals(parent_dataset.records[0].signals[0].to_arrow())


def test_composed_child_resolves_imports_in_filtered_task_reads_with_one_parent_query(tmp_path, monkeypatch):
    parent_dataset = make_dataset()
    parent_dir = _write(tmp_path, dataset=parent_dataset)
    parent_ref = DatasetRef(
        dataset_id=parent_dataset.metadata.dataset_id,
        version=parent_dataset.metadata.dataset_version,
    )
    with TimeFReader(DatasetVersion.open_local(parent_dir)) as parent_reader:
        child = TimeFDataset(metadata=_child_metadata(parent_ref))
        child.set_dependencies((_lock(parent_dir, parent_ref),))
        for index, record in enumerate(parent_reader.iter_records()):
            child.import_record(record, parent="timenet/hello-world")
            child.add_task(
                task=ClassificationTask(
                    id=f"child-cls-{index}",
                    inputs=(record,),
                    targets=("x",),
                    split=Split.TEST if index % 2 else Split.TRAIN,
                ),
            )
        _write(tmp_path, dataset=child)

    parent_reads = 0
    read_records = DuckDBControlReader.read_records

    def counting(self, *args, **kwargs):
        nonlocal parent_reads
        parent_reads += self.path.is_relative_to(parent_dir)
        return read_records(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
    with LocalRegistry(tmp_path).open_reader("test/composed-child", "1.0.0") as child_reader:
        # A filtered task read hydrates its input records itself, bypassing the reader's record cache.
        tasks = list(child_reader.iter_tasks(split="test"))
        assert [task.id for task in tasks] == ["child-cls-1"]
        assert tasks[0].inputs[0].signals
        assert parent_reads == 1
        assert all(record.signals for record in child_reader.read().records)
        assert parent_reads == 2


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


def test_tasks_are_cached_after_first_access(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        first = reader.tasks
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


@pytest.mark.parametrize("format_version", [2, 3, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="timef_format_version"):
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


def test_iter_tasks_streams_one_record_and_points_at_its_hydrated_objects(tmp_path):
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
        record0 = next(reader.iter_records(("record-0",)))
        tasks = list(reader.iter_tasks((record0,)))
        every = reader.tasks

    assert {task.id for task in tasks} == {task.id for task in every if task.inputs[0].record_id == "record-0"}
    assert all(task.inputs[0] is record0 for task in tasks)
    answer = next(task for task in tasks if task.id == "task-answer-0")
    cohort = next(annotation for annotation in record0.annotations if annotation.id == "cohort-shared")
    assert answer.input_annotations[0] is cohort


def test_split_filters_run_in_duckdb_and_survive_a_full_read(tmp_path):
    dataset = make_dataset()
    tasks = {task.id: task for task in dataset.tasks}
    tasks["task-cls-0"].split = Split.TRAIN
    tasks["task-scalar-0"].split = Split.TEST
    version_dir = _write(tmp_path, dataset)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert [task.id for task in reader.iter_tasks(split="train")] == ["task-cls-0"]
        assert reader.task_table(split=Split.TEST)["task_id"].to_pylist() == ["task-scalar-0"]
        assert reader.target_table(split=Split.TEST)["task_id"].to_pylist() == ["task-scalar-0"]
        loaded = reader.read()

    assert {task.id: task.split for task in loaded.tasks if task.split is not None} == {
        "task-cls-0": Split.TRAIN,
        "task-scalar-0": Split.TEST,
    }
