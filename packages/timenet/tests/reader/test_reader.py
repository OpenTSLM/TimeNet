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
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import PackedValueLocator
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
    License,
    ObjectKind,
    ScalarPredictionTask,
    Split,
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
    with TimeFReader(open_local_version(version_dir)) as reader:
        return reader.read()


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_full_round_trip(tmp_path, backend):
    original = make_dataset()
    answer = next(task for task in original.tasks if isinstance(task, AnswerTask))
    text = "A detailed clinical report with café notes. " * 300
    answer.prompt = text
    answer.rationale = text + "Explanation."
    answer.targets = (text + "Answer.", "short answer")
    record = answer.inputs[0]
    record.annotate(Annotation(key="report", value=text))
    record.annotate(Annotation(key="reports", value=[text, "short report"]))

    restored = _read(_write(tmp_path, dataset=original, values_backend=backend))
    assert_datasets_equal(original, restored)
    restored_answer = next(task for task in restored.tasks if task.id == answer.id)
    assert restored_answer.prompt == text
    assert restored_answer.rationale == text + "Explanation."
    # Targets are excluded from task equality, so compare them directly.
    assert restored_answer.targets == (text + "Answer.", "short answer")


def _dataset_with_packed_text():
    dataset = make_dataset()
    text = "A detailed clinical report with café notes. " * 300
    for task in dataset.tasks:
        task.prompt = f"{text}{task.id}"
        task.rationale = text + "Explanation."
        task.split = Split.TEST
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.targets = (text + "Answer.",)
    answer.split = Split.TRAIN
    for record in dataset.records:
        record.annotate(Annotation(key="report", value=f"{text}{record.id}"))
    answer.annotate(Annotation(key="report", value=text))
    return dataset, answer, text


def _report_content_ids(version_dir: Path) -> dict[tuple[str, int], int]:
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        rows = connection.execute(
            """SELECT o.object_type, o.object_id, o.content_id
               FROM annotation_occurrences o JOIN annotation_contents c USING (content_id)
               WHERE c.name = 'report'"""
        ).fetchall()
    return {(object_type, object_id): content_id for object_type, object_id, content_id in rows}


@pytest.fixture
def validated_locators(monkeypatch):
    validated = []
    original = PackedValueLocator.model_validate

    def validate(row):
        locator = original(row)
        validated.append((locator.object_kind, locator.object_id))
        return locator

    monkeypatch.setattr(PackedValueLocator, "model_validate", validate)
    return validated


def test_filtered_reads_validate_only_selected_text_locators(tmp_path, validated_locators):
    dataset, answer, text = _dataset_with_packed_text()
    (record,) = answer.inputs
    version_dir = _write(tmp_path, dataset)
    contents = _report_content_ids(version_dir)
    record_report = (ObjectKind.ANNOTATION_CONTENT, contents[ObjectKind.RECORD, record.id])
    task_report = (ObjectKind.ANNOTATION_CONTENT, contents[ObjectKind.TASK, answer.id])
    validated_locators.clear()

    with TimeFReader(open_local_version(version_dir)) as reader:
        loaded_record = next(reader.iter_records((record.id,)))
        (report,) = [annotation for annotation in loaded_record.annotations if annotation.key == "report"]
        assert report.value == f"{text}{record.id}"
        assert set(validated_locators) == {record_report}

        validated_locators.clear()
        (loaded_task,) = reader.iter_tasks((loaded_record,), split=Split.TRAIN)
        assert loaded_task.prompt == f"{text}{answer.id}"
        assert loaded_task.rationale == text + "Explanation."
        assert loaded_task.targets is not None
        assert loaded_task.targets[0] == text + "Answer."
        assert loaded_task.annotations[0].value == text
        assert set(validated_locators) == {
            (ObjectKind.TASK, answer.id),
            task_report,
            *((ObjectKind.TASK, parent.id) for parent in answer.from_tasks),
        }

        validated_locators.clear()
        assert reader.read_task_table((record.id,), split=Split.TRAIN)["prompt"].to_pylist() == [f"{text}{answer.id}"]
        assert reader.read_target_table((record.id,), split=Split.TRAIN)["text_value"].to_pylist() == [text + "Answer."]
        assert set(validated_locators) == {(ObjectKind.TASK, answer.id)}

        validated_locators.clear()
        assert reader.read_annotation_table(ObjectKind.TASK)["text_value"].to_pylist() == [text]
        assert set(validated_locators) == {task_report}


def test_task_stream_reads_text_locators_one_batch_at_a_time(tmp_path, monkeypatch, validated_locators):
    dataset, _, text = _dataset_with_packed_text()
    version = open_local_version(_write(tmp_path, dataset))
    validated_locators.clear()
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 2)
    first_batch_ids = {task.id for task in sorted(dataset.tasks, key=lambda task: task.id)[:2]}

    with TimeFReader(version) as reader:
        stream = reader.iter_tasks()
        first = next(stream)
        assert first.prompt == f"{text}{first.id}"
        assert {identity for kind, identity in validated_locators if kind == ObjectKind.TASK} == first_batch_ids
        remaining = list(stream)
        assert {task.id for task in [first, *remaining]} == {task.id for task in dataset.tasks}
        assert all(task.prompt == f"{text}{task.id}" for task in remaining)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
@pytest.mark.parametrize("subjects", [(), ("patient-1",), ("patient-2", "patient-1")])
def test_subject_annotations_round_trip_and_filter(tmp_path, backend, subjects):
    dataset = _referenced_annotation_dataset()
    record = dataset.records[0]
    if subjects:
        record.annotate(Annotation(key="subject_ids", value=list(subjects)))
    version_dir = _write(tmp_path, dataset=dataset, values_backend=backend)

    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
        (selected,) = reader.iter_records((record.id,))

    for back in (restored.records[0], selected):
        values = [annotation.value for annotation in back.annotations if annotation.key == "subject_ids"]
        assert values == ([list(subjects)] if subjects else [])
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        matches = connection.execute(
            """SELECT DISTINCT r.record_id
               FROM records r
               JOIN annotation_occurrences o
                 ON o.object_type = 'Record' AND o.object_id = r.record_id
               JOIN annotation_contents c USING (content_id)
               WHERE c.name = 'subject_ids'
                 AND list_contains(c.text_list_value, ?)""",
            ["patient-1"],
        ).fetchall()
    assert matches == ([(record.id,)] if subjects else [])


def _referenced_annotation_dataset(*, with_task: bool = True) -> TimeFDataset:
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
    )
    record = dataset.add_record(
        record=Record(
            sources=(Source(name="Source", signals=(signal,)),),
        )
    )
    options = record.annotate(Annotation(key="answer_options", value=["yes", "no"]))
    if with_task:
        dataset.add_task(
            task=AnswerTask(
                prompt="Rhythm?",
                targets=("yes",),
                inputs=(record,),
                input_annotations=(options,),
            )
        )
    return dataset


def test_annotation_occurrence_and_task_object_references_round_trip(tmp_path):
    original = _referenced_annotation_dataset()
    restored = _read(_write(tmp_path, dataset=_referenced_annotation_dataset()))
    assert_datasets_equal(original, restored)
    assert restored.tasks[0].inputs == (restored.records[0],)
    assert restored.tasks[0].input_annotations == restored.records[0].annotations


# A constant to reuse
HEALTHY_ANNOTATION = Annotation(key="condition", value="healthy", description="Reported by the source.")


def test_a_read_back_annotation_equals_the_constant_it_was_written_from(tmp_path):
    dataset = _referenced_annotation_dataset()
    record = dataset.records[0]
    record.annotate(HEALTHY_ANNOTATION)
    record.signals[0].annotate(HEALTHY_ANNOTATION)

    version_dir = _write(tmp_path, dataset=dataset)
    restored = _read(version_dir).records[0]

    occurrences = [
        annotation
        for annotation in (*restored.annotations, *restored.signals[0].annotations)
        if annotation.key == "condition"
    ]
    assert len(occurrences) == 2
    assert all(type(annotation) is Annotation for annotation in occurrences)
    assert occurrences == [HEALTHY_ANNOTATION, HEALTHY_ANNOTATION]
    assert occurrences[0] is not occurrences[1]
    # Both occurrences share the constant's content, which the writer stores once.
    assert {annotation._content_key for annotation in occurrences} == {HEALTHY_ANNOTATION._content_key}
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        stored = connection.execute(
            """SELECT count(DISTINCT content_id), count(*)
               FROM annotation_occurrences JOIN annotation_contents USING (content_id)
               WHERE name = 'condition'"""
        ).fetchone()
    assert stored == (1, 2)


def _tasks_dataset(*, streaming: bool) -> TimeFDataset:
    dataset = _referenced_annotation_dataset(with_task=False)
    record = dataset.records[0]
    options = record.annotations[0]
    tasks = [
        AnswerTask(
            prompt=f"Question {index}?",
            targets=("yes",),
            rationale=f"reason {index}",
            inputs=(record,),
            input_annotations=(options,),
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
    task = next(task for task in restored.tasks if task.prompt == "Question 3?")
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
    with TimeFReader(open_local_version(version_dir)) as reader:
        signal = next(iter(reader.iter_records())).signals[0]
        expected = signal.to_arrow().slice(3, 7)
        assert signal.read_steps(3, 10).equals(expected)


def test_metadata_and_schema_come_from_the_manifest_without_opening_control(tmp_path):
    dataset = make_dataset()
    reader = TimeFReader(open_local_version(_write(tmp_path, dataset)))
    assert reader._control is None
    assert reader.metadata == dataset.metadata
    assert reader.schema == dataset.schema
    assert reader._control is None


def test_iter_records_matches_full_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(open_local_version(version_dir)) as reader:
        streamed = {record.id for record in reader.iter_records()}
    with TimeFReader(open_local_version(version_dir)) as reader:
        materialized = {record.id for record in reader.read().records}
    assert streamed == materialized


def test_signal_values_remain_lazy_until_access(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original_open = pq.ParquetFile
    opens = 0

    def counting_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    with TimeFReader(open_local_version(version_dir)) as reader:
        dataset = reader.read()
        monkeypatch.setattr(pq, "ParquetFile", counting_open)
        signal = dataset.records[0].signals[0]
        assert opens == 0
        signal.to_arrow()
        assert opens >= 1


def test_signal_values_batch_lazy_chunk_locator_queries(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original = DuckDBControlReader.chunk_locations_by_ids
    calls = []

    def counting_batch(reader, signals):
        calls.append(tuple(signals))
        return original(reader, signals)

    monkeypatch.setattr(DuckDBControlReader, "chunk_locations_by_ids", counting_batch)
    with TimeFReader(open_local_version(version_dir)) as reader:
        records = tuple(reader.iter_records())
        signals = tuple(signal for record in records for signal in record.signals)
        assert calls == []

        for signal in signals:
            signal.to_arrow()

    assert len(calls) == 1
    assert len(calls[0]) == len(signals)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_dataset_and_lazy_loaders_are_picklable(tmp_path, backend):
    dataset = TimeFReader(open_local_version(_write(tmp_path, values_backend=backend))).read()
    expected = dataset.records[0].signals[0].to_arrow()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.records[0].signals[0].to_arrow().equals(expected)


def test_tasks_are_cached_after_first_access(tmp_path):
    with TimeFReader(open_local_version(_write(tmp_path))) as reader:
        first = reader.tasks
        assert reader.tasks is first


def test_pickling_drops_connections_and_decoded_caches(tmp_path, monkeypatch):
    dataset = make_dataset()
    text = "A detailed clinical report. " * 300
    dataset.tasks[0].prompt = text
    assert dataset.tasks[0].prompt is not None
    assert dataset.tasks[0].prompt.read() == text
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        original = pq.ParquetFile.read
        read_calls = 0

        def count_reads(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, "read", count_reads)
        loaded = reader.read()
        loaded.derive_schema()
        task = next(task for task in loaded.tasks if task.id == dataset.tasks[0].id)
        assert task.prompt is not None
        assert task.prompt.length == len(text)
        assert read_calls == 0
        assert task.prompt.read() == text
        assert read_calls == 1
        loaded.records[0].signals[0].to_arrow()
        assert text in reader.read_task_table()["prompt"].to_pylist()
        assert read_calls == 1
        state = reader.__getstate__()
        with pickle.loads(pickle.dumps(reader)) as restored:
            assert text in restored.read_task_table()["prompt"].to_pylist()
        assert read_calls == 2
    with reader:
        assert text in reader.read_task_table()["prompt"].to_pylist()
    assert read_calls == 3
    assert state["_tasks"] is None
    assert state["_control"] is None
    assert state["_records"] is None
    assert state["_values"] is None


def test_iter_records_preserves_requested_order(tmp_path):
    dataset = make_dataset()
    ordered_records = sorted(dataset.records, key=lambda record: record.id)
    requested = (ordered_records[-1].id, ordered_records[0].id)
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        selected = tuple(reader.iter_records(record_ids=requested))
    assert tuple(record.id for record in selected) == requested


def test_iter_records_rejects_an_unknown_id_before_yielding(tmp_path):
    dataset = make_dataset()
    with (
        TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader,
        pytest.raises(TimeFValidationError, match="no such record"),
    ):
        next(reader.iter_records(record_ids=(dataset.records[0].id, 99)))


def test_iter_records_can_skip_annotations_without_losing_values(tmp_path):
    with TimeFReader(open_local_version(_write(tmp_path))) as reader:
        record = next(reader.iter_records(with_annotations=False))
        assert record.annotations == ()
        assert all(source.annotations == () for source in record.walk_sources())
        assert all(signal.annotations == () for signal in record.signals)
        assert record.signals[0].to_arrow()


def test_missing_root_or_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_local_version(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        open_local_version(empty)


@pytest.mark.parametrize("format_version", [2, 3, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="timef_format_version"):
        open_local_version(version_dir)


def test_verify_accepts_an_intact_dataset_and_detects_a_changed_artifact(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()

    control_path = version_dir / "control.duckdb"
    payload = bytearray(control_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    control_path.write_bytes(payload)
    with (
        TimeFReader(open_local_version(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="checksum mismatch"),
    ):
        reader.verify()


def test_iter_tasks_streams_one_record_and_points_at_its_built_objects(tmp_path):
    dataset = make_dataset()
    expected_answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (record,) = expected_answer.inputs
    (expected_context,) = expected_answer.input_annotations
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        record0 = next(reader.iter_records((record.id,)))
        tasks = list(reader.iter_tasks((record0,)))
        every = reader.tasks

    assert {task.id for task in tasks} == {
        task.id for task in every if any(item.id == record.id for item in task.inputs)
    }
    assert all(task.inputs[0] is record0 for task in tasks)
    answer = next(task for task in tasks if task.id == expected_answer.id)
    cohort = next(annotation for annotation in record0.annotations if annotation == expected_context)
    assert answer.input_annotations[0] is cohort


def test_split_filters_run_in_duckdb_and_survive_a_full_read(tmp_path):
    dataset = make_dataset()
    train_task = next(task for task in dataset.tasks if isinstance(task, ClassificationTask))
    test_task = next(task for task in dataset.tasks if isinstance(task, ScalarPredictionTask))
    train_task.split = Split.TRAIN
    test_task.split = Split.TEST
    version_dir = _write(tmp_path, dataset)

    with TimeFReader(open_local_version(version_dir)) as reader:
        assert [task.id for task in reader.iter_tasks(split="train")] == [train_task.id]
        assert reader.read_task_table(split=Split.TEST)["task_id"].to_pylist() == [test_task.id]
        assert reader.read_target_table(split=Split.TEST)["task_id"].to_pylist() == [test_task.id]
        loaded = reader.read()

    assert {task.id: task.split for task in loaded.tasks if task.split is not None} == {
        train_task.id: Split.TRAIN,
        test_task.id: Split.TEST,
    }


def test_task_views_reject_a_non_int_record_id(tmp_path):
    dataset = make_dataset()
    record0 = dataset.records[0]
    with TimeFReader(open_local_version(_write(tmp_path, dataset))) as reader:
        for view in (reader.read_task_table, reader.read_target_table):
            with pytest.raises(TimeFValidationError, match="positive int64 id"):
                view(record_ids=(str(record0.id),))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "column,value", [("field", "promtp"), ("field", "offsets"), ("row_index", -1), ("value_kind", "texts")]
)
def test_invalid_external_locator_is_rejected_before_exposing_placeholders(tmp_path, column, value):
    dataset = make_dataset()
    dataset.tasks[0].prompt = "A detailed clinical report. " * 300
    version_dir = _write(tmp_path, dataset)
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute(
            f"UPDATE external_values SET {column} = ? WHERE field = 'prompt'",  # noqa: S608 - fixed test columns
            [value],
        )

    with (
        TimeFReader(version) as reader,
        pytest.raises(TimeFFormatError, match="invalid external locator"),
    ):
        reader.read_task_table()
