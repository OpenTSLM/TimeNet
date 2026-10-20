from fractions import Fraction
from types import SimpleNamespace

import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.control_audit import audit_control_database
import timenet.format.control_writer as control_writer_module
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    StepPoint,
    TimeInterval,
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(
    spec_type="voltage",
    name="Voltage",
    unit_value=ureg.millivolt,
    dtype="float32",
)
# The ids the dataset and the writer give the objects of _dataset(). Each kind counts from 1, and the
# writer numbers Sources and annotation contents in the order it writes them.
RECORD_ID = 1
MONITOR_ID = 1
ECG_ID = 2
LEAD_I_ID = 1
LEAD_II_ID = 2
SEX_MALE_ID = 1
TASK_ID = 1


def _dataset() -> TimeFDataset:
    axis = RegularAxis(period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    lead_ii = Signal.from_loader(
        name="II",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([3.0, 4.0], type=pa.float32()),
    )
    ecg = Source(name="ECG", signals=(lead_i, lead_ii))
    monitor = Source(name="Monitor", sources=(ecg,))
    record = Record(sources=(monitor,))
    record.annotate(Annotation(key="patient_sex", value="male"))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/control-writer",
            dataset_version=Version(1, 0, 0),
            name="Control writer",
            description="Test hierarchy",
            license=License.MIT,
        )
    )
    dataset.add_record(record=record)
    task = AnswerTask(
        inputs=(record,),
        prompt="Alive?",
        targets=("Yes",),
        input_annotations=(record.annotations[0],),
    )
    task.annotate(Annotation(key="task_kind", value="diagnosis"))
    dataset.add_task(task=task)
    return dataset


def test_control_writer_serializes_recursive_hierarchy_and_shared_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer_module, "_ROW_BATCH_SIZE", 2)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM records").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM sources").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM signals").fetchone() == (2,)
        assert connection.execute("SELECT count(*) FROM axes").fetchone() == (1,)
        assert connection.execute(
            "SELECT task_type, prompt, scope_type, target_schema, unit, target_name, mode FROM tasks"
        ).fetchone() == ("answer", "Alive?", None, None, None, None, None)
        assert connection.execute(
            "SELECT position, target_kind, text_value, integer_value, record_id FROM task_targets"
        ).fetchone() == (0, "text", "Yes", None, None)
        assert connection.execute("SELECT task_id, field, record_id FROM task_record_refs").fetchone() == (
            TASK_ID,
            "inputs",
            RECORD_ID,
        )
        assert connection.execute("SELECT parent_source_id FROM sources WHERE source_id = ?", [ECG_ID]).fetchone() == (
            MONITOR_ID,
        )
        assert connection.execute(
            """SELECT contents.content_id, occurrences.object_type, records.record_id
               FROM annotation_occurrences occurrences
               JOIN annotation_contents contents USING (content_id)
               JOIN records ON records.record_id = occurrences.object_id
               WHERE contents.content_id = ? AND occurrences.object_type = 'Record'""",
            [SEX_MALE_ID],
        ).fetchone() == (SEX_MALE_ID, "Record", RECORD_ID)


def test_control_writer_preserves_relationships_across_task_batches(tmp_path, monkeypatch):
    dataset = _dataset()
    record = dataset.records[0]
    annotation = record.annotations[0]
    parent = dataset.tasks[0]
    for index in range(2, 7):
        task = AnswerTask(
            inputs=(record,),
            prompt=f"Question {index}?",
            targets=(f"Answer {index}",),
            input_annotations=(annotation,),
            from_tasks=(parent,),
        )
        dataset.add_task(task=task)
        parent = task
    monkeypatch.setattr(control_writer_module, "_TASK_BATCH_SIZE", 2)
    path = tmp_path / "control.duckdb"
    first, *rest = dataset.tasks

    DuckDBControlWriter(path).write_hierarchy(dataset, tasks=(*rest, first))

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM tasks").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_targets").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_record_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_annotation_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_dependencies").fetchone() == (5,)
        assert connection.execute(
            "SELECT task_id, parent_task_id FROM task_dependencies WHERE task_id IN (2, 3) ORDER BY task_id"
        ).fetchall() == [(2, TASK_ID), (3, 2)]


def _numbered(entity, entity_id):
    """Give an object the id a dataset would, for a task stream that reaches the writer directly."""
    object.__setattr__(entity, "_id", entity_id)
    return entity


def _child(**fields) -> AnswerTask:
    return _numbered(AnswerTask(**fields), 302)


def _task_on_an_unwritten_parent(task: AnswerTask) -> AnswerTask:
    outside = _numbered(AnswerTask(inputs=task.inputs, targets=("outside",)), 399)
    return _child(inputs=task.inputs, targets=("child",), from_tasks=(outside,))


def _task_on_an_unwritten_occurrence(task: AnswerTask) -> AnswerTask:
    outside = Annotation(key="outside", value="x")
    return _child(inputs=task.inputs, targets=("child",), input_annotations=(outside,))


def _task_on_itself(task: AnswerTask) -> AnswerTask:
    child = _child(inputs=task.inputs, targets=("child",))
    child.from_tasks = (child,)
    return child


def _task_on_an_unknown_input_record(task: AnswerTask) -> AnswerTask:
    return _child(inputs=(_numbered(Record(), 99),), prompt="q", targets=("x",))


def _task_with_an_unknown_target_record(task: AnswerTask) -> AnswerTask:
    return _child(inputs=task.inputs, targets=(_numbered(Record(), 99),))


def _unknown_signal() -> Signal:
    return _numbered(Signal(name="I", spec=SPEC, time_axis=RegularAxis(period_us=Fraction(2_000)), data=[1.0]), 999)


def _task_with_a_step_on_an_unknown_signal(task: AnswerTask) -> AnswerTask:
    return _child(inputs=task.inputs, targets=(StepPoint(start=1, time_series=_unknown_signal()),))


def _task_with_a_span_on_an_unknown_signal(task: AnswerTask) -> AnswerTask:
    span = TimeInterval(start_us=0, end_us=2000, time_series=(_unknown_signal(),))
    return _child(inputs=task.inputs, targets=(span,))


@pytest.mark.parametrize(
    ("make_task", "message"),
    [
        (_task_on_an_unwritten_parent, "task 302 depends on task 399, which is not in the dataset"),
        (
            _task_on_an_unwritten_occurrence,
            "task 302 field 'input_annotations' refers to unknown annotation occurrence",
        ),
        (_task_on_itself, "task 302 depends on itself"),
        (_task_on_an_unknown_input_record, "task 302 field 'inputs' refers to unknown record 99"),
        (_task_with_an_unknown_target_record, "task 302 target 0 refers to unknown Record 99"),
        (_task_with_a_step_on_an_unknown_signal, "task 302 target 0 refers to unknown Signal 999"),
        (_task_with_a_span_on_an_unknown_signal, r"task 302 target 0 refers to unknown Signals \[999\]"),
    ],
)
def test_control_writer_rejects_task_references_outside_the_dataset(tmp_path, make_task, message):
    dataset = _dataset()
    (task,) = dataset.tasks
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=message):
        DuckDBControlWriter(path).write_hierarchy(dataset, tasks=(task, make_task(task)))

    assert not path.exists()


def test_control_writer_output_passes_the_relational_audit(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)


def test_control_writer_rejects_a_cross_record_parent_before_publication(tmp_path, monkeypatch):
    original = DuckDBControlWriter._write_objects

    def write_corrupt_relationship(writer, connection, dataset, tasks):
        task_counts = original(writer, connection, dataset, tasks)
        connection.execute("INSERT INTO records (record_id, clock_id, metadata) VALUES (99, 1, '{}')")
        connection.execute("UPDATE sources SET record_id = 99 WHERE source_id = ?", [MONITOR_ID])
        return task_counts

    monkeypatch.setattr(DuckDBControlWriter, "_write_objects", write_corrupt_relationship)
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="belong to different records"):
        DuckDBControlWriter(path).write_hierarchy(_dataset())

    assert not path.exists()


def test_control_writer_rejects_an_irregular_axis_with_the_wrong_length(tmp_path):
    axis = IrregularAxis(first_us=0, last_us=2_000)
    signal = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_offsets_loader=lambda: pa.array([0, 1_000, 2_000], type=pa.int64()),
    )
    dataset = TimeFDataset(metadata=_dataset().metadata)
    dataset.add_record(record=Record(sources=(Source(name="ECG", signals=(signal,)),)))

    with pytest.raises(TimeFValidationError, match="returned 3 time offsets but it declares n_values=2"):
        DuckDBControlWriter(tmp_path / "control.duckdb").write_hierarchy(dataset)


@pytest.mark.parametrize(
    ("placements", "message"),
    [
        ({}, f"signal {LEAD_I_ID} declares 2 values but its chunks store 0"),
        (
            {(LEAD_I_ID, 0): SimpleNamespace(n_values=1), (LEAD_II_ID, 0): SimpleNamespace(n_values=2)},
            f"signal {LEAD_I_ID} declares 2 values but its chunks store 1",
        ),
        ({(3, 0): SimpleNamespace(n_values=2)}, "refers to unknown signal 3"),
        ({(LEAD_I_ID, 0): SimpleNamespace(n_values=0)}, f"chunk 0 of signal {LEAD_I_ID} is empty"),
    ],
)
def test_control_writer_rejects_chunks_that_do_not_cover_each_signal(tmp_path, placements, message):
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=message):
        DuckDBControlWriter(path).write_hierarchy(_dataset(), placements)

    assert not path.exists()


def test_control_writer_stores_value_chunk_locations(tmp_path):
    path = tmp_path / "control.duckdb"
    placement = SimpleNamespace(
        chunk_file="values/part-00000000.parquet",
        data_index=SimpleNamespace(major_idx=2, minor_idx=3),
        spec_type="voltage",
        signal="I",
        n_values=2,
    )
    placements = {
        (LEAD_I_ID, 0): placement,
        (LEAD_II_ID, 0): SimpleNamespace(
            chunk_file=placement.chunk_file,
            data_index=SimpleNamespace(major_idx=2, minor_idx=4),
            spec_type="voltage",
            signal="II",
            n_values=2,
        ),
    }

    DuckDBControlWriter(path).write_hierarchy(_dataset(), placements)  # ty: ignore[invalid-argument-type]

    with connect_control(path, read_only=True) as connection:
        assert connection.execute(
            """SELECT value_path, chunk_major_index, chunk_minor_index, signal_chunks.n_values
               FROM signal_chunks
               WHERE signal_id = ?""",
            [LEAD_I_ID],
        ).fetchone() == ("values/part-00000000.parquet", 2, 3, 2)
