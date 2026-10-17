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
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(
    spec_type="voltage",
    name="Voltage",
    unit_value=ureg.millivolt,
    dtype="float32",
)
RECORD_ID = 10
MONITOR_ID = 20
ECG_ID = 21
LEAD_I_ID = 101
LEAD_II_ID = 102
SEX_MALE_ID = 201
TASK_ID = 301


def _dataset() -> TimeFDataset:
    axis = RegularAxis(id=1, period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        id=LEAD_I_ID,
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    lead_ii = Signal.from_loader(
        id=LEAD_II_ID,
        name="II",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([3.0, 4.0], type=pa.float32()),
    )
    ecg = Source(id=ECG_ID, name="ECG", signals=(lead_i, lead_ii))
    monitor = Source(id=MONITOR_ID, name="Monitor", sources=(ecg,))
    record = Record(id=RECORD_ID, sources=(monitor,))
    record.annotate(Annotation(id=SEX_MALE_ID, key="patient_sex", value="male"))
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
        id=TASK_ID,
        inputs=(record,),
        prompt="Alive?",
        targets=("Yes",),
        input_annotations=(record.annotations[0],),
    )
    task.annotate(Annotation(id=202, key="task_kind", value="diagnosis"))
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
        assert connection.execute(
            """SELECT parent.source_id
               FROM sources child
               JOIN sources parent ON parent.source_key = child.parent_source_key
               WHERE child.source_id = ?""",
            [ECG_ID],
        ).fetchone() == (MONITOR_ID,)
        assert connection.execute(
            """SELECT contents.content_id, occurrences.object_type, records.record_id
               FROM annotation_occurrences occurrences
               JOIN annotation_contents contents USING (content_id)
               JOIN records ON records.record_id = occurrences.object_id
               WHERE contents.content_id = ? AND occurrences.object_type = 'Record'""",
            [SEX_MALE_ID],
        ).fetchone() == (SEX_MALE_ID, "Record", RECORD_ID)


def test_control_relationships_use_integer_keys(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    relationship_columns = (
        ("sources", "record_key"),
        ("sources", "parent_source_key"),
        ("signals", "source_key"),
        ("signals", "axis_key"),
        ("annotation_occurrences", "content_id"),
        ("annotation_occurrences", "object_id"),
        ("task_targets", "task_id"),
        ("task_targets", "record_id"),
        ("task_targets", "signal_id"),
        ("task_record_refs", "task_id"),
        ("task_record_refs", "record_id"),
        ("task_annotation_refs", "occurrence_id"),
        ("task_dependencies", "parent_task_id"),
    )
    with connect_control(path, read_only=True) as connection:
        for table, column in relationship_columns:
            row = connection.execute(
                """SELECT data_type
                   FROM information_schema.columns
                   WHERE table_name = ? AND column_name = ?""",
                [table, column],
            ).fetchone()
            assert row is not None
            (stored_type,) = row
            assert stored_type == "BIGINT", f"{table}.{column} stores {stored_type}"


def test_control_writer_preserves_relationships_across_task_batches(tmp_path, monkeypatch):
    dataset = _dataset()
    record = dataset.records[0]
    annotation = record.annotations[0]
    parent = dataset.tasks[0]
    for index in range(2, 7):
        task = AnswerTask(
            id=300 + index,
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

    DuckDBControlWriter(path).write_hierarchy(dataset)

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM tasks").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_targets").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_record_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_annotation_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_dependencies").fetchone() == (5,)
        assert connection.execute("SELECT parent_task_id FROM task_dependencies WHERE task_id = 303").fetchone() == (
            302,
        )


def test_control_writer_removes_database_after_validation_failure(tmp_path):
    dataset = _dataset()
    annotation = dataset.records[0].annotations[0]
    dataset.records[0].annotate(
        Annotation(
            id=annotation.id,
            key="patient_sex",
            value="female",
        )
    )
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="reused with different content"):
        DuckDBControlWriter(path).write_hierarchy(dataset)

    assert not path.exists()


def test_control_writer_accepts_a_parent_task_from_a_later_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer_module, "_TASK_BATCH_SIZE", 1)
    dataset = _dataset()
    (task,) = dataset.tasks
    parent = AnswerTask(id=399, inputs=task.inputs, targets=("parent",))
    child = AnswerTask(id=302, inputs=task.inputs, targets=("child",), from_tasks=(parent,))
    path = tmp_path / "control.duckdb"

    DuckDBControlWriter(path).write_hierarchy(dataset, tasks=(task, child, parent))

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT task_id, position, parent_task_id FROM task_dependencies").fetchall() == [
            (302, 0, 399)
        ]


def _task_on_an_unwritten_parent(task: AnswerTask) -> AnswerTask:
    outside = AnswerTask(id=399, inputs=task.inputs, targets=("outside",))
    return AnswerTask(id=302, inputs=task.inputs, targets=("child",), from_tasks=(outside,))


def _task_on_an_unwritten_occurrence(task: AnswerTask) -> AnswerTask:
    outside = Record(id=99).annotate(Annotation(id=299, key="outside", value="x"))
    return AnswerTask(id=302, inputs=task.inputs, targets=("child",), input_annotations=(outside,))


def _task_on_itself(task: AnswerTask) -> AnswerTask:
    child = AnswerTask(id=302, inputs=task.inputs, targets=("child",))
    child.from_tasks = (child,)
    return child


@pytest.mark.parametrize(
    ("make_task", "message"),
    [
        (_task_on_an_unwritten_parent, "task 302 depends on task 399, which is not in the dataset"),
        (
            _task_on_an_unwritten_occurrence,
            "task 302 field 'input_annotations' refers to unknown annotation occurrence",
        ),
        (_task_on_itself, "task 302 depends on itself"),
    ],
)
def test_control_writer_rejects_task_references_to_rows_it_did_not_write(tmp_path, make_task, message):
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
        connection.execute("INSERT INTO records (record_key, record_id, clock_id, metadata) VALUES (-1, 99, 1, '{}')")
        connection.execute("UPDATE sources SET record_key = -1 WHERE source_id = ?", [MONITOR_ID])
        return task_counts

    monkeypatch.setattr(DuckDBControlWriter, "_write_objects", write_corrupt_relationship)
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="belong to different records"):
        DuckDBControlWriter(path).write_hierarchy(_dataset())

    assert not path.exists()


def test_control_writer_rejects_ids_reused_across_records(tmp_path):
    axis = RegularAxis(id=1, period_us=Fraction(2_000))

    def signal(signal_id: int) -> Signal:
        return Signal.from_loader(
            id=signal_id,
            name="I",
            spec=SPEC,
            time_axis=axis,
            n_values=2,
            loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        )

    dataset = SimpleNamespace(
        metadata=SimpleNamespace(dataset_id="org/dupes"),
        annotations=(),
        records=(
            Record(id=10, sources=(Source(id=ECG_ID, name="ECG", signals=(signal(LEAD_I_ID),)),)),
            Record(id=11, sources=(Source(id=ECG_ID, name="ECG", signals=(signal(LEAD_II_ID),)),)),
        ),
        iter_tasks=lambda: iter(()),
    )

    with pytest.raises(TimeFValidationError, match=f"source id {ECG_ID} is not unique"):
        DuckDBControlWriter(tmp_path / "control.duckdb").write_hierarchy(dataset)  # ty: ignore[invalid-argument-type]


def test_control_writer_rejects_an_irregular_axis_with_the_wrong_length(tmp_path):
    axis = IrregularAxis(id=2, first_us=0, last_us=2_000)
    signal = Signal.from_loader(
        id=LEAD_I_ID,
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
        time_offsets_loader=lambda: pa.array([0, 1_000, 2_000], type=pa.int64()),
    )
    dataset = SimpleNamespace(
        metadata=SimpleNamespace(dataset_id="org/irregular"),
        annotations=(),
        records=(Record(id=RECORD_ID, sources=(Source(id=ECG_ID, name="ECG", signals=(signal,)),)),),
        iter_tasks=lambda: iter(()),
    )

    with pytest.raises(TimeFValidationError, match="returned 3 time offsets but it declares n_values=2"):
        DuckDBControlWriter(tmp_path / "control.duckdb").write_hierarchy(dataset)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("placements", "message"),
    [
        ({}, f"signal {LEAD_I_ID} declares 2 values but its chunks store 0"),
        (
            {(LEAD_I_ID, 0): SimpleNamespace(n_values=1), (LEAD_II_ID, 0): SimpleNamespace(n_values=2)},
            f"signal {LEAD_I_ID} declares 2 values but its chunks store 1",
        ),
        ({(103, 0): SimpleNamespace(n_values=2)}, "refers to unknown signal 103"),
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
               FROM signal_chunks JOIN signals USING (signal_key)
               WHERE signal_id = ?""",
            [LEAD_I_ID],
        ).fetchone() == ("values/part-00000000.parquet", 2, 3, 2)
