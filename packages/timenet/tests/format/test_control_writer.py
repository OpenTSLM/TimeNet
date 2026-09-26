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
from timenet.types import Annotation, AnswerTask, DatasetMetadata, InputModality, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(
    spec_type="voltage",
    name="Voltage",
    unit_value=ureg.millivolt,
    dtype="float32",
)


def _dataset() -> TimeFDataset:
    axis = RegularAxis(axis_id="axis-1", period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        id="lead-i",
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    lead_ii = Signal.from_loader(
        id="lead-ii",
        name="II",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([3.0, 4.0], type=pa.float32()),
    )
    ecg = Source(id="ecg", name="ECG", signals=(lead_i, lead_ii))
    monitor = Source(id="monitor", name="Monitor", sources=(ecg,))
    record = Record(record_id="record-1", sources=(monitor,))
    record.annotate(Annotation(id="sex-male", key="patient_sex", value="male"))
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
        input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
        id="task-1",
        inputs=(record,),
        prompt="Alive?",
        targets=("Yes",),
        input_annotations=(record.annotations[0],),
    )
    task.annotate(Annotation(id="task-kind", key="task_kind", value="diagnosis"))
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
            "SELECT position, target_kind, text_value, integer_value, record_key FROM task_targets"
        ).fetchone() == (0, "text", "Yes", None, None)
        assert connection.execute(
            """SELECT refs.field, records.record_id
               FROM task_record_refs refs JOIN records USING (record_key)"""
        ).fetchone() == (
            "inputs",
            "record-1",
        )
        assert connection.execute("SELECT typeof(record_key) FROM task_record_refs").fetchone() == ("BIGINT",)
        assert connection.execute(
            """SELECT parent.source_id
               FROM sources child
               JOIN sources parent ON parent.source_key = child.parent_source_key
               WHERE child.source_id = 'ecg'"""
        ).fetchone() == ("monitor",)
        assert connection.execute(
            """SELECT contents.content_id, occurrences.object_type, records.record_id
               FROM annotation_occurrences occurrences
               JOIN annotation_contents contents USING (content_key)
               JOIN records ON records.record_key = occurrences.object_key
               WHERE contents.content_id = 'sex-male' AND occurrences.object_type = 'Record'"""
        ).fetchone() == ("sex-male", "Record", "record-1")


def test_control_relationships_use_integer_keys(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    relationship_columns = (
        ("sources", "record_key"),
        ("sources", "parent_source_key"),
        ("signals", "source_key"),
        ("signals", "axis_key"),
        ("annotation_occurrences", "content_key"),
        ("annotation_occurrences", "object_key"),
        ("task_targets", "task_key"),
        ("task_targets", "record_key"),
        ("task_targets", "signal_key"),
        ("task_record_refs", "task_key"),
        ("task_record_refs", "record_key"),
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
            input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
            id=f"task-{index}",
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
        assert connection.execute(
            """SELECT parent.task_id, child.task_id
               FROM task_dependencies dependencies
               JOIN tasks child ON child.task_key = dependencies.task_key
               JOIN tasks parent ON parent.task_key = dependencies.parent_task_key
               WHERE child.task_id = 'task-3'"""
        ).fetchone() == ("task-2", "task-3")


def test_control_writer_removes_database_after_validation_failure(tmp_path):
    dataset = _dataset()
    annotation = dataset.records[0].annotations[0]
    dataset.records[0].annotate(
        Annotation(
            id=annotation.content_id,
            key="patient_sex",
            value="female",
        )
    )
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="reused with different content"):
        DuckDBControlWriter(path).write_hierarchy(dataset)

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
        connection.execute("INSERT INTO records (record_key, record_id, metadata) VALUES (-1, 'ghost', '{}')")
        connection.execute("UPDATE sources SET record_key = -1 WHERE source_id = 'monitor'")
        return task_counts

    monkeypatch.setattr(DuckDBControlWriter, "_write_objects", write_corrupt_relationship)
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match="belong to different records"):
        DuckDBControlWriter(path).write_hierarchy(_dataset())

    assert not path.exists()


def test_control_writer_rejects_ids_reused_across_records(tmp_path):
    axis = RegularAxis(axis_id="axis-1", period_us=Fraction(2_000))

    def signal(signal_id: str) -> Signal:
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
            Record(record_id="record-1", sources=(Source(id="ecg", name="ECG", signals=(signal("lead-i"),)),)),
            Record(record_id="record-2", sources=(Source(id="ecg", name="ECG", signals=(signal("lead-ii"),)),)),
        ),
        iter_tasks=lambda: iter(()),
    )

    with pytest.raises(TimeFValidationError, match="source id 'ecg' is not unique"):
        DuckDBControlWriter(tmp_path / "control.duckdb").write_hierarchy(dataset)  # ty: ignore[invalid-argument-type]


def test_control_writer_rejects_an_irregular_axis_with_the_wrong_length(tmp_path):
    axis = IrregularAxis(axis_id="axis-irregular", first_us=0, last_us=2_000)
    signal = Signal.from_loader(
        id="lead-i",
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
        records=(Record(record_id="record-1", sources=(Source(id="ecg", name="ECG", signals=(signal,)),)),),
        iter_tasks=lambda: iter(()),
    )

    with pytest.raises(TimeFValidationError, match="declares 2 values but its axis 'axis-irregular' has 3"):
        DuckDBControlWriter(tmp_path / "control.duckdb").write_hierarchy(dataset)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("placements", "message"),
    [
        ({}, "signal 'lead-i' declares 2 values but its chunks store 0"),
        (
            {("lead-i", 0): SimpleNamespace(n_values=1), ("lead-ii", 0): SimpleNamespace(n_values=2)},
            "signal 'lead-i' declares 2 values but its chunks store 1",
        ),
        ({("lead-iii", 0): SimpleNamespace(n_values=2)}, "refers to unknown signal 'lead-iii'"),
        ({("lead-i", 0): SimpleNamespace(n_values=0)}, "chunk 0 of signal 'lead-i' is empty"),
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
        ("lead-i", 0): placement,
        ("lead-ii", 0): SimpleNamespace(
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
               WHERE signal_id = 'lead-i'"""
        ).fetchone() == ("values/part-00000000.parquet", 2, 3, 2)
