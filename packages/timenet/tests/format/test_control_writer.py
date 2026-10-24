from fractions import Fraction
import math
from types import SimpleNamespace

import pyarrow as pa
import pytest

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
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
# These are the ids that the dataset and the writer give to the objects of _dataset(). The ids of each
# kind start at 1. The writer numbers Sources and annotation contents in the order that it writes them.
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
            "SELECT position, target_kind, text_value, integer_value, record_key FROM task_targets"
        ).fetchone() == (0, "text", "Yes", None, None)
        assert connection.execute(
            """SELECT refs.field, records.record_id
               FROM task_record_refs refs JOIN records USING (record_key)"""
        ).fetchone() == (
            "inputs",
            RECORD_ID,
        )
        assert connection.execute("SELECT typeof(record_key) FROM task_record_refs").fetchone() == ("BIGINT",)
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
               WHERE child.task_id = 3"""
        ).fetchone() == (2, 3)


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
               FROM signal_chunks JOIN signals USING (signal_key)
               WHERE signal_id = ?""",
            [LEAD_I_ID],
        ).fetchone() == ("values/part-00000000.parquet", 2, 3, 2)


def test_control_writer_stores_equal_contents_and_regular_and_ordinal_axes_once(tmp_path):
    dataset = _dataset()
    for value in (0.0, -0.0, 1, True, float("nan"), -math.nan):
        dataset.records[0].annotate(Annotation(key="probe", value=value))
    irregular = tuple(
        Signal.from_irregular([1.0, 2.0, 3.0], time_offsets_us=offsets, spec=SPEC, name="x")
        for offsets in ([0, 100, 1_000], [0, 900, 1_000])
    )
    regular = Signal.from_loader(
        name="III",
        spec=SPEC,
        time_axis=RegularAxis(period_us=Fraction(2_000)),
        n_values=2,
        loader=lambda: pa.array([5.0, 6.0], type=pa.float32()),
    )
    ordinal = tuple(Signal.from_values([1.0, 2.0], spec=SPEC, name="o", time_axis=OrdinalAxis()) for _ in range(2))
    other = Record(sources=(Source(name="ECG", signals=(regular, *irregular, *ordinal)),))
    other.annotate(Annotation(key="patient_sex", value="male"))
    dataset.add_record(record=other)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with connect_control(path, read_only=True) as connection:
        contents = connection.execute(
            "SELECT name, count(*) FROM annotation_contents GROUP BY name ORDER BY name"
        ).fetchall()
        axes = connection.execute(
            "SELECT axis_type, count(*) FROM axes GROUP BY axis_type ORDER BY axis_type"
        ).fetchall()
    # The two NaNs share a row. 0.0 and -0.0 get separate rows, and so do True and 1.
    # A shared row makes one value read back as the other.
    assert contents == [("patient_sex", 1), ("probe", 5), ("task_kind", 1)]
    # Equal endpoints do not mean equal offsets, so the two irregular axes keep their own rows.
    assert axes == [("irregular", 2), ("ordinal", 1), ("regular", 1)]
