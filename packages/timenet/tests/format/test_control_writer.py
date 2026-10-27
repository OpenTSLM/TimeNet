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
from timenet.format.storage import StorageSources
from timenet.format.stored_values import PackedValueWriter
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    StepPoint,
    TimeInterval,
    TimeSeriesSpec,
    Version,
    read_value,
    ureg,
)


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


def _dataset(*, with_task: bool = True) -> TimeFDataset:
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
    if not with_task:
        return dataset
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
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(_dataset())

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

    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(
        dataset, tasks=(*rest, first)
    )

    with connect_control(path, read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM tasks").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_targets").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_record_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_annotation_refs").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM task_dependencies").fetchone() == (5,)
        assert connection.execute(
            "SELECT task_id, parent_task_id FROM task_dependencies WHERE task_id IN (2, 3) ORDER BY task_id"
        ).fetchall() == [(2, TASK_ID), (3, 2)]


def _foreign_dataset() -> TimeFDataset:
    """Return a dataset whose record 2, signal 3 and task 2 have ids that _dataset() does not use."""
    foreign = _dataset()
    signal = Signal(name="III", spec=SPEC, time_axis=RegularAxis(period_us=Fraction(2_000)), data=[1.0])
    record = foreign.add_record(record=Record(sources=(Source(name="ECG", signals=(signal,)),)))
    foreign.add_task(task=AnswerTask(inputs=(record,), targets=("No",)))
    return foreign


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "from_tasks",
            lambda task, foreign: (foreign.tasks[1],),
            "task 1 depends on a task with id 2 that is not in the dataset",
        ),
        (
            "from_tasks",
            lambda task, foreign: (foreign.tasks[0],),
            "task 1 depends on a task with id 1 that is not in the dataset",
        ),
        (
            "from_tasks",
            lambda task, foreign: (AnswerTask(inputs=task.inputs, targets=("No",)),),
            "task 1 depends on a task that is not in the dataset",
        ),
        ("from_tasks", lambda task, foreign: (task,), "task 1 depends on itself"),
        (
            "input_annotations",
            lambda task, foreign: (Annotation(key="outside", value="x"),),
            "task 1 field 'input_annotations' refers to an annotation that is not in the dataset",
        ),
    ],
)
def test_control_writer_rejects_task_references_outside_the_dataset(tmp_path, field, value, message):
    dataset = _dataset()
    (task,) = dataset.tasks
    # add_task checks references when it registers a task, so a later change reaches the writer.
    setattr(task, field, value(task, _foreign_dataset()))
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=message):
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

    assert not path.exists()


# Record 1 and Signal 1 of the foreign dataset have the same ids as objects of _dataset(). Record 2
# and Signal 3 have ids that _dataset() does not use.
@pytest.mark.parametrize("record_index", [0, 1], ids=["same-ids", "new-ids"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inputs", lambda record: (record,), "references a Record that is not registered"),
        ("targets", lambda record: (record,), "references a Record that is not registered"),
        ("targets", lambda record: (record.signals[0],), "references Signal '.+', which is not registered"),
        (
            "targets",
            lambda record: (StepPoint(start=1, time_series=record.signals[0]),),
            "references Signal '.+', which is not registered",
        ),
        (
            "targets",
            lambda record: (TimeInterval(start_us=0, end_us=2_000, time_series=(record.signals[0],)),),
            "references Signal '.+', which is not registered",
        ),
        (
            "scope",
            lambda record: TimeInterval(start_us=0, end_us=2_000, time_series=(record.signals[0],)),
            "references Signal '.+', which is not registered",
        ),
    ],
)
def test_control_writer_rejects_records_and_signals_of_another_dataset(tmp_path, record_index, field, value, message):
    dataset = _dataset()
    (task,) = dataset.tasks
    setattr(task, field, value(_foreign_dataset().records[record_index]))
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=message):
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

    assert not path.exists()


def test_control_writer_rejects_a_target_signal_that_left_the_hierarchy(tmp_path):
    dataset = _dataset()
    (task,) = dataset.tasks
    (record,) = dataset.records
    ecg = record.sources[0].sources[0]
    lead_i, lead_ii = ecg.signals
    task.targets = (lead_ii,)
    ecg.signals = (lead_i,)
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=f"task 1 target 0 refers to unknown Signal {LEAD_II_ID}"):
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

    assert not path.exists()


def test_control_writer_rejects_two_tasks_with_one_id(tmp_path):
    dataset = _dataset()
    (task,) = dataset.tasks
    twin = dataset.add_task(task=AnswerTask(inputs=task.inputs, targets=("No",)))
    twin._id = task.id
    path = tmp_path / "control.duckdb"

    with pytest.raises(TimeFValidationError, match=f"task id {TASK_ID} is not unique"):
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

    assert not path.exists()


def test_a_streamed_parent_from_another_dataset_is_rejected_when_its_id_streams_later(tmp_path):
    dataset = _dataset(with_task=False)
    record = dataset.records[0]
    foreign_parent = _foreign_dataset().tasks[1]

    def stream():
        yield AnswerTask(inputs=(record,), targets=("Yes",), from_tasks=(foreign_parent,))
        yield AnswerTask(inputs=(record,), targets=("No",))

    dataset.set_task_stream([AnswerTask], stream)

    with pytest.raises(TimeFValidationError, match="task 1 depends on a task with id 2 that is not in the dataset"):
        dataset.write(path=tmp_path)


def test_a_streamed_parent_that_streams_after_its_child_is_stored(tmp_path):
    dataset = _dataset(with_task=False)
    record = dataset.records[0]
    parent = AnswerTask(inputs=(record,), prompt="Parent?", targets=("No",))
    child = AnswerTask(inputs=(record,), prompt="Child?", targets=("Yes",), from_tasks=(parent,))
    dataset.set_task_stream([AnswerTask], lambda: iter((child, parent)))

    restored = TimeFDataset.open(path=dataset.write(path=tmp_path))

    by_prompt = {read_value(task.prompt): task for task in restored.tasks}
    assert by_prompt["Child?"].from_tasks == (by_prompt["Parent?"],)


def test_control_writer_output_passes_the_relational_audit(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(_dataset())

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
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(_dataset())

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
        DuckDBControlWriter(
            tmp_path / "control.duckdb", external=PackedValueWriter(tmp_path, StorageSources())
        ).write_hierarchy(dataset)


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
        DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(
            _dataset(), placements
        )

    assert not path.exists()


def test_control_writer_stores_value_chunk_locations(tmp_path, monkeypatch):
    # One row per batch, so every chunk row is written by its own batch.
    monkeypatch.setattr(control_writer_module, "_ROW_BATCH_SIZE", 1)
    path = tmp_path / "control.duckdb"
    placement = SimpleNamespace(
        chunk_file="values/part-00000000.parquet",
        data_index=SimpleNamespace(major_idx=2, minor_idx=3),
        spec_type="voltage",
        signal="I",
        n_values=2,
        source_key=None,
    )
    placements = {
        (LEAD_I_ID, 0): placement,
        (LEAD_II_ID, 0): SimpleNamespace(
            chunk_file=placement.chunk_file,
            data_index=SimpleNamespace(major_idx=2, minor_idx=4),
            spec_type="voltage",
            signal="II",
            n_values=2,
            source_key=None,
        ),
    }

    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(
        _dataset(),
        placements,  # ty: ignore[invalid-argument-type]
    )

    with connect_control(path, read_only=True) as connection:
        assert connection.execute(
            """SELECT signal_id, value_path, chunk_major_index, chunk_minor_index, signal_chunks.n_values
               FROM signal_chunks
               ORDER BY signal_id"""
        ).fetchall() == [
            (LEAD_I_ID, "values/part-00000000.parquet", 2, 3, 2),
            (LEAD_II_ID, "values/part-00000000.parquet", 2, 4, 2),
        ]


def test_control_writer_stores_equal_text_contents_once(tmp_path):
    dataset = _dataset(with_task=False)
    signal = Signal.from_values([1.0, 2.0], spec=SPEC, name="o", time_axis=OrdinalAxis())
    other = Record(sources=(Source(name="ECG", signals=(signal,)),))
    dataset.add_record(record=other)
    for record, labels in zip(dataset.records, (["a", "b"], ("a", "b")), strict=True):
        record.annotate(Annotation(key="labels", value=labels))
        record.annotate(Annotation(key="note", value="same"))
    other.annotate(Annotation(key="labels", value=["a", "c"]))
    signal.annotate(Annotation(key="note", value="same"))
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

    with connect_control(path, read_only=True) as connection:
        rows = connection.execute(
            """SELECT name, count(DISTINCT content_id), count(*)
               FROM annotation_occurrences JOIN annotation_contents USING (content_id)
               WHERE name IN ('labels', 'note')
               GROUP BY name ORDER BY name"""
        ).fetchall()
    # Equal text, given as a list or a tuple, shares one row. Different text keeps its own.
    assert rows == [("labels", 2, 3), ("note", 1, 3)]


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
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)

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
