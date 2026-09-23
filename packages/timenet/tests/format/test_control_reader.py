import pyarrow as pa
import pytest

from timenet.dataset import Record
from timenet.errors import TimeFFormatError, TimeFValidationError
import timenet.format.control_reader as control_reader_module
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.testing import make_dataset
from timenet.types import Annotation, TimePoint

from .test_control_writer import _dataset


def test_control_reader_hydrates_recursive_hierarchy_and_annotations(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    DuckDBControlWriter(path).write_hierarchy(dataset)
    values = {
        "lead-i": pa.array([1.0, 2.0], type=pa.float32()),
        "lead-ii": pa.array([3.0, 4.0], type=pa.float32()),
    }

    with DuckDBControlReader(path, value_loader=lambda signal_id, _spec: values[signal_id]) as reader:
        (record,) = reader.read_records()
        (task,) = reader.read_tasks((record,))

    assert record.id == "record-1"
    assert record.sources[0].id == "monitor"
    assert record.sources[0].sources[0].id == "ecg"
    lead_i, lead_ii = record.signals
    assert lead_i.time_axis is lead_ii.time_axis
    assert lead_i.spec is lead_ii.spec
    assert lead_i.metadata == lead_ii.metadata == {}
    assert lead_i.metadata is not lead_ii.metadata
    assert lead_i.to_arrow().to_pylist() == [1.0, 2.0]
    assert record.annotations[0].content_id == "sex-male"
    assert record.annotations[0].occurrence_id is not None
    assert task.inputs[0] is record
    assert task.prompt == "Alive?"
    assert task.targets == ("Yes",)
    assert task.annotations[0].name == "task_kind"


def test_control_reader_hydrates_each_target_storage_type(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    signal = record.signals[0]
    point = TimePoint.seconds(0.001, time_series_ids=(signal.id,))
    dataset.tasks[0].targets = ("Yes", 7, 0.5, True, record, signal, point)
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)

    assert task.targets is not None
    text, integer, floating, boolean, target_record, target_signal, target_point = task.targets
    assert (text, integer, floating, boolean) == ("Yes", 7, 0.5, True)
    assert target_record is records[0]
    assert target_signal is records[0].signals[0]
    assert target_point == point


def test_control_reader_preserves_ordered_integer_relationships(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    record = dataset.records[0]
    second = record.annotate(Annotation(id="age-65", key="patient_age", value=65, unit="year"))
    dataset.tasks[0].input_annotations = (second, record.annotations[0])
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        (task,) = reader.read_tasks(records)

    assert task.inputs == records
    assert tuple(annotation.content_id for annotation in task.input_annotations) == ("age-65", "sex-male")


def test_text_targets_do_not_build_a_signal_index(tmp_path, monkeypatch):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        monkeypatch.setattr(
            Record,
            "signals",
            property(lambda _record: pytest.fail("text targets walked the Signal hierarchy")),
        )
        (task,) = reader.read_tasks(records)

    assert task.targets == ("Yes",)


def test_selected_record_hydration_does_not_read_unrelated_hierarchies(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(make_dataset())
    with connect_control(path) as connection:
        connection.execute(
            """UPDATE axes SET axis_type = 'corrupt'
               WHERE axis_key IN (
                   SELECT signals.axis_key
                   FROM signals
                   JOIN sources USING (source_key)
                   JOIN records USING (record_key)
                   WHERE records.record_id = 'record-1'
               )"""
        )
    with DuckDBControlReader(path) as reader:
        (selected,) = reader.read_records(("record-0",))
        assert selected.id == "record-0"
        with pytest.raises(TimeFFormatError, match="unknown type"):
            reader.read_records(("record-1",))


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "DELETE FROM annotation_contents WHERE content_id = 'sex-male'",
            "missing content",
        ),
        (
            "UPDATE sources SET parent_source_key = 999 WHERE source_id = 'ecg'",
            "missing parent",
        ),
        (
            "UPDATE signals SET source_key = 999 WHERE signal_id = 'lead-i'",
            "missing source",
        ),
        (
            "UPDATE annotation_occurrences SET object_key = 999 "
            "WHERE content_key IN ("
            "SELECT content_key FROM annotation_contents WHERE content_id = 'sex-male')",
            "missing Record",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Unknown' "
            "WHERE content_key IN ("
            "SELECT content_key FROM annotation_contents WHERE content_id = 'sex-male')",
            "unknown object type",
        ),
    ],
)
def test_control_reader_rejects_corrupt_relationships(tmp_path, statement, message):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute(statement)

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFFormatError, match=message):
        reader.read_records()


def test_iter_tasks_for_one_record_yields_its_tasks_and_shares_record_objects(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(make_dataset())

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        record0 = next(record for record in records if record.record_id == "record-0")
        tasks = list(reader.iter_tasks((record0,)))

    assert [task.id for task in tasks] == ["task-answer-0", "task-cls-0", "task-localize-0", "task-scalar-0"]
    assert all(task.inputs[0] is record0 for task in tasks)
    answer = tasks[0]
    assert answer.from_tasks == (tasks[1],)
    assert answer.from_tasks[0] is tasks[1]


def test_iter_tasks_hydrates_a_parent_that_sorts_after_its_dependant(tmp_path, monkeypatch):
    monkeypatch.setattr(control_reader_module, "_TASK_BATCH_SIZE", 1)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(make_dataset())

    with DuckDBControlReader(path) as reader:
        records = reader.read_records()
        tasks = list(reader.iter_tasks(records))

    by_id = {task.id: task for task in tasks}
    assert by_id["task-answer-0"].from_tasks[0] is by_id["task-cls-0"]
    assert tuple(sorted(by_id)) == tuple(task.id for task in tasks)


def test_read_tasks_rejects_records_the_database_does_not_store(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())

    with DuckDBControlReader(path) as reader, pytest.raises(TimeFValidationError, match="no such record"):
        reader.read_tasks((Record(record_id="ghost"),))


def test_typed_scope_and_payload_columns_round_trip(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        restored = {task.id: task for task in reader.read_tasks(reader.read_records())}

    for task in dataset.tasks:
        back = restored[task.id]
        assert type(back) is type(task)
        assert back.scope == task.scope
        for field in ("target_schema", "unit", "target_name", "mode"):
            if hasattr(task, field):
                assert getattr(back, field) == getattr(task, field), field


@pytest.mark.parametrize(
    ("value", "kind"),
    [("W", "text"), (64, "integer"), (0.5, "float"), (True, "boolean"), (["yes", "no"], "text_list"), (None, None)],
)
def test_annotation_values_round_trip_in_their_own_typed_column(tmp_path, value, kind):
    dataset = _dataset()
    record = dataset.records[0]
    annotation = Annotation(
        key="probe", value=value, id="probe-content", span=None if value is not None else TimePoint(start_us=0)
    )
    record.annotate(annotation)
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        (restored,) = reader.read_records((record.record_id,))
    back = next(item for item in restored.annotations if item.id == "probe-content")
    stored = (
        connect_control(path, read_only=True)
        .execute("SELECT value_kind FROM annotation_contents WHERE content_id = 'probe-content'")
        .fetchone()
    )

    assert back.value == value
    assert type(back.value) is type(value)
    assert stored == (kind,)


def test_arrow_task_views_match_the_hydrated_objects(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(dataset)

    with DuckDBControlReader(path) as reader:
        tasks = {task.id: task for task in reader.read_tasks(reader.read_records())}
        table = reader.task_table().to_pylist()
        targets = reader.target_table().to_pylist()
        subset = reader.task_table(record_ids=("record-0",)).to_pylist()
        annotations = reader.annotation_table(object_type="Record").to_pylist()

    assert [row["task_id"] for row in table] == sorted(tasks)
    answer = next(row for row in table if row["task_id"] == "task-answer-0")
    assert answer["task_type"] == "answer"
    assert answer["input_record_ids"] == [record.id for record in tasks["task-answer-0"].inputs]
    assert answer["from_task_ids"] == ["task-cls-0"]
    assert answer["scope_type"] is None
    assert {row["task_id"] for row in subset} == {task.id for task in tasks.values() if task.inputs[0].id == "record-0"}
    assert sum(1 for row in targets if row["task_id"] == "task-cls-0") == len(tasks["task-cls-0"].targets or ())
    text_targets = {row["task_id"]: row["text_value"] for row in targets if row["target_kind"] == "text"}
    assert text_targets["task-cls-0"] == "normal"
    age = next(row for row in annotations if row["key"] == "age" and row["object_id"] == "record-0")
    assert (age["value_kind"], age["integer_value"], age["unit"]) == ("integer", 64, "years")
