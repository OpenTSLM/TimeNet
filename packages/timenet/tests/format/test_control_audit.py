from fractions import Fraction

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.control_audit import audit_control_database
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt, dtype="float32")
RECORD_ID = 1
# The writer numbers Sources in the order it writes them: the root first, then its child.
MONITOR_ID = 1
ECG_ID = 2


def _dataset() -> TimeFDataset:
    axis = RegularAxis(period_us=Fraction(2_000))
    lead_i = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=axis,
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    ecg = Source(name="ECG", signals=(lead_i,))
    monitor = Source(name="Monitor", sources=(ecg,))
    record = Record(sources=(monitor,))
    record.annotate(Annotation(key="patient_sex", value="male"))
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/control-audit",
            dataset_version=Version(1, 0, 0),
            name="Control audit",
            description="Test hierarchy",
            license=License.MIT,
        )
    )
    dataset.add_record(record=record)
    dataset.add_task(task=AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",)))
    return dataset


@pytest.fixture
def control_path(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())
    return path


def test_audit_accepts_a_freshly_written_database(control_path):
    with connect_control(control_path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)


def test_audit_requires_chunks_to_cover_each_signal(control_path):
    with connect_control(control_path) as connection:
        with pytest.raises(TimeFFormatError, match="value chunks whose lengths do not match"):
            audit_control_database(connection)
        connection.execute("INSERT INTO signal_chunks VALUES (1, 0, 'values.parquet', 0, NULL, 2)")
        audit_control_database(connection)
        connection.execute("UPDATE signal_chunks SET n_values = 1")
        with pytest.raises(TimeFFormatError, match="signal 1 has value chunks whose lengths do not match"):
            audit_control_database(connection)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "INSERT INTO records (record_id, clock_id, metadata) SELECT record_id, clock_id, metadata FROM records",
            "records contains duplicate values for record_id",
        ),
        (
            "UPDATE sources SET record_id = -1 WHERE source_id = 1",
            r"sources\.record_id refers to missing records\.record_id value -1",
        ),
        (
            "UPDATE signals SET source_id = 999",
            r"signals\.source_id refers to missing sources\.source_id value 999",
        ),
        (
            "UPDATE signals SET axis_id = 999",
            r"signals\.axis_id refers to missing axes\.axis_id value 999",
        ),
        (
            "INSERT INTO axis_offsets VALUES (999, 0, 0)",
            r"axis_offsets\.axis_id refers to missing axes\.axis_id value 999",
        ),
        (
            "INSERT INTO signal_chunks VALUES (999, 0, 'values.parquet', 0, NULL, 1)",
            r"signal_chunks\.signal_id refers to missing signals\.signal_id value 999",
        ),
        (
            "UPDATE sources SET parent_source_id = source_id WHERE source_id = 2",
            r"invalid sources\.parent_source_id value",
        ),
        (
            "UPDATE signals SET n_values = 0",
            r"invalid signals\.n_values value",
        ),
        (
            "INSERT INTO records (record_id, clock_id, metadata) VALUES (99, 1, '{}'); "
            "UPDATE sources SET record_id = 99 WHERE source_id = 2",
            f"source {ECG_ID} and its parent belong to different records",
        ),
        (
            "UPDATE sources SET parent_source_id = 2 WHERE source_id = 1",
            "contains a cycle",
        ),
        (
            "INSERT INTO axis_offsets SELECT axis_id, 0, 0 FROM axes",
            "axis length that does not match",
        ),
        (
            "UPDATE records SET record_id = 0; UPDATE sources SET record_id = 0; "
            "UPDATE task_record_refs SET record_id = 0",
            r"invalid records\.record_id value",
        ),
        (
            "UPDATE sources SET source_id = source_id - 100, parent_source_id = parent_source_id - 100; "
            "UPDATE signals SET source_id = source_id - 100",
            r"invalid sources\.source_id value",
        ),
        (
            "UPDATE axes SET axis_id = 0; UPDATE signals SET axis_id = 0",
            r"invalid axes\.axis_id value",
        ),
        ("UPDATE signals SET signal_id = -1", r"invalid signals\.signal_id value"),
        (
            "UPDATE tasks SET task_id = 0; UPDATE task_targets SET task_id = 0; "
            "UPDATE task_record_refs SET task_id = 0",
            r"invalid tasks\.task_id value",
        ),
        ("INSERT INTO tasks SELECT * FROM tasks", "tasks contains duplicate values for task_id"),
        ("UPDATE task_record_refs SET field = 'outputs'", r"invalid task_record_refs\.field value"),
        (
            "INSERT INTO task_dependencies VALUES (1, 0, 1)",
            r"invalid task_dependencies\.parent_task_id value",
        ),
        (
            "INSERT INTO task_dependencies VALUES (1, 0, 999)",
            r"task_dependencies\.parent_task_id refers to missing tasks\.task_id value 999",
        ),
        (
            "UPDATE tasks SET input_modalities = ['text', 'no_input']",
            "task 1 has invalid input_modalities",
        ),
        (
            "UPDATE tasks SET scope_type = 'time_interval', scope_start = 0, scope_end = 2000, "
            "scope_signal_ids = [999]",
            r"tasks\.scope_signal_ids refers to missing signals\.signal_id value 999",
        ),
        (
            "UPDATE task_targets SET target_kind = 'time_interval', text_value = NULL, span_start = 0, "
            "span_end = 2000, span_signal_ids = [999]",
            r"task_targets\.span_signal_ids refers to missing signals\.signal_id value 999",
        ),
        (
            "UPDATE task_targets SET target_kind = 'step_point', text_value = NULL, span_start = 0, signal_id = 999",
            r"task_targets\.signal_id refers to missing signals\.signal_id value 999",
        ),
        (
            "INSERT INTO records (record_id, clock_id, metadata) VALUES (99, 1, '{}'); "
            "UPDATE tasks SET scope_type = 'time_interval', scope_start = 0, scope_end = 2000, "
            "scope_signal_ids = [1]; "
            "UPDATE task_record_refs SET record_id = 99",
            "task 1 time span must resolve to exactly one input Record",
        ),
        (
            "INSERT INTO records (record_id, clock_id, metadata) VALUES (99, 1, '{}'); "
            "UPDATE task_targets SET target_kind = 'time_interval', text_value = NULL, span_start = 0, "
            "span_end = 2000, span_signal_ids = [1]; "
            "UPDATE task_record_refs SET record_id = 99",
            "task 1 time span must resolve to exactly one input Record",
        ),
        (
            "UPDATE annotation_occurrences SET object_id = 999",
            "annotation occurrence .* refers to missing Record 999",
        ),
        (
            # Ids are unique only within one kind. Signal 1 exists, so the update moves the record id past it.
            "UPDATE annotation_occurrences SET object_type = 'Signal', object_id = object_id + 1",
            f"refers to missing Signal {RECORD_ID + 1}",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Dataset'",
            f"on the Dataset has object id {RECORD_ID}",
        ),
        (
            "UPDATE annotation_occurrences SET object_id = NULL",
            "annotation occurrence .* refers to missing Record None",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Unknown'",
            r"invalid annotation_occurrences\.object_type value",
        ),
        (
            "UPDATE annotation_occurrences SET content_id = 999",
            r"annotation_occurrences\.content_id refers to missing annotation_contents\.content_id",
        ),
    ],
)
def test_audit_names_the_first_violated_invariant(control_path, statement, message):
    with connect_control(control_path) as connection:
        for part in statement.split("; "):
            connection.execute(part)
        with pytest.raises(TimeFFormatError, match=message):
            audit_control_database(connection, require_chunks=False)
