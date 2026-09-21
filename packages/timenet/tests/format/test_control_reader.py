import pyarrow as pa
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.testing import make_dataset
from timenet.types import TimePoint

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
