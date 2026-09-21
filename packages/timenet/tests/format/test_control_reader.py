import pyarrow as pa
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control

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
    assert lead_i.to_arrow().to_pylist() == [1.0, 2.0]
    assert record.annotations[0].content_id == "sex-male"
    assert record.annotations[0].occurrence_id is not None
    assert task.inputs[0] is record
    assert task.prompt == "Alive?"
    assert task.target == "Yes"
    assert task.annotations[0].name == "task_kind"


def test_control_reader_scopes_related_queries_to_requested_records(tmp_path):
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path).write_hierarchy(_dataset())
    with connect_control(path) as connection:
        connection.execute(
            "INSERT INTO records SELECT 'record-2', start_time_us, time_span_start_us, "
            "time_span_end_us, metadata FROM records WHERE record_id = 'record-1'"
        )
        connection.execute("INSERT INTO sources VALUES ('source-2', 'record-2', NULL, 'Source 2', '{}')")
        connection.execute(
            """INSERT INTO signals
               SELECT 'signal-2', 'source-2', name, 'missing-axis', spec_type, spec_name, unit, dtype,
                      categories, value_shape, dimension_names, nullable, n_values, metadata
               FROM signals LIMIT 1"""
        )

    with DuckDBControlReader(path) as reader:
        (record,) = reader.read_records(["record-1"])
        assert record.id == "record-1"
        with pytest.raises(TimeFFormatError, match="missing axis"):
            reader.read_records()


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("DELETE FROM annotation_contents WHERE content_id = 'sex-male'", "missing content"),
        ("UPDATE sources SET parent_source_id = 'missing' WHERE source_id = 'ecg'", "missing parent"),
        ("UPDATE signals SET source_id = 'missing' WHERE signal_id = 'lead-i'", "missing source"),
        (
            "UPDATE annotation_occurrences SET object_id = 'missing' WHERE content_id = 'sex-male'",
            "missing Record",
        ),
        (
            "UPDATE annotation_occurrences SET object_type = 'Unknown' WHERE content_id = 'sex-male'",
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
