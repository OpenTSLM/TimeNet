import pytest

from timenet.errors import TimeFFormatError
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control


def test_audit_accepts_a_freshly_written_database(control_path):
    with connect_control(control_path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)


def test_audit_requires_chunks_unless_told_otherwise(control_path):
    with (
        connect_control(control_path, read_only=True) as connection,
        pytest.raises(TimeFFormatError, match="value chunks whose lengths do not match"),
    ):
        audit_control_database(connection)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "INSERT INTO records (record_key, record_id, clock_id, metadata, read_order) SELECT record_key, record_id, clock_id, metadata, read_order FROM records",
            "records contains duplicate values for record_key",
        ),
        (
            "UPDATE sources SET record_key = -1 WHERE source_id = 'monitor'",
            r"sources\.record_key refers to missing records\.record_key",
        ),
        (
            "UPDATE sources SET parent_source_key = source_key WHERE source_id = 'ecg'",
            r"invalid sources\.parent_source_key value",
        ),
        (
            "UPDATE signals SET n_values = 0",
            r"invalid signals\.n_values value",
        ),
        (
            "INSERT INTO records (record_key, record_id, clock_id, metadata, read_order) VALUES (-1, 'ghost', 1, '{}', 0); "
            "UPDATE sources SET record_key = -1 WHERE source_id = 'ecg'",
            "belong to different records",
        ),
        (
            "UPDATE sources SET parent_source_key = (SELECT source_key FROM sources WHERE source_id = 'ecg') "
            "WHERE source_id = 'monitor'",
            "contains a cycle",
        ),
        (
            "INSERT INTO external_values (object_kind, object_id, field, position, value_path, row_index, "
            "value_kind, n_values, value_digest) "
            "SELECT 'Axis', axis_id, 'offsets', 0, 'payloads/part-00000000.parquet', 0, 'offsets', 2, "
            "'sha256:' || repeat('0', 64) FROM axes",
            "axis length that does not match",
        ),
    ],
)
def test_audit_names_the_first_violated_invariant(control_path, statement, message):
    with connect_control(control_path) as connection:
        for part in statement.split("; "):
            connection.execute(part)
        with pytest.raises(TimeFFormatError, match=message):
            audit_control_database(connection, require_chunks=False)
