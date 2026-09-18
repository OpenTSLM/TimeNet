import duckdb
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.duckdb import (
    CONTROL_SCHEMA_VERSION,
    check_control_schema,
    connect_control,
    create_control_schema,
    transaction,
)


def test_control_schema_is_version_one_and_has_no_stored_relationship_constraints(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        version = connection.execute("SELECT value FROM control_metadata WHERE key = 'schema_version'").fetchone()
        stored_constraints = connection.execute(
            "SELECT constraint_type FROM duckdb_constraints() WHERE constraint_type <> 'NOT NULL'"
        ).fetchall()

    assert CONTROL_SCHEMA_VERSION == 1
    assert version == ("1",)
    assert stored_constraints == []


def test_control_transaction_rolls_back_all_rows(tmp_path):
    with connect_control(tmp_path.joinpath("control.duckdb")) as connection:
        create_control_schema(connection)

        with pytest.raises(RuntimeError, match="stop"), transaction(connection):
            connection.execute(
                "INSERT INTO records VALUES (?, NULL, NULL, NULL, ?)",
                ["record-1", "{}"],
            )
            raise RuntimeError("stop")

        assert connection.execute("SELECT count(*) FROM records").fetchone() == (0,)


def test_control_schema_rejects_an_unknown_version(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = '999'")

        with pytest.raises(TimeFFormatError, match="unsupported control schema version"):
            check_control_schema(connection)


def test_control_schema_rejects_a_non_timef_database(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")

        with pytest.raises(TimeFFormatError, match="schema metadata"):
            check_control_schema(connection)
