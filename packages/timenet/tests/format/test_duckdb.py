import duckdb
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.duckdb import (
    check_control_schema,
    connect_control,
    create_control_schema,
    transaction,
)


def test_control_transaction_rolls_back_all_rows(tmp_path):
    with connect_control(tmp_path.joinpath("control.duckdb")) as connection:
        create_control_schema(connection)

        with pytest.raises(RuntimeError, match="stop"), transaction(connection):
            connection.execute(
                """INSERT INTO records (
                       record_id, start_time_us, time_span_start_us, time_span_end_us, metadata
                   ) VALUES (?, NULL, NULL, NULL, ?)""",
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
