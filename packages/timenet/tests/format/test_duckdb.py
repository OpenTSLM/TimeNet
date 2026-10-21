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


def test_control_schema_stores_its_version_and_no_constraints(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        version = connection.execute("SELECT value FROM control_metadata WHERE key = 'schema_version'").fetchone()
        stored_constraints = connection.execute(
            "SELECT constraint_type FROM duckdb_constraints() WHERE constraint_type <> 'NOT NULL'"
        ).fetchall()

    assert version == (str(CONTROL_SCHEMA_VERSION),)
    assert stored_constraints == []


def test_control_transaction_rolls_back_all_rows(tmp_path):
    with connect_control(tmp_path.joinpath("control.duckdb")) as connection:
        create_control_schema(connection)

        with pytest.raises(RuntimeError, match="stop"), transaction(connection):
            connection.execute(
                """INSERT INTO records (
                       record_id, clock_id, time_span_start_us, time_span_end_us, metadata
                   ) VALUES (?, 1, NULL, NULL, ?)""",
                [10, "{}"],
            )
            raise RuntimeError("stop")

        assert connection.execute("SELECT count(*) FROM records").fetchone() == (0,)


def test_entity_id_columns_are_bigint_and_dataset_id_stays_varchar(tmp_path):
    with connect_control(tmp_path.joinpath("control.duckdb")) as connection:
        create_control_schema(connection)
        types = dict(
            connection.execute(
                "SELECT table_name || '.' || column_name, data_type FROM information_schema.columns"
            ).fetchall()
        )

    id_columns = {name: kind for name, kind in types.items() if name.endswith("_id") and not name.endswith(".clock_id")}
    assert id_columns.pop("datasets.dataset_id") == "VARCHAR"
    assert set(id_columns.values()) == {"BIGINT"}
    assert {name.split(".")[1] for name in id_columns} >= {
        "record_id",
        "source_id",
        "axis_id",
        "signal_id",
        "content_id",
        "occurrence_id",
        "task_id",
    }


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
