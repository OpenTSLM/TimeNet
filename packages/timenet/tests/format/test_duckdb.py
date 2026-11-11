import duckdb
import pytest

from timenet.errors import FormatVersionSubject, TimeFFormatError, TimeFFormatVersionError
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
                       record_id, clock_id, time_span_start_us, time_span_end_us, metadata, read_order
                   ) VALUES (?, 1, NULL, NULL, ?, 0)""",
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


def test_a_newer_control_schema_names_the_file_and_both_versions(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = '3'")

        with pytest.raises(TimeFFormatVersionError) as caught:
            check_control_schema(connection)

    assert str(caught.value) == f"{path} uses control schema version 3, but the newest that this timenet reads is 2."
    assert caught.value.subject is FormatVersionSubject.CONTROL_SCHEMA


def test_a_newer_in_memory_control_schema_names_no_file():
    with duckdb.connect() as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = '3'")

        with pytest.raises(
            TimeFFormatVersionError, match=r"^an in-memory control database uses control schema version 3, but"
        ):
            check_control_schema(connection)


def test_an_older_control_schema_names_the_oldest_version_that_is_read(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = '1'")

        with pytest.raises(TimeFFormatVersionError) as caught:
            check_control_schema(connection)

    assert str(caught.value) == f"{path} uses control schema version 1, but the oldest that this timenet reads is 2."
    assert caught.value.path == path


@pytest.mark.parametrize("value", ["0", "two", "-2", "02"])
def test_an_invalid_control_schema_version_reports_as_unsupported(tmp_path, value):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = ?", [value])

        with pytest.raises(TimeFFormatError, match="unsupported control schema version") as caught:
            check_control_schema(connection)

    assert not isinstance(caught.value, TimeFFormatVersionError)


def test_control_schema_rejects_a_non_timef_database(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")

        with pytest.raises(TimeFFormatError, match="schema metadata"):
            check_control_schema(connection)
