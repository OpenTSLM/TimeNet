from concurrent.futures import ThreadPoolExecutor

import duckdb
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import (
    CONTROL_SCHEMA_VERSION,
    ControlSession,
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
                """INSERT INTO records (
                       record_id, clock_id, time_span_start_us, time_span_end_us, metadata
                   ) VALUES (?, 1, NULL, NULL, ?)""",
                ["record-1", "{}"],
            )
            raise RuntimeError("stop")

        assert connection.execute("SELECT count(*) FROM records").fetchone() == (0,)


@pytest.mark.parametrize("version", ["2", "999"])
def test_control_schema_rejects_an_unknown_version(tmp_path, version):
    path = tmp_path.joinpath("control.duckdb")
    with connect_control(path) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = ?", [version])

        with pytest.raises(TimeFFormatError, match="unsupported control schema version"):
            check_control_schema(connection)


def test_control_schema_rejects_a_non_timef_database(tmp_path):
    path = tmp_path.joinpath("control.duckdb")
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")

        with pytest.raises(TimeFFormatError, match="schema metadata"):
            check_control_schema(connection)


@pytest.fixture
def control_paths(tmp_path):
    paths = [tmp_path / name / "control.duckdb" for name in ("child", "parent's")]
    for index, path in enumerate(paths):
        path.parent.mkdir()
        with connect_control(path) as connection:
            create_control_schema(connection)
            connection.execute("INSERT INTO control_metadata VALUES ('dataset', ?)", [str(index)])
    return paths


def test_shared_session_keeps_readers_independent_and_releases_files(control_paths):
    broken = control_paths[0].with_name("broken.duckdb")
    with connect_control(broken) as connection:
        create_control_schema(connection)
        connection.execute("UPDATE control_metadata SET value = '999' WHERE key = 'schema_version'")
    session = ControlSession()
    readers = [DuckDBControlReader(path, session=session) for path in control_paths]
    query = "SELECT value FROM control_metadata WHERE key = 'dataset'"
    try:
        readers[0].connection.execute(query)
        readers[1].connection.execute(query)
        assert readers[0].connection.fetchone() == ("0",)
        assert readers[1].connection.fetchone() == ("1",)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda reader: reader.connection.execute(query).fetchone(), readers))
        assert results == [("0",), ("1",)]
        for reader in readers:
            with pytest.raises(duckdb.Error, match="read-only"):
                reader.connection.execute("DELETE FROM control_metadata")
        # A failed open leaves the other readers of the session usable.
        with pytest.raises(TimeFFormatError, match="unsupported control schema"):
            DuckDBControlReader(broken, session=session)
        with pytest.raises(TimeFFormatError, match="could not open control database"):
            DuckDBControlReader(broken.with_name("missing.duckdb"), session=session)
        readers[0].close()
        readers[0].close()
        assert readers[1].connection.execute(query).fetchone() == ("1",)
    finally:
        for reader in readers:
            reader.close()

    # All attached files are released when the final cursor closes.
    for path in control_paths:
        with connect_control(path) as connection:
            connection.execute("INSERT INTO control_metadata VALUES ('reopened', 'yes')")
    with DuckDBControlReader(control_paths[1], session=session) as reader:
        assert reader.connection.execute(query).fetchone() == ("1",)
