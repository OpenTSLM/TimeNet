"""What the database refuses to store.

The draft points an annotation at its object with an ``object_type``/``object_id`` pair. A foreign
key references one table, so that pair cannot have one, and nothing stops an occurrence naming an
object that does not exist. These tests pin the integrity the typed columns buy instead.
"""

import duckdb
import pytest

from timenet.control_plane import TimeFWriter, schema as ddl
import timenet.control_plane.writer as writer_module
from timenet.testing import make_dataset


@pytest.fixture
def connection(tmp_path):
    """An empty control database with the real schema, open for writing."""
    connection = duckdb.connect()
    connection.execute(f"ATTACH '{tmp_path / 'control.duckdb'}' AS control")
    connection.execute("USE control")
    for statement in (s.strip() for s in ddl.DDL.split(";") if s.strip()):
        connection.execute(statement)
    connection.execute("INSERT INTO datasets VALUES ('dataset', NULL)")
    connection.execute("INSERT INTO records VALUES ('r1', NULL, NULL)")
    connection.execute("INSERT INTO sources VALUES ('s1', 'r1', NULL, '0000', 0, 'Monitor', 0, NULL)")
    connection.execute("INSERT INTO axes VALUES ('a1', 'regular', 2000, 1, 0, NULL, NULL)")
    connection.execute("INSERT INTO specs VALUES ('sp1', 'ECG', 'mV', 'float32', false)")
    connection.execute("INSERT INTO annotation_contents VALUES ('c1', 'patient_sex', 'male', NULL, NULL)")
    yield connection
    connection.close()


def _occurrence(**overrides):
    row = {
        "occurrence_id": 1,
        "content_id": "c1",
        "object_type": "record",
        "on_dataset_id": None,
        "on_task_id": None,
        "on_record_id": "r1",
        "on_source_id": None,
        "on_signal_id": None,
        "scope_record_id": "r1",
        "span_type": "static",
        "start_us": None,
        "end_us": None,
        "provenance": None,
        "confidence": None,
        "metadata": None,
    }
    row.update(overrides)
    return tuple(row.values())


def test_annotation_on_a_missing_signal_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="foreign key"):
        connection.execute(
            "INSERT INTO annotation_occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _occurrence(object_type="signal", on_record_id=None, on_signal_id="does-not-exist"),
        )


def test_annotation_pointing_at_nothing_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute(
            "INSERT INTO annotation_occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _occurrence(on_record_id=None),
        )


def test_annotation_pointing_at_two_things_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute(
            "INSERT INTO annotation_occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _occurrence(on_source_id="s1"),
        )


def test_unknown_object_type_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute(
            "INSERT INTO annotation_occurrences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _occurrence(object_type="banana"),
        )


def test_linking_a_signal_to_a_missing_source_is_refused(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    with pytest.raises(duckdb.ConstraintException, match="foreign key"):
        connection.execute("INSERT INTO source_signals VALUES ('missing-source', 'sig1', 0)")


def test_linking_a_missing_signal_to_a_source_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="foreign key"):
        connection.execute("INSERT INTO source_signals VALUES ('s1', 'missing-signal', 0)")


def test_one_signal_can_hang_off_several_sources(connection):
    """A series used by several records is stored once and linked many times."""
    connection.execute("INSERT INTO records VALUES ('r2', NULL, NULL)")
    connection.execute("INSERT INTO sources VALUES ('s2', 'r2', NULL, '0000', 0, 'Monitor', 0, NULL)")
    connection.execute("INSERT INTO signals VALUES ('shared', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO source_signals VALUES ('s1', 'shared', 0)")
    connection.execute("INSERT INTO source_signals VALUES ('s2', 'shared', 0)")
    rows = connection.execute("SELECT count(*) FROM source_signals WHERE signal_id = 'shared'").fetchone()
    assert rows is not None
    assert rows[0] == 2


def test_source_under_a_missing_record_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="foreign key"):
        connection.execute("INSERT INTO sources VALUES ('s2', 'missing', NULL, '0001', 0, 'X', 1, NULL)")


def test_source_under_a_missing_parent_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="foreign key"):
        connection.execute("INSERT INTO sources VALUES ('s2', 'r1', 'missing', '0000.0000', 1, 'X', 0, NULL)")


def test_task_item_with_a_bad_role_is_refused(connection):
    connection.execute("INSERT INTO tasks VALUES ('t1', 'Diagnose.', NULL)")
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO task_items VALUES ('t1', 'sideways', 0, 'text', 'x', NULL)")


def test_duplicate_signal_id_is_refused(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    with pytest.raises(duckdb.ConstraintException):
        connection.execute("INSERT INTO signals VALUES ('sig1', 'II', 'a1', 'sp1', 10, NULL)")


def test_writer_refuses_to_overwrite_a_committed_version(tmp_path):
    dataset = make_dataset(n_records=1, n_values=32)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with pytest.raises(Exception, match="already holds a committed version"):
        TimeFWriter(tmp_path, dataset.metadata)


def test_a_failed_build_leaves_nothing_behind(tmp_path):
    dataset = make_dataset(n_records=1, n_values=32)
    with pytest.raises(RuntimeError), TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
        raise RuntimeError("build blew up after writing")
    # The version committed before the error, so it exists; what must not survive is a staging dir.
    assert not list(tmp_path.glob("**/*.tmp-*"))


def test_staging_is_removed_when_the_write_itself_fails(tmp_path, monkeypatch):
    dataset = make_dataset(n_records=1, n_values=32)

    def explode(*args, **kwargs):
        raise RuntimeError("values plane failed")

    monkeypatch.setattr(writer_module.ValuesPlaneWriter, "write", explode)
    with pytest.raises(RuntimeError, match="values plane failed"), TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    assert not list(tmp_path.glob("**/*.tmp-*"))
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()
