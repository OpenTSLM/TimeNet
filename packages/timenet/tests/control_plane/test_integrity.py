"""What the writer refuses to publish.

The shipped database declares no primary or foreign keys: they cost ten times the file size, and a
version is immutable once written, so an invariant that holds at publish time holds for the rest of
its life. These tests pin that the write-time checks still catch everything those constraints used
to catch, at the one moment it can go wrong.

CHECK constraints are still declared, because they cost nothing, so the cases they cover still fail
at insert time.
"""

import duckdb
import pytest

from timenet.control_plane import TimeFWriter, schema as ddl
from timenet.control_plane.values import ValuesPlaneWriter
from timenet.control_plane.writer import _validate
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset


@pytest.fixture
def connection(tmp_path):
    """A control database holding one valid record, source, axis, spec, and annotation content."""
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
    connection.execute("INSERT INTO annotations VALUES ('c1', 'patient_sex', 'male', NULL, NULL)")
    connection.execute("INSERT INTO values_artifacts VALUES ('f.parquet', 'parquet')")
    yield connection
    connection.close()


def _occurrence(**overrides):
    row = {
        "occurrence_id": 1,
        "annotation_id": "c1",
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


def _insert_occurrence(connection, **overrides):
    connection.execute(
        "INSERT INTO entities_to_annotations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", _occurrence(**overrides)
    )


def test_a_clean_database_validates(connection):
    _validate(connection)


def test_annotation_on_a_missing_signal_is_caught(connection):
    _insert_occurrence(connection, object_type="signal", on_record_id=None, on_signal_id="does-not-exist")
    with pytest.raises(TimeFValidationError, match="occurrence names a signal that does not exist"):
        _validate(connection)


def test_source_under_a_missing_record_is_caught(connection):
    connection.execute("INSERT INTO sources VALUES ('s2', 'missing', NULL, '0001', 0, 'X', 1, NULL)")
    with pytest.raises(TimeFValidationError, match="source names a record that does not exist"):
        _validate(connection)


def test_source_under_a_missing_parent_is_caught(connection):
    connection.execute("INSERT INTO sources VALUES ('s2', 'r1', 'missing', '0000.0000', 1, 'X', 0, NULL)")
    with pytest.raises(TimeFValidationError, match="source names a parent that does not exist"):
        _validate(connection)


def test_source_in_a_different_record_than_its_parent_is_caught(connection):
    connection.execute("INSERT INTO records VALUES ('r2', NULL, NULL)")
    connection.execute("INSERT INTO sources VALUES ('s2', 'r2', 's1', '0000.0000', 1, 'X', 0, NULL)")
    with pytest.raises(TimeFValidationError, match="different record than its parent"):
        _validate(connection)


def test_linking_a_missing_signal_is_caught(connection):
    connection.execute("INSERT INTO source_signals VALUES ('s1', 'missing-signal', 0)")
    with pytest.raises(TimeFValidationError, match="link names a signal that does not exist"):
        _validate(connection)


def test_linking_to_a_missing_source_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO source_signals VALUES ('missing-source', 'sig1', 0)")
    with pytest.raises(TimeFValidationError, match="link names a source that does not exist"):
        _validate(connection)


def test_duplicate_signal_id_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signals VALUES ('sig1', 'II', 'a1', 'sp1', 10, NULL)")
    with pytest.raises(TimeFValidationError, match="duplicate signal_id"):
        _validate(connection)


def test_signal_naming_a_missing_axis_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'missing-axis', 'sp1', 10, NULL)")
    with pytest.raises(TimeFValidationError, match="signal names an axis that does not exist"):
        _validate(connection)


def test_chunk_naming_a_missing_signal_is_caught(connection):
    connection.execute("INSERT INTO signal_chunks VALUES ('missing', 0, 'f.parquet', 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="chunk names a signal that does not exist"):
        _validate(connection)


# The locator is backend-neutral, so what a Parquet shard and a Zarr array each need out of it can
# no longer be said in a column type. These checks say it instead.


def test_chunk_naming_an_undeclared_artifact_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('sig1', 0, 'never-written.parquet', 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="chunk names an artifact the values plane did not declare"):
        _validate(connection)


def test_parquet_chunk_without_a_row_offset_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('sig1', 0, 'f.parquet', 0, NULL, 10)")
    with pytest.raises(TimeFValidationError, match="parquet chunk without a row offset"):
        _validate(connection)


def test_zarr_chunk_with_a_row_offset_is_caught(connection):
    connection.execute("INSERT INTO values_artifacts VALUES ('time_series.zarr/ecg', 'zarr')")
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('sig1', 0, 'time_series.zarr/ecg', 128, 0, 10)")
    with pytest.raises(TimeFValidationError, match="zarr chunk with a row offset"):
        _validate(connection)


def test_a_zarr_chunk_needs_only_an_element_offset(connection):
    connection.execute("INSERT INTO values_artifacts VALUES ('time_series.zarr/ecg', 'zarr')")
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('sig1', 0, 'time_series.zarr/ecg', 128, NULL, 10)")
    _validate(connection)


def test_an_unknown_values_backend_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO values_artifacts VALUES ('x', 'hdf5')")


def test_signal_length_disagreeing_with_its_chunks_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES ('sig1', 'I', 'a1', 'sp1', 99, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('sig1', 0, 'f.parquet', 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="signal length disagrees with its chunks"):
        _validate(connection)


def test_task_item_naming_a_missing_record_is_caught(connection):
    connection.execute("INSERT INTO tasks VALUES ('t1', 'Diagnose.', NULL)")
    connection.execute("INSERT INTO task_items VALUES ('t1', 'input', 0, 'record', NULL, 'missing')")
    with pytest.raises(TimeFValidationError, match="task item names a record that does not exist"):
        _validate(connection)


def test_one_signal_can_hang_off_several_sources(connection):
    """A series used by several records is stored once and linked many times."""
    connection.execute("INSERT INTO records VALUES ('r2', NULL, NULL)")
    connection.execute("INSERT INTO sources VALUES ('s2', 'r2', NULL, '0000', 0, 'Monitor', 0, NULL)")
    connection.execute("INSERT INTO signals VALUES ('shared', 'I', 'a1', 'sp1', 10, NULL)")
    connection.execute("INSERT INTO signal_chunks VALUES ('shared', 0, 'f.parquet', 0, 0, 10)")
    connection.execute("INSERT INTO source_signals VALUES ('s1', 'shared', 0)")
    connection.execute("INSERT INTO source_signals VALUES ('s2', 'shared', 0)")
    _validate(connection)
    rows = connection.execute("SELECT count(*) FROM source_signals WHERE signal_id = 'shared'").fetchone()
    assert rows is not None
    assert rows[0] == 2


# CHECK constraints cost no storage, so they stay declared and still fire at insert time.


def test_annotation_pointing_at_nothing_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        _insert_occurrence(connection, on_record_id=None)


def test_annotation_pointing_at_two_things_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        _insert_occurrence(connection, on_source_id="s1")


def test_unknown_object_type_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        _insert_occurrence(connection, object_type="banana")


def test_task_item_with_a_bad_role_is_refused(connection):
    connection.execute("INSERT INTO tasks VALUES ('t1', 'Diagnose.', NULL)")
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO task_items VALUES ('t1', 'sideways', 0, 'text', 'x', NULL)")


# the writer's own guards


def test_writer_refuses_to_overwrite_a_committed_version(tmp_path):
    dataset = make_dataset(n_records=1, n_values=32)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with pytest.raises(TimeFValidationError, match="already holds a committed version"):
        TimeFWriter(tmp_path, dataset.metadata)


def test_a_build_that_fails_validation_publishes_nothing(tmp_path, monkeypatch):
    """A validation failure must leave no version behind, not a half-valid one."""
    dataset = make_dataset(n_records=2, n_values=32)
    monkeypatch.setattr(
        ddl, "VALIDATIONS", (*ddl.VALIDATIONS, ("a check that always fails", "SELECT record_id FROM records"))
    )
    with (
        pytest.raises(TimeFValidationError, match="a check that always fails"),
        TimeFWriter(tmp_path, dataset.metadata) as writer,
    ):
        writer.write(dataset)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()
    assert not list(tmp_path.glob("**/*.tmp-*"))


def test_staging_is_removed_when_the_write_itself_fails(tmp_path, monkeypatch):
    dataset = make_dataset(n_records=1, n_values=32)

    def explode(*args, **kwargs):
        raise RuntimeError("values plane failed")

    monkeypatch.setattr(ValuesPlaneWriter, "add", explode)
    with pytest.raises(RuntimeError, match="values plane failed"), TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    assert not list(tmp_path.glob("**/*.tmp-*"))
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()
