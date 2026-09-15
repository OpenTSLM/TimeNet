"""The DDL and the write-time validations, driven directly against a database."""

import dataclasses

import duckdb
import pytest

from timenet.control_plane import schema as ddl
from timenet.errors import TimeFValidationError
from timenet.types import TASKS, TaskType


@pytest.fixture
def db():
    """An empty in-memory control database with every table created."""
    connection = duckdb.connect()
    for statement in [s.strip() for s in ddl.DDL.split(";") if s.strip()]:
        connection.execute(statement)
    yield connection
    connection.close()


def _offenders(connection, description):
    """Run one named validation and return the rows it found."""
    query = next(query for text, query in ddl.VALIDATIONS if text == description)
    return connection.execute(query).fetchall()


def _failing(connection):
    """Return the description of every validation that finds a row."""
    return [description for description, query in ddl.VALIDATIONS if connection.execute(query).fetchall()]


# ---- the DDL ----------------------------------------------------------------------------------


def test_ddl_creates_every_declared_table(db):
    created = {row[0] for row in db.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert created == set(ddl.TABLES)


def test_every_surrogate_id_column_uses_the_declared_width(db):
    for table, column in ddl.KEYED_TABLES.items():
        declared = db.execute(
            "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
            [table, column],
        ).fetchone()
        assert declared is not None, f"{table}.{column} is missing"
        assert declared[0] == ddl.ID_TYPE, f"{table}.{column} is {declared[0]}, not {ddl.ID_TYPE}"


def test_an_empty_database_passes_every_validation(db):
    assert _failing(db) == []


def test_no_table_ships_a_key_constraint(db):
    # The measured case for dropping them: ECG-QA is 19.7 MB plain and 189.0 MB with keys and indexes.
    constraints = db.execute("SELECT constraint_type FROM duckdb_constraints()").fetchall()
    assert not [kind for (kind,) in constraints if kind in {"PRIMARY KEY", "FOREIGN KEY", "UNIQUE"}]


# ---- dense ids --------------------------------------------------------------------------------


def _insert_records(connection, ids):
    for index, record_id in enumerate(ids):
        connection.execute("INSERT INTO records VALUES (?, ?, NULL, NULL, NULL, [])", [record_id, f"record-{index}"])


def test_dense_ids_accept_a_run_from_zero(db):
    _insert_records(db, [0, 1, 2])
    assert not _offenders(db, "records.record_id is not dense")


def test_dense_ids_catch_a_gap(db):
    _insert_records(db, [0, 1, 3])
    assert _offenders(db, "records.record_id is not dense")


def test_dense_ids_catch_a_repeat(db):
    # One repeat plus one gap leaves the row count unchanged, so counting rows alone would miss it.
    _insert_records(db, [0, 1, 1, 3])
    assert _offenders(db, "records.record_id is not dense")


def test_dense_ids_catch_a_run_that_does_not_start_at_zero(db):
    _insert_records(db, [1, 2, 3])
    assert _offenders(db, "records.record_id is not dense")


# ---- external ids -----------------------------------------------------------------------------


def test_duplicate_external_ids_are_rejected(db):
    db.execute("INSERT INTO records VALUES (0, 'rec', NULL, NULL, NULL, [])")
    db.execute("INSERT INTO records VALUES (1, 'rec', NULL, NULL, NULL, [])")
    assert _offenders(db, "duplicate records.external_id")


# ---- references -------------------------------------------------------------------------------


def test_a_link_naming_no_record_is_rejected(db):
    db.execute("INSERT INTO record_time_series VALUES (7, 0, 0)")
    assert _offenders(db, "link names a record that does not exist")


def test_a_chunk_naming_no_series_is_rejected(db):
    db.execute("INSERT INTO time_series_chunks VALUES (3, 0, 0, 0, 0, 10)")
    assert _offenders(db, "chunk names a series that does not exist")


def test_a_chunk_naming_an_undeclared_artifact_is_rejected(db):
    db.execute("INSERT INTO time_series_chunks VALUES (0, 0, 9, 0, 0, 10)")
    assert _offenders(db, "chunk names an artifact the values plane did not declare")


def test_a_parquet_chunk_without_a_row_offset_is_rejected(db):
    db.execute("INSERT INTO values_artifacts VALUES (0, 'time_series/part-00000000.parquet', 'parquet')")
    db.execute("INSERT INTO time_series_chunks VALUES (0, 0, 0, 0, NULL, 10)")
    assert _offenders(db, "parquet chunk without a row offset")


def test_a_zarr_chunk_with_a_row_offset_is_rejected(db):
    db.execute("INSERT INTO values_artifacts VALUES (0, 'time_series.zarr/s/a', 'zarr')")
    db.execute("INSERT INTO time_series_chunks VALUES (0, 0, 0, 0, 4, 10)")
    assert _offenders(db, "zarr chunk with a row offset")


def _insert_spec(connection, spec_id=0, spec_type="s"):
    connection.execute(
        "INSERT INTO specs VALUES (?, ?, 'S', 'millivolt', NULL, NULL, NULL, 'float32', [], [], [], false)",
        [spec_id, spec_type],
    )


def _insert_series(connection, *, n_values):
    _insert_spec(connection)
    connection.execute("INSERT INTO axes VALUES (0, 'regular', 1000, 1, 0, NULL, NULL)")
    connection.execute("INSERT INTO time_series VALUES (0, 'ts-0', 'a', NULL, 0, 0, ?)", [n_values])
    connection.execute("INSERT INTO values_artifacts VALUES (0, 'time_series/part-00000000.parquet', 'parquet')")


def test_a_series_length_disagreeing_with_its_chunks_is_rejected(db):
    _insert_series(db, n_values=10)
    db.execute("INSERT INTO time_series_chunks VALUES (0, 0, 0, 0, 0, 4)")
    assert _offenders(db, "series length disagrees with its chunks")


def test_a_series_length_matching_its_chunks_passes(db):
    _insert_series(db, n_values=10)
    db.execute("INSERT INTO time_series_chunks VALUES (0, 0, 0, 0, 0, 4)")
    db.execute("INSERT INTO time_series_chunks VALUES (0, 1, 0, 0, 1, 6)")
    assert not _offenders(db, "series length disagrees with its chunks")


def test_a_series_with_no_chunk_is_rejected(db):
    _insert_series(db, n_values=10)
    assert _offenders(db, "series has no chunk in the values plane")


def test_two_specs_claiming_one_type_are_rejected(db):
    _insert_spec(db, spec_id=0)
    _insert_spec(db, spec_id=1)
    assert _offenders(db, "duplicate specs.spec_type")


def test_a_spec_with_half_a_data_source_is_rejected(db):
    db.execute(
        "INSERT INTO specs VALUES (0, 's', 'S', 'millivolt', 'device', NULL, NULL, 'float32', [], [], [], false)"
    )
    assert _offenders(db, "spec carries half a data source")


def test_an_annotation_whose_key_the_schema_does_not_declare_is_rejected(db):
    db.execute("INSERT INTO annotations VALUES (0, 'ann-0', 'undeclared', NULL, NULL, NULL, NULL, NULL)")
    assert _offenders(db, "annotation names a key the schema does not declare")


def test_an_attachment_naming_no_annotation_is_rejected(db):
    db.execute("INSERT INTO record_annotations VALUES (0, 4, 0, 0)")
    assert _offenders(db, "record_annotations names an annotation that does not exist")


def _insert_task(connection, task_type="answer"):
    connection.execute("INSERT INTO tasks VALUES (0, 'task-0', ?, NULL, NULL)", [task_type])


def test_a_task_ref_naming_no_record_is_rejected(db):
    # An id that named nothing resolved to null, which is what a dangling reference looks like here.
    _insert_task(db, "ts_generation")
    db.execute("INSERT INTO task_refs VALUES (0, 'target_record_id', 0, 'record', NULL)")
    assert _offenders(db, "task payload names a record that does not exist")


def test_a_task_item_naming_no_record_is_rejected(db):
    _insert_task(db)
    db.execute("INSERT INTO task_items VALUES (0, 'input', 0, 'record', NULL, NULL)")
    assert _offenders(db, "task item names a record that does not exist")


def test_a_record_naming_no_task_is_rejected(db):
    _insert_records(db, [0])
    db.execute("INSERT INTO record_tasks VALUES (0, NULL)")
    assert _offenders(db, "record names a task that does not exist")


def test_a_text_item_carrying_a_record_is_rejected(db):
    _insert_task(db)
    db.execute("INSERT INTO task_items VALUES (0, 'target', 0, 'text', 'yes', 4)")
    assert _offenders(db, "task item disagrees with its item_type")


def test_a_dangling_derivation_is_not_checked(db):
    # A streamed task skips the cross-task checks add_task runs, so a dangling derivation reaches the
    # writer and the reader is what reports it. Checking here would reject a write main accepted.
    _insert_task(db)
    db.execute("INSERT INTO task_from_tasks VALUES (0, 0, 'missing')")
    assert _failing(db) == []


def test_a_reversed_interval_is_rejected(db):
    db.execute("INSERT INTO task_spans VALUES (0, 'scope', 0, 'seconds', 500, 100, NULL)")
    assert _offenders(db, "an interval span ends before it starts")


def test_a_point_span_has_no_end_to_check(db):
    db.execute("INSERT INTO task_spans VALUES (0, 'scope', 0, 'seconds', 500, NULL, NULL)")
    assert not _offenders(db, "an interval span ends before it starts")


def test_an_annotation_end_without_a_start_is_rejected(db):
    db.execute("INSERT INTO annotations VALUES (0, 'ann-0', 'k', NULL, NULL, NULL, 10, NULL)")
    assert _offenders(db, "annotation span bound without a start")


# ---- the task payload declaration ---------------------------------------------------------------


@pytest.mark.parametrize("task_type", list(TaskType))
def test_every_task_type_declares_its_whole_payload(task_type):
    declared = {field.name for field in ddl.task_payload(task_type)}
    cls = TASKS[task_type]
    stored = {field.name for field in dataclasses.fields(cls)} - ddl._TASK_FRAME
    if cls.answer_is_record:
        stored.discard("target")
    assert declared == stored


def test_a_payload_declaration_that_drifts_is_rejected(monkeypatch):
    shortened = dict(ddl.TASK_PAYLOAD)
    shortened[TaskType.CLASSIFICATION] = (ddl.PayloadField("target", ddl.PayloadKind.TEXT),)
    monkeypatch.setattr(ddl, "TASK_PAYLOAD", shortened)
    with pytest.raises(TimeFValidationError, match="do not match the control-plane declaration"):
        ddl.task_payload(TaskType.CLASSIFICATION)


def test_a_list_valued_scalar_field_is_rejected(monkeypatch):
    listed = dict(ddl.TASK_PAYLOAD)
    listed[TaskType.ANSWER] = (ddl.PayloadField("target", ddl.PayloadKind.TEXT, is_list=True),)
    monkeypatch.setattr(ddl, "TASK_PAYLOAD", listed)
    with pytest.raises(TimeFValidationError, match="task_fields stores one value per row"):
        ddl.task_payload(TaskType.ANSWER)


def test_only_ref_and_span_fields_store_elements():
    for task_type in TaskType:
        for declared in ddl.task_payload(task_type):
            expected = declared.kind not in {ddl.PayloadKind.TEXT, ddl.PayloadKind.NUMBER}
            assert declared.stores_elements is expected
