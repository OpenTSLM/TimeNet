"""What the writer refuses to publish.

The shipped database declares no primary or foreign keys: they cost ten times the file size, and a
version is immutable once written, so an invariant that holds at publish time holds for the rest of
its life. These tests pin that the write-time checks still catch everything those constraints used
to catch, at the one moment it can go wrong.

Density is checked the same way. Nothing in the file says the surrogate ids run 0, 1, 2 with no gap,
but the reader's worker partition is ``record_id % num_workers``, which silently drops rows if they
do not, so a gap has to be caught here.

CHECK constraints are still declared, because they cost nothing, so the cases they cover still fail
at insert time.
"""

from fractions import Fraction
from typing import cast

import duckdb
import numpy as np
import pytest

from timenet.control_plane import (
    Annotation,
    DeclarativeDataset,
    Record,
    RecordRef,
    Signal,
    Source,
    Task,
    TimeFReader,
    TimeFWriter,
    schema as ddl,
)
from timenet.control_plane.values import ValuesPlaneWriter
from timenet.control_plane.writer import _validate
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.testing import make_dataset, make_metadata, make_record
from timenet.types import TimeSeriesSpec, ureg


@pytest.fixture
def connection(tmp_path):
    """A control database holding one valid record, source, axis, spec, annotation, and artifact.

    Every id is 0, so a test that adds a second row of a kind has to use 1 to stay dense.
    """
    connection = duckdb.connect()
    connection.execute(f"ATTACH '{tmp_path / 'control.duckdb'}' AS control")
    connection.execute("USE control")
    for statement in (s.strip() for s in ddl.DDL.split(";") if s.strip()):
        connection.execute(statement)
    connection.execute("INSERT INTO records VALUES (0, 'r1', NULL)")
    connection.execute("INSERT INTO sources VALUES (0, 's1', 0, NULL, '0000', 0, 'Monitor', 0)")
    connection.execute("INSERT INTO axes VALUES (0, 'regular', 2000, 1, 0, NULL, NULL)")
    connection.execute("INSERT INTO specs VALUES (0, 'ECG', 'ECG voltage', 'mV', 'float32', false)")
    connection.execute("INSERT INTO annotations VALUES (0, 'patient_sex', 'male', NULL)")
    connection.execute("INSERT INTO values_artifacts VALUES (0, 'f.parquet', 'parquet')")
    yield connection
    connection.close()


def _signal(connection, signal_id=0, *, external_id="sig1", n_values=10):
    """Insert one valid signal, so a test can point something at it."""
    connection.execute(
        "INSERT INTO signals VALUES (?, ?, 'I', 0, 0, ?)",
        [signal_id, external_id, n_values],
    )


def test_a_clean_database_validates(connection):
    _validate(connection)


# The dense integer ids, which nothing in the shipped file constrains.


def test_a_gap_in_the_ids_is_caught(connection):
    connection.execute("INSERT INTO records VALUES (2, 'r3', NULL)")
    with pytest.raises(TimeFValidationError, match=r"records\.record_id is not dense"):
        _validate(connection)


def test_a_repeated_id_is_caught(connection):
    connection.execute("INSERT INTO records VALUES (0, 'r-again', NULL)")
    with pytest.raises(TimeFValidationError, match=r"records\.record_id is not dense"):
        _validate(connection)


def test_ids_that_do_not_start_at_zero_are_caught(connection):
    connection.execute("INSERT INTO tasks VALUES (1, 't1', 'Diagnose.')")
    with pytest.raises(TimeFValidationError, match=r"tasks\.task_id is not dense"):
        _validate(connection)


def test_attachment_ids_are_counted_per_table(connection):
    """Each target kind counts from 0, so two tables both holding attachment 0 is correct."""
    connection.execute("INSERT INTO dataset_annotations VALUES (0, 0, 'static', NULL, NULL, NULL, NULL)")
    connection.execute("INSERT INTO record_annotations VALUES (0, 0, 0, 'static', NULL, NULL, NULL, NULL)")
    _validate(connection)


# The caller's own id, which is nullable but must still name one thing.


def test_a_repeated_external_id_is_caught(connection):
    connection.execute("INSERT INTO records VALUES (1, 'r1', NULL)")
    with pytest.raises(TimeFValidationError, match=r"duplicate records\.external_id"):
        _validate(connection)


def test_records_without_an_external_id_do_not_collide(connection):
    """external_id is nullable, so several unnamed records are fine and only the names are unique."""
    connection.execute("INSERT INTO records VALUES (1, NULL, NULL)")
    connection.execute("INSERT INTO records VALUES (2, NULL, NULL)")
    _validate(connection)


# The source tree.


def test_source_under_a_missing_record_is_caught(connection):
    connection.execute("INSERT INTO sources VALUES (1, 's2', 9, NULL, '0001', 0, 'X', 1)")
    with pytest.raises(TimeFValidationError, match="source names a record that does not exist"):
        _validate(connection)


def test_source_under_a_missing_parent_is_caught(connection):
    connection.execute("INSERT INTO sources VALUES (1, 's2', 0, 9, '0000.0000', 1, 'X', 0)")
    with pytest.raises(TimeFValidationError, match="source names a parent that does not exist"):
        _validate(connection)


def test_source_in_a_different_record_than_its_parent_is_caught(connection):
    connection.execute("INSERT INTO records VALUES (1, 'r2', NULL)")
    connection.execute("INSERT INTO sources VALUES (1, 's2', 1, 0, '0000.0000', 1, 'X', 0)")
    with pytest.raises(TimeFValidationError, match="different record than its parent"):
        _validate(connection)


def test_a_path_that_is_not_its_parent_s_is_caught(connection):
    """A subtree query is a prefix match, so a path that does not extend its parent's breaks it."""
    connection.execute("INSERT INTO sources VALUES (1, 's2', 0, 0, '9999.0000', 1, 'X', 0)")
    with pytest.raises(TimeFValidationError, match="source path disagrees with its parent's"):
        _validate(connection)


# Signals and the link table.


def test_linking_a_missing_signal_is_caught(connection):
    connection.execute("INSERT INTO source_signals VALUES (0, 9, 0)")
    with pytest.raises(TimeFValidationError, match="link names a signal that does not exist"):
        _validate(connection)


def test_linking_to_a_missing_source_is_caught(connection):
    _signal(connection)
    connection.execute("INSERT INTO source_signals VALUES (9, 0, 0)")
    with pytest.raises(TimeFValidationError, match="link names a source that does not exist"):
        _validate(connection)


def test_the_same_link_twice_is_caught(connection):
    _signal(connection)
    connection.execute("INSERT INTO source_signals VALUES (0, 0, 0)")
    connection.execute("INSERT INTO source_signals VALUES (0, 0, 1)")
    with pytest.raises(TimeFValidationError, match=r"duplicate \(source_id, signal_id\) link"):
        _validate(connection)


def test_duplicate_signal_id_is_caught(connection):
    _signal(connection, 0, external_id="sig1")
    _signal(connection, 0, external_id="sig2")
    with pytest.raises(TimeFValidationError, match=r"signals\.signal_id is not dense"):
        _validate(connection)


def test_signal_naming_a_missing_axis_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES (0, 'sig1', 'I', 9, 0, 10)")
    with pytest.raises(TimeFValidationError, match="signal names an axis that does not exist"):
        _validate(connection)


def test_signal_naming_a_missing_spec_is_caught(connection):
    connection.execute("INSERT INTO signals VALUES (0, 'sig1', 'I', 0, 9, 10)")
    with pytest.raises(TimeFValidationError, match="signal names a spec that does not exist"):
        _validate(connection)


def test_one_signal_can_hang_off_several_sources(connection):
    """A series used by several records is stored once and linked many times."""
    connection.execute("INSERT INTO records VALUES (1, 'r2', NULL)")
    connection.execute("INSERT INTO sources VALUES (1, 's2', 1, NULL, '0000', 0, 'Monitor', 0)")
    _signal(connection, 0, external_id="shared")
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 0, 0, 0, 10)")
    connection.execute("INSERT INTO source_signals VALUES (0, 0, 0)")
    connection.execute("INSERT INTO source_signals VALUES (1, 0, 0)")
    _validate(connection)
    rows = connection.execute("SELECT count(*) FROM source_signals WHERE signal_id = 0").fetchone()
    assert rows is not None
    assert rows[0] == 2


# Chunks and the artifacts they name.


def test_chunk_naming_a_missing_signal_is_caught(connection):
    connection.execute("INSERT INTO signal_chunks VALUES (9, 0, 0, 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="chunk names a signal that does not exist"):
        _validate(connection)


# The locator is backend-neutral, so what a Parquet shard and a Zarr array each need out of it can
# no longer be said in a column type. These checks say it instead.


def test_chunk_naming_an_undeclared_artifact_is_caught(connection):
    _signal(connection)
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 9, 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="chunk names an artifact the values plane did not declare"):
        _validate(connection)


def test_parquet_chunk_without_a_row_offset_is_caught(connection):
    _signal(connection)
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 0, 0, NULL, 10)")
    with pytest.raises(TimeFValidationError, match="parquet chunk without a row offset"):
        _validate(connection)


def test_zarr_chunk_with_a_row_offset_is_caught(connection):
    connection.execute("INSERT INTO values_artifacts VALUES (1, 'time_series.zarr/ecg', 'zarr')")
    _signal(connection)
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 1, 128, 0, 10)")
    with pytest.raises(TimeFValidationError, match="zarr chunk with a row offset"):
        _validate(connection)


def test_a_zarr_chunk_needs_only_an_element_offset(connection):
    connection.execute("INSERT INTO values_artifacts VALUES (1, 'time_series.zarr/ecg', 'zarr')")
    _signal(connection)
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 1, 128, NULL, 10)")
    _validate(connection)


def test_an_unknown_values_backend_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO values_artifacts VALUES (1, 'x', 'hdf5')")


def test_signal_length_disagreeing_with_its_chunks_is_caught(connection):
    _signal(connection, n_values=99)
    connection.execute("INSERT INTO signal_chunks VALUES (0, 0, 0, 0, 0, 10)")
    with pytest.raises(TimeFValidationError, match="signal length disagrees with its chunks"):
        _validate(connection)


def test_a_signal_with_no_chunks_at_all_is_caught(connection):
    """The one signals/chunks mismatch a grouped inner join drops: no chunk row means no group.

    A consumer would read an empty array for a signal that declares 5,000 values, with nothing
    anywhere saying the values are missing.
    """
    _signal(connection, n_values=5_000)
    with pytest.raises(TimeFValidationError, match="signal length disagrees with its chunks"):
        _validate(connection)


def test_an_empty_signal_needs_no_chunks(connection):
    """A pin on the check's boundary, not a guard on the outer join that put it there.

    It passes with the inner join back too. The writer never reaches this state either:
    ``plan_chunks`` returns one chunk for a zero-length signal and the Zarr writer records
    ``n_values`` unconditionally, so a signal with no values still gets a chunk row. What this
    holds is the arithmetic, so a later rewrite of the check cannot start calling 0 against no
    chunks a disagreement.
    """
    _signal(connection, n_values=0)
    _validate(connection)


class _CountedRows:
    """One query's result, counting the rows the caller pulls out of it."""

    def __init__(self, result, counter):
        self._result = result
        self._counter = counter

    def fetchone(self):
        return self._result.fetchone()

    def fetchall(self):
        rows = self._result.fetchall()
        self._counter.rows_fetched += len(rows)
        return rows


class _CountingConnection:
    """A connection counting how many rows the caller pulls across into Python."""

    def __init__(self, connection):
        self._connection = connection
        self.rows_fetched = 0

    def execute(self, query):
        return _CountedRows(self._connection.execute(query), self)


def test_the_error_path_never_fetches_every_offending_row(connection):
    """A corrupt build of a multi-million-row corpus has to report, not run out of memory.

    Memory is not observable from here, so the rows that cross into Python are counted instead.
    Ten rows are wrong and three are fetched, so the database did the counting: the message costs
    the same whether ten rows are wrong or three million. Fetching them all to print three of them
    was a ``MemoryError`` on the one path that exists to explain what went wrong.
    """
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    for position in range(10):
        connection.execute("INSERT INTO task_items VALUES (0, 'input', ?, 'record', NULL, 99)", [position])
    counting = _CountingConnection(connection)
    with pytest.raises(TimeFValidationError, match=r"10 row\(s\), for example 0, 0, 0\."):
        _validate(cast(duckdb.DuckDBPyConnection, counting))
    assert counting.rows_fetched == 3


# Attachments, one table per target kind.


def test_annotation_on_a_missing_signal_is_caught(connection):
    connection.execute("INSERT INTO signal_annotations VALUES (0, 0, 9, 'static', NULL, NULL, NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="signal_annotations names a signal that does not exist"):
        _validate(connection)


def test_attachment_naming_a_missing_annotation_is_caught(connection):
    connection.execute("INSERT INTO record_annotations VALUES (0, 9, 0, 'static', NULL, NULL, NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="record_annotations names an annotation that does not exist"):
        _validate(connection)


def test_source_attachment_scope_naming_a_missing_record_is_caught(connection):
    connection.execute("INSERT INTO source_annotations VALUES (0, 0, 0, 9, 'static', NULL, NULL, NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="scope names a record that does not exist"):
        _validate(connection)


def test_source_attachment_scope_disagreeing_with_its_source_is_caught(connection):
    """The scope column is denormalized, so it has to agree with the source's own record."""
    connection.execute("INSERT INTO records VALUES (1, 'r2', NULL)")
    connection.execute("INSERT INTO source_annotations VALUES (0, 0, 0, 1, 'static', NULL, NULL, NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="scope disagrees with its source's record"):
        _validate(connection)


def test_an_interval_that_ends_before_it_starts_is_caught(connection):
    connection.execute("INSERT INTO record_annotations VALUES (0, 0, 0, 'interval', 10, 5, NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="interval annotation ends before it starts"):
        _validate(connection)


# Tasks and their items.


def test_task_item_naming_a_record_that_was_not_found_is_caught(connection):
    """An unresolved record name arrives here as a record item with no record id."""
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    connection.execute("INSERT INTO task_items VALUES (0, 'input', 0, 'record', NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="task item names a record that does not exist"):
        _validate(connection)


def test_task_item_naming_a_missing_record_id_is_caught(connection):
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    connection.execute("INSERT INTO task_items VALUES (0, 'input', 0, 'record', NULL, 9)")
    with pytest.raises(TimeFValidationError, match="task item names a record id that does not exist"):
        _validate(connection)


def test_task_item_with_no_text_is_caught(connection):
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    connection.execute("INSERT INTO task_items VALUES (0, 'input', 0, 'text', NULL, NULL)")
    with pytest.raises(TimeFValidationError, match="text item carries no text"):
        _validate(connection)


def test_task_item_naming_a_missing_task_is_caught(connection):
    connection.execute("INSERT INTO task_items VALUES (9, 'input', 0, 'text', 'x', NULL)")
    with pytest.raises(TimeFValidationError, match="task item names a task that does not exist"):
        _validate(connection)


# CHECK constraints cost no storage, so they stay declared and still fire at insert time.


def test_an_unknown_span_type_is_refused(connection):
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO record_annotations VALUES (0, 0, 0, 'banana', NULL, NULL, NULL, NULL)")


def test_task_item_with_a_bad_role_is_refused(connection):
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO task_items VALUES (0, 'sideways', 0, 'text', 'x', NULL)")


def test_task_item_of_an_unknown_kind_is_refused(connection):
    connection.execute("INSERT INTO tasks VALUES (0, 't1', 'Diagnose.')")
    with pytest.raises(duckdb.ConstraintException, match="CHECK"):
        connection.execute("INSERT INTO task_items VALUES (0, 'input', 0, 'banana', 'x', NULL)")


# One signal id, one series. Sharing a series across records is the point of source_signals, so the
# writer stores the payload once; that only holds together while every use of an id is the same
# series. Before the link table a second, different series hit a primary key and the build failed.


def _metric(signal_id, name, values, *, period_us=1_000_000, unit=""):
    """Build one standalone signal, so a test can vary exactly one of its parts."""
    return Signal(
        id=signal_id,
        name=name,
        values=np.asarray(values, dtype=np.float32),
        time_axis=RegularAxis(period_us=Fraction(period_us, 1)),
        spec=TimeSeriesSpec(spec_type="metric", name="metric", unit_value=ureg.Unit(unit), dtype="float32"),
    )


def _two_records(first, second):
    """Return a dataset whose two records each hold one of the given signals."""
    dataset = DeclarativeDataset(metadata=make_metadata())
    for index, signal in enumerate((first, second)):
        dataset.add_record(Record(id=f"r-{index}", sources=[Source(id=f"src-{index}", name="host", signals=[signal])]))
    return dataset


def test_the_same_series_under_one_id_is_stored_once(tmp_path):
    """A pin on sharing, which held before the check and has to keep holding after it.

    Two records referencing one series is designed behaviour: in ARFBench 7,013 of 9,187 are. The
    two signal objects here are equal but distinct, and carry equal but distinct annotations, so
    everything the check compares has to be compared by content and not by object identity. This
    passes with the check removed; what it guards is the check refusing too much.
    """
    dataset = _two_records(
        _metric("cpu", "cpu", range(8)).annotate(Annotation.static(name="lead", value="II")),
        _metric("cpu", "cpu", range(8)).annotate(Annotation.static(name="lead", value="II")),
    )
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with TimeFReader(tmp_path / dataset.metadata.dataset_id / "1.0.0") as reader:
        counts = reader.counts()
        assert counts["signals"] == 1
        assert counts["source_signals"] == 2
        assert counts["signal_annotations"] == 1


@pytest.mark.parametrize(
    ("second", "reported"),
    [
        (_metric("cpu", "cpu-b", range(8)), r"name 'cpu-a' then 'cpu-b'"),
        (_metric("cpu", "cpu-a", range(100, 108)), "the same shape, different values"),
        (_metric("cpu", "cpu-a", range(9)), "8 values then 9"),
        (_metric("cpu", "cpu-a", range(8), unit="V"), "a different spec"),
        (_metric("cpu", "cpu-a", range(8), period_us=500_000), "a different time axis"),
        (
            _metric("cpu", "cpu-a", range(8)).annotate(Annotation.static(name="lead", value="II")),
            "different annotations",
        ),
    ],
    ids=["name", "values", "length", "spec", "axis", "annotations"],
)
def test_one_id_used_for_two_different_series_is_refused(tmp_path, second, reported):
    dataset = _two_records(_metric("cpu", "cpu-a", range(8)), second)
    with (
        pytest.raises(TimeFValidationError, match=f"signal id 'cpu' is used for two different series: {reported}"),
        TimeFWriter(tmp_path, dataset.metadata) as writer,
    ):
        writer.write(dataset)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()
    assert not list(tmp_path.glob("**/*.tmp-*"))


def test_two_annotations_of_one_name_under_one_signal_id_are_refused(tmp_path):
    """Annotations hang off the signal, so the second use's were dropped and nothing said so.

    Everything else about these two agrees, so only the annotations can catch it. The two say
    different things about the same signal under the same name, which is the shape that reads as a
    plain overwrite rather than as a missing attachment.
    """
    dataset = _two_records(
        _metric("cpu", "cpu", range(8)).annotate(Annotation.static(name="label", value="first")),
        _metric("cpu", "cpu", range(8)).annotate(Annotation.static(name="label", value="second")),
    )
    with (
        pytest.raises(
            TimeFValidationError, match="signal id 'cpu' is used for two different series: different annotations"
        ),
        TimeFWriter(tmp_path, dataset.metadata) as writer,
    ):
        writer.write(dataset)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()


def test_one_id_whose_annotations_differ_only_in_their_span_is_refused(tmp_path):
    """The attachment's own columns are part of the annotation, not just the name/value/unit payload.

    Two attachments of one payload are stored once and pointed at twice, so a check that stopped at
    ``content_key`` would call these the same and drop the second interval.
    """
    dataset = _two_records(
        _metric("cpu", "cpu", range(8)).annotate(
            Annotation.interval(name="artefact", value="motion", start_us=0, end_us=1_000)
        ),
        _metric("cpu", "cpu", range(8)).annotate(
            Annotation.interval(name="artefact", value="motion", start_us=4_000, end_us=5_000)
        ),
    )
    with (
        pytest.raises(
            TimeFValidationError, match="signal id 'cpu' is used for two different series: different annotations"
        ),
        TimeFWriter(tmp_path, dataset.metadata) as writer,
    ):
        writer.write(dataset)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()


# the writer's own guards


def test_a_source_position_that_does_not_fit_its_path_is_refused():
    with pytest.raises(TimeFValidationError, match="does not fit 4 digits"):
        ddl.source_path(None, 10_000)


def test_a_quote_in_the_output_path_does_not_break_the_attach(tmp_path):
    """A home directory such as /Users/o'brien would otherwise end the ATTACH literal early."""
    dataset = make_dataset(n_records=1, n_values=32)
    root = tmp_path / "o'brien"
    root.mkdir()
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)
    assert (root / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()


def test_writer_refuses_to_overwrite_a_committed_version(tmp_path):
    dataset = make_dataset(n_records=1, n_values=32)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with pytest.raises(TimeFValidationError, match="already holds a committed version"):
        TimeFWriter(tmp_path, dataset.metadata)


def test_a_task_naming_a_record_that_does_not_exist_is_refused(tmp_path):
    """A task can name a record it never held, so the name is only checked once the load is in."""
    dataset = DeclarativeDataset(metadata=make_metadata())
    dataset.add_record(make_record("record-000", n_values=32))
    dataset.add_task(Task(id="orphan", prompt="Diagnose.", inputs=[RecordRef("no-such-record")]))
    with (
        pytest.raises(TimeFValidationError, match="task item names a record that does not exist"),
        TimeFWriter(tmp_path, dataset.metadata) as writer,
    ):
        writer.write(dataset)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0" / "manifest.json").exists()


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
