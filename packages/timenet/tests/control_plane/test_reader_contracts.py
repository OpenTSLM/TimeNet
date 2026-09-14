"""What the reader promises about a database it did not write itself.

A control database can arrive from anywhere: a registry, a download, a writer one release ahead.
These tests pin down what the reader does with one it cannot read, and what it hands back for the
reads that have no bytes behind them. The rule running through them is that the reader says what is
wrong with the file rather than defaulting its way past it: a missing axis column is not a zero, and
an empty window still carries the dtype its signal declares.
"""

from fractions import Fraction
import os

import duckdb
import numpy as np
import pytest

from timenet.control_plane import (
    DeclarativeDataset,
    Record,
    Signal,
    Source,
    TimeFReader,
    TimeFWriter,
    schema as ddl,
)
from timenet.control_plane.reader import _authorize, _axis_of, _sql_identifier, _sql_literal
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.constants import CONTROL_DB_FILE
from timenet.testing import make_dataset, make_metadata
from timenet.types import TimeSeriesSpec, ureg


STAGE_SPEC = TimeSeriesSpec(
    spec_type="sleep_stage", name="Sleep stage", unit_value=ureg.Unit("dimensionless"), dtype="str"
)
EEG_SPEC = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.Unit("uV"), dtype="float32")
EPOCH_AXIS = RegularAxis(period_us=Fraction(30_000_000))
_SET_VERSION = "UPDATE meta SET value = ? WHERE key = 'schema_version'"


@pytest.fixture
def version(tmp_path):
    """Write the fixture dataset and return its committed version directory."""
    dataset = make_dataset(n_records=2, n_values=64)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def text_version(tmp_path):
    """Write one record holding a text signal beside a float one, and return its version."""
    stages = Signal(
        id="stages",
        name="Sleep stages",
        values=np.array(["W", "N1", "N2", "N3"], dtype=object),
        time_axis=EPOCH_AXIS,
        spec=STAGE_SPEC,
    )
    eeg = Signal(
        id="eeg",
        name="EEG Fpz-Cz",
        values=np.arange(4, dtype=np.float32),
        time_axis=EPOCH_AXIS,
        spec=EEG_SPEC,
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages, eeg])])],
    )
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _edit(version, statement, parameters=()):
    """Run one statement against a written control database, as a stray writer would have."""
    connection = duckdb.connect(str(version / CONTROL_DB_FILE))
    connection.execute(statement, list(parameters))
    connection.close()


def _signal_id(reader, external_id):
    """Return the surrogate id of the one signal built under this id."""
    (signal,) = [found for found in reader.record("night-000").signals() if found.external_id == external_id]
    return signal.signal_id


def test_a_newer_schema_version_is_refused_by_name(version):
    """A database one release ahead must say so, not die on the first missing table."""
    _edit(version, _SET_VERSION, [str(ddl.SCHEMA_VERSION + 1)])
    with pytest.raises(TimeFFormatError) as raised:
        TimeFReader(version)
    assert str(ddl.SCHEMA_VERSION + 1) in str(raised.value)
    assert str(ddl.SCHEMA_VERSION) in str(raised.value)


def test_an_older_schema_version_is_refused_too(version):
    """The format is pre-release, so there is no migration and no older version to read."""
    _edit(version, _SET_VERSION, [str(ddl.SCHEMA_VERSION - 1)])
    with pytest.raises(TimeFFormatError, match="rebuild the dataset"):
        TimeFReader(version)


def test_a_refused_database_is_not_left_open(version):
    """__init__ opens the file before it checks it, and nothing closes a reader that never built."""
    _edit(version, _SET_VERSION, [str(ddl.SCHEMA_VERSION + 1)])
    with pytest.raises(TimeFFormatError) as raised:
        TimeFReader(version)
    assert "rebuild the dataset" in str(raised.value)
    # The traceback held above still references the reader, so a handle it kept is still open. A
    # second connection to a file already open read-only is what DuckDB refuses, which is the check.
    duckdb.connect(str(version / CONTROL_DB_FILE)).close()


def test_a_regular_axis_missing_its_period_is_a_format_error():
    with pytest.raises(TimeFFormatError, match="period_denominator=None"):
        _axis_of("regular", 1_000_000, None, 0, None, None)


def test_a_regular_axis_missing_its_start_index_is_a_format_error():
    with pytest.raises(TimeFFormatError, match="start_index=None"):
        _axis_of("regular", 1_000_000, 1, None, None, None)


def test_an_irregular_axis_missing_an_endpoint_is_a_format_error():
    """The endpoint that used to be defaulted to 0, which made the axis run backwards."""
    with pytest.raises(TimeFFormatError, match="last_us=None"):
        _axis_of("irregular", None, None, None, 5_000_000, None)


def test_an_unknown_axis_type_is_a_format_error():
    with pytest.raises(TimeFFormatError, match="'sundial'"):
        _axis_of("sundial", None, None, None, None, None)


def test_a_complete_axis_row_rebuilds_the_axis_it_was_written_from():
    assert _axis_of("regular", 1_000_000, 4, 7, None, None) == RegularAxis(
        period_us=Fraction(1_000_000, 4), start_index=7
    )
    assert _axis_of("irregular", None, None, None, 5, 11) == IrregularAxis(first_us=5, last_us=11)
    assert _axis_of("ordinal", None, None, None, None, None) == OrdinalAxis()


def test_a_broken_axes_row_stops_the_record_read_that_meets_it(version):
    """A reader that cannot rebuild an axis must not hand back a record placed somewhere else."""
    _edit(version, "UPDATE axes SET last_us = NULL, first_us = 5, axis_type = 'irregular'")
    with TimeFReader(version) as reader, pytest.raises(TimeFFormatError, match="irregular axis"):
        reader.record("record-000")


def test_an_empty_window_carries_the_dtype_the_signal_declares(text_version):
    """An empty window reads no bytes, so its dtype has to come from the spec, text included."""
    with TimeFReader(text_version) as reader:
        stages = _signal_id(reader, "stages")
        whole = reader.values(stages)
        assert whole.dtype == np.dtype(object)
        assert reader.values_window(stages, 1, 3).dtype == whole.dtype
        assert reader.values_window(stages, 2, 2).dtype == whole.dtype
        assert reader.values_window(stages, 4, 9).dtype == whole.dtype
        assert reader.values_window(stages, 2, 2).tolist() == []


def test_an_empty_window_of_a_numeric_signal_keeps_its_own_dtype(text_version):
    with TimeFReader(text_version) as reader:
        eeg = _signal_id(reader, "eeg")
        assert reader.values(eeg).dtype == np.dtype("float32")
        assert reader.values_window(eeg, 1, 1).dtype == np.dtype("float32")


def test_a_signal_whose_chunks_are_gone_reads_back_in_its_own_dtype_either_way(text_version):
    """With no chunk row, both reads have only the spec to take a dtype from, so both must use it."""
    _edit(text_version, "DELETE FROM signal_chunks")
    with TimeFReader(text_version) as reader:
        stages = _signal_id(reader, "stages")
        eeg = _signal_id(reader, "eeg")
        assert reader.values(stages).dtype == np.dtype(object)
        assert reader.values_window(stages, 0, 4).dtype == np.dtype(object)
        assert reader.values(eeg).dtype == np.dtype("float32")
        assert reader.values_window(eeg, 0, 4).dtype == np.dtype("float32")


def test_a_worker_slice_that_is_not_a_slice_raises_timenets_own_error(version):
    with TimeFReader(version) as reader:
        with pytest.raises(TimeFValidationError):
            next(reader.iter_records(num_workers=0))
        # TimeFValidationError is a ValueError, so a caller that already catches one still does.
        with pytest.raises(ValueError, match="is not a slice of the corpus"):
            next(reader.iter_records(worker_index=2, num_workers=2))
        with pytest.raises(TimeFValidationError, match="batch_size"):
            next(reader.iter_tasks(batch_size=0))


def test_a_quote_in_a_path_survives_into_the_sql_literal():
    assert _sql_literal("/Users/o'brien/control.duckdb") == "/Users/o''brien/control.duckdb"
    assert _sql_identifier('a "name"') == '"a ""name"""'


def test_an_s3_url_without_credentials_reads_anonymously(monkeypatch, tmp_path):
    """DuckDB validates a credential chain on CREATE SECRET, and a public bucket needs no chain."""
    for name in [key for key in os.environ if key.startswith("AWS_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    # Redirecting the config files is not enough on a runner with an instance role: the chain would
    # reach the metadata service and resolve credentials this test needs it not to have.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    # The environment this test needs is one where the chain resolves nothing. Say so, so the test
    # fails loudly on a machine that is signed in rather than passing without proving anything.
    with pytest.raises(duckdb.Error, match="Secret"):
        connection.execute("CREATE SECRET IF NOT EXISTS (TYPE s3, PROVIDER credential_chain)")

    _authorize(connection, "s3://open-bucket/dataset/control.duckdb")

    assert connection.execute("SELECT count(*) FROM duckdb_secrets()").fetchone() == (0,)
    connection.close()


def test_a_gs_url_is_never_given_an_s3_secret():
    """An s3-scoped secret can never apply to a gs:// URL, so the read goes out unsigned instead."""
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    _authorize(connection, "gs://open-bucket/dataset/control.duckdb")
    assert connection.execute("SELECT count(*) FROM duckdb_secrets()").fetchone() == (0,)
    connection.close()


def test_an_s3_url_whose_secret_cannot_be_created_at_all_is_refused(tmp_path):
    """Only an empty chain is the anonymous read. A chain DuckDB could not even run is a setup bug.

    Suppressed, it comes back later as DuckDB's "database does not exist" on a bucket the caller is
    signed in to, which says nothing about the credentials that never reached the request.
    """
    connection = duckdb.connect()
    connection.execute(f"SET extension_directory = '{tmp_path}'")
    connection.execute("SET custom_extension_repository = 'http://127.0.0.1:9/'")
    with pytest.raises(TimeFFormatError, match="credential secret") as raised:
        _authorize(connection, "s3://private-bucket/dataset/control.duckdb")
    assert "aws" in str(raised.value)
    assert connection.execute("SELECT count(*) FROM duckdb_secrets()").fetchone() == (0,)
    connection.close()


def test_an_azure_url_without_the_azure_extension_is_refused(tmp_path):
    """Without the extension DuckDB has no filesystem for az://, which is worth saying plainly."""
    connection = duckdb.connect()
    connection.execute(f"SET extension_directory = '{tmp_path}'")
    connection.execute("SET custom_extension_repository = 'http://127.0.0.1:9/'")
    with pytest.raises(TimeFFormatError, match="azure extension"):
        _authorize(connection, "az://container/dataset/control.duckdb")
    connection.close()
