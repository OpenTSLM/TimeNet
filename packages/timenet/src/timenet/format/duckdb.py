"""DuckDB schema and connection helpers for the TimeF relational control plane."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from timenet.errors import TimeFFormatError


CONTROL_FILE = "control.duckdb"
"""Name of the relational control-plane database in a TimeF version."""

CONTROL_SCHEMA_VERSION = 2
"""Schema version written into :data:`CONTROL_FILE`."""


_SCHEMA = """
CREATE TABLE control_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE records (
    record_id VARCHAR PRIMARY KEY,
    start_time_us BIGINT,
    time_span_start_us BIGINT,
    time_span_end_us BIGINT,
    metadata JSON NOT NULL
);

CREATE TABLE sources (
    source_id VARCHAR PRIMARY KEY,
    record_id VARCHAR NOT NULL REFERENCES records(record_id),
    parent_source_id VARCHAR REFERENCES sources(source_id),
    name VARCHAR NOT NULL,
    metadata JSON NOT NULL,
    CHECK (parent_source_id IS NULL OR parent_source_id <> source_id)
);

CREATE TABLE axes (
    axis_id VARCHAR PRIMARY KEY,
    axis_type VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator BIGINT,
    origin_us BIGINT,
    first_us BIGINT,
    last_us BIGINT
);

CREATE TABLE axis_offsets (
    axis_id VARCHAR NOT NULL REFERENCES axes(axis_id),
    position BIGINT NOT NULL,
    offset_us BIGINT NOT NULL,
    PRIMARY KEY (axis_id, position)
);

CREATE TABLE signals (
    signal_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL REFERENCES sources(source_id),
    name VARCHAR NOT NULL,
    axis_id VARCHAR NOT NULL REFERENCES axes(axis_id),
    spec_type VARCHAR NOT NULL,
    spec_name VARCHAR NOT NULL,
    unit VARCHAR,
    dtype VARCHAR NOT NULL,
    categories JSON NOT NULL,
    value_shape JSON NOT NULL,
    dimension_names JSON NOT NULL,
    nullable BOOLEAN NOT NULL,
    n_values BIGINT NOT NULL CHECK (n_values > 0),
    metadata JSON NOT NULL
);

CREATE TABLE signal_chunks (
    signal_id VARCHAR NOT NULL REFERENCES signals(signal_id),
    chunk_index BIGINT NOT NULL,
    value_path VARCHAR NOT NULL,
    chunk_major_index BIGINT NOT NULL,
    chunk_minor_index BIGINT,
    n_values BIGINT NOT NULL CHECK (n_values > 0),
    PRIMARY KEY (signal_id, chunk_index)
);

CREATE TABLE annotation_contents (
    content_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    value JSON,
    unit VARCHAR,
    metadata JSON NOT NULL
);

CREATE TABLE annotation_occurrences (
    occurrence_id VARCHAR PRIMARY KEY,
    content_id VARCHAR NOT NULL REFERENCES annotation_contents(content_id),
    object_type VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    span_type VARCHAR NOT NULL,
    start_us BIGINT,
    end_us BIGINT,
    signal_ids JSON,
    provenance JSON,
    confidence DOUBLE,
    metadata JSON NOT NULL,
    CHECK (object_type IN ('Dataset', 'Task', 'Record', 'Source', 'Signal')),
    CHECK (span_type IN ('static', 'point', 'interval'))
);

CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    task_type VARCHAR NOT NULL,
    prompt VARCHAR,
    scope JSON,
    payload JSON NOT NULL,
    rationale VARCHAR,
    metadata JSON NOT NULL
);

CREATE TABLE task_record_refs (
    task_id VARCHAR NOT NULL REFERENCES tasks(task_id),
    field VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    record_id VARCHAR NOT NULL REFERENCES records(record_id),
    PRIMARY KEY (task_id, field, position)
);

CREATE TABLE task_signal_refs (
    task_id VARCHAR NOT NULL REFERENCES tasks(task_id),
    field VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    signal_id VARCHAR NOT NULL REFERENCES signals(signal_id),
    PRIMARY KEY (task_id, field, position)
);

CREATE TABLE task_annotation_refs (
    task_id VARCHAR NOT NULL REFERENCES tasks(task_id),
    field VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    occurrence_id VARCHAR NOT NULL REFERENCES annotation_occurrences(occurrence_id),
    PRIMARY KEY (task_id, field, position)
);

CREATE TABLE task_dependencies (
    task_id VARCHAR NOT NULL REFERENCES tasks(task_id),
    position BIGINT NOT NULL,
    parent_task_id VARCHAR NOT NULL REFERENCES tasks(task_id),
    PRIMARY KEY (task_id, position),
    CHECK (task_id <> parent_task_id)
);
"""


def connect_control(path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a TimeF control database.

    Args:
        path: Local path to ``control.duckdb``.
        read_only: Open an immutable published database without write access.

    Returns:
        An open DuckDB connection owned by the caller.
    """
    return duckdb.connect(str(path), read_only=read_only)


def create_control_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create every control-plane table in one transaction.

    Args:
        connection: A writable connection to a new database.
    """
    with transaction(connection):
        connection.execute(_SCHEMA)
        connection.execute(
            "INSERT INTO control_metadata VALUES (?, ?)",
            ["schema_version", str(CONTROL_SCHEMA_VERSION)],
        )


def check_control_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject a control database with an unsupported schema version.

    Args:
        connection: An open TimeF control database.

    Raises:
        TimeFFormatError: If the schema metadata is missing, malformed, or unsupported.
    """
    try:
        row = connection.execute("SELECT value FROM control_metadata WHERE key = 'schema_version'").fetchone()
    except duckdb.Error as exc:
        raise TimeFFormatError("control.duckdb does not contain valid schema metadata") from exc
    if row is None or row[0] != str(CONTROL_SCHEMA_VERSION):
        found = None if row is None else row[0]
        raise TimeFFormatError(f"unsupported control schema version {found!r}; expected {CONTROL_SCHEMA_VERSION}")


@contextmanager
def transaction(connection: duckdb.DuckDBPyConnection) -> Iterator[None]:
    """Commit a block atomically or roll it back when any operation fails.

    Args:
        connection: The connection that owns the transaction.

    Yields:
        Control while the transaction is open.
    """
    connection.begin()
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
