"""DuckDB schema and connection helpers for the TimeF relational control plane."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from timenet.errors import TimeFFormatError


CONTROL_FILE = "control.duckdb"
"""Name of the relational control-plane database in a TimeF version."""

CONTROL_SCHEMA_VERSION = 5
"""Schema version written into :data:`CONTROL_FILE`."""


_SCHEMA = """
CREATE TABLE control_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE SEQUENCE object_key_sequence START 1;
CREATE SEQUENCE axis_key_sequence START 1;
CREATE SEQUENCE content_key_sequence START 1;
CREATE SEQUENCE occurrence_key_sequence START 1;
CREATE SEQUENCE target_item_key_sequence START 1;

CREATE TABLE datasets (
    dataset_key BIGINT PRIMARY KEY DEFAULT nextval('object_key_sequence'),
    dataset_id VARCHAR NOT NULL UNIQUE
);

CREATE TABLE records (
    record_key BIGINT PRIMARY KEY DEFAULT nextval('object_key_sequence'),
    record_id VARCHAR NOT NULL UNIQUE,
    start_time_us BIGINT,
    time_span_start_us BIGINT,
    time_span_end_us BIGINT,
    metadata JSON NOT NULL
);

CREATE TABLE sources (
    source_key BIGINT PRIMARY KEY DEFAULT nextval('object_key_sequence'),
    source_id VARCHAR NOT NULL UNIQUE,
    record_key BIGINT NOT NULL REFERENCES records(record_key),
    parent_source_key BIGINT REFERENCES sources(source_key),
    name VARCHAR NOT NULL,
    metadata JSON NOT NULL,
    CHECK (parent_source_key IS NULL OR parent_source_key <> source_key)
);

CREATE TABLE axes (
    axis_key BIGINT PRIMARY KEY DEFAULT nextval('axis_key_sequence'),
    axis_id VARCHAR NOT NULL UNIQUE,
    axis_type VARCHAR NOT NULL,
    period_numerator_us BIGINT,
    period_denominator BIGINT,
    origin_us BIGINT,
    first_us BIGINT,
    last_us BIGINT
);

CREATE TABLE axis_offsets (
    axis_key BIGINT NOT NULL REFERENCES axes(axis_key),
    position BIGINT NOT NULL,
    offset_us BIGINT NOT NULL,
    PRIMARY KEY (axis_key, position)
);

CREATE TABLE signals (
    signal_key BIGINT PRIMARY KEY DEFAULT nextval('object_key_sequence'),
    signal_id VARCHAR NOT NULL UNIQUE,
    source_key BIGINT NOT NULL REFERENCES sources(source_key),
    name VARCHAR NOT NULL,
    axis_key BIGINT NOT NULL REFERENCES axes(axis_key),
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
    signal_key BIGINT NOT NULL REFERENCES signals(signal_key),
    chunk_index BIGINT NOT NULL,
    value_path VARCHAR NOT NULL,
    chunk_major_index BIGINT NOT NULL,
    chunk_minor_index BIGINT,
    n_values BIGINT NOT NULL CHECK (n_values > 0),
    PRIMARY KEY (signal_key, chunk_index)
);

CREATE TABLE annotation_contents (
    content_key BIGINT PRIMARY KEY DEFAULT nextval('content_key_sequence'),
    content_id VARCHAR NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    value JSON,
    unit VARCHAR,
    metadata JSON NOT NULL
);

CREATE TABLE annotation_occurrences (
    occurrence_key BIGINT PRIMARY KEY DEFAULT nextval('occurrence_key_sequence'),
    occurrence_id VARCHAR NOT NULL UNIQUE,
    content_key BIGINT NOT NULL REFERENCES annotation_contents(content_key),
    object_type VARCHAR NOT NULL,
    object_key BIGINT NOT NULL,
    span_type VARCHAR NOT NULL,
    start_us BIGINT,
    end_us BIGINT,
    signal_keys BIGINT[],
    provenance JSON,
    confidence DOUBLE,
    metadata JSON NOT NULL,
    CHECK (object_type IN ('Dataset', 'Task', 'Record', 'Source', 'Signal')),
    CHECK (span_type IN ('static', 'point', 'interval'))
);

CREATE TABLE tasks (
    task_key BIGINT PRIMARY KEY DEFAULT nextval('object_key_sequence'),
    task_id VARCHAR NOT NULL UNIQUE,
    task_type VARCHAR NOT NULL,
    prompt VARCHAR,
    scope JSON,
    has_inline_targets BOOLEAN NOT NULL,
    payload JSON NOT NULL,
    rationale VARCHAR,
    metadata JSON NOT NULL
);

CREATE TABLE target_items (
    target_item_key BIGINT PRIMARY KEY DEFAULT nextval('target_item_key_sequence'),
    target_kind VARCHAR NOT NULL,
    CHECK (target_kind IN (
        'text', 'integer', 'float', 'boolean', 'record', 'signal',
        'time_point', 'time_interval', 'step_point', 'step_interval'
    ))
);

CREATE TABLE target_text_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    value VARCHAR NOT NULL
);

CREATE TABLE target_integer_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    value BIGINT NOT NULL
);

CREATE TABLE target_float_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    value DOUBLE NOT NULL
);

CREATE TABLE target_boolean_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    value BOOLEAN NOT NULL
);

CREATE TABLE target_record_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    record_key BIGINT NOT NULL REFERENCES records(record_key)
);

CREATE TABLE target_signal_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    signal_key BIGINT NOT NULL REFERENCES signals(signal_key)
);

CREATE TABLE target_span_values (
    target_item_key BIGINT PRIMARY KEY REFERENCES target_items(target_item_key),
    span_start BIGINT NOT NULL,
    span_end BIGINT,
    signal_keys BIGINT[]
);

CREATE TABLE task_targets (
    task_key BIGINT NOT NULL REFERENCES tasks(task_key),
    position BIGINT NOT NULL,
    target_item_key BIGINT NOT NULL UNIQUE REFERENCES target_items(target_item_key),
    PRIMARY KEY (task_key, position)
);

CREATE TABLE task_record_refs (
    task_key BIGINT NOT NULL REFERENCES tasks(task_key),
    field VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    record_key BIGINT NOT NULL REFERENCES records(record_key),
    PRIMARY KEY (task_key, field, position)
);

CREATE TABLE task_annotation_refs (
    task_key BIGINT NOT NULL REFERENCES tasks(task_key),
    field VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    occurrence_key BIGINT NOT NULL REFERENCES annotation_occurrences(occurrence_key),
    PRIMARY KEY (task_key, field, position)
);

CREATE TABLE task_dependencies (
    task_key BIGINT NOT NULL REFERENCES tasks(task_key),
    position BIGINT NOT NULL,
    parent_task_key BIGINT NOT NULL REFERENCES tasks(task_key),
    PRIMARY KEY (task_key, position),
    CHECK (task_key <> parent_task_key)
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
