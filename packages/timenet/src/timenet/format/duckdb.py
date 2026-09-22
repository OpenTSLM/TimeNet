"""DuckDB schema and connection helpers for the TimeF relational control plane."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.control_schema import schema_ddl


CONTROL_FILE = "control.duckdb"
"""Name of the relational control-plane database in a TimeF version."""

CONTROL_SCHEMA_VERSION = 2
"""Schema version written into :data:`CONTROL_FILE`."""


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
        connection.execute(schema_ddl())
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
