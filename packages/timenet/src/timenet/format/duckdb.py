"""DuckDB schema and connection helpers for the TimeF relational control plane."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import re

import duckdb

from timenet.errors import FormatVersionSubject, TimeFFormatError, TimeFFormatVersionError
from timenet.format.control_schema import schema_ddl


CONTROL_FILE = "control.duckdb"
"""Name of the relational control-plane database in a TimeF version."""

CONTROL_SCHEMA_VERSION = 2
"""Schema version written into :data:`CONTROL_FILE`, and the newest that this TimeNet reads."""

OLDEST_READABLE_CONTROL_SCHEMA_VERSION = 2
"""The oldest schema version of :data:`CONTROL_FILE` that this TimeNet reads.

Version 2 is the schema of TimeNet 0.12. Version 1 names layouts used during early development, and no TimeNet reads them.
"""

_IN_MEMORY_DATABASE = "an in-memory control database"
"""What an error names in place of a file for a control database without one."""

CATALOG_ERRORS = (duckdb.CatalogException, duckdb.BinderException, duckdb.ConversionException)
"""DuckDB errors that mean a catalog lacks a table, column, or type that a read expects."""


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


def check_control_schema(connection: duckdb.DuckDBPyConnection) -> int:
    """Return the version of a supported control database.

    Args:
        connection: An open TimeF control database.

    Raises:
        TimeFFormatVersionError: If the schema is older or newer than this TimeNet reads.
        TimeFFormatError: If the schema metadata is missing or malformed.
    """
    try:
        row = connection.execute("SELECT value FROM control_metadata WHERE key = 'schema_version'").fetchone()
    except duckdb.Error as exc:
        raise TimeFFormatError("control.duckdb does not contain valid schema metadata") from exc
    found = None if row is None else row[0]
    # The writer stores str(version), so a version has no sign, no leading zero, and no other digits.
    if not isinstance(found, str) or re.fullmatch(r"[1-9][0-9]*", found) is None:
        raise TimeFFormatError(f"unsupported control schema version {found!r}; expected {CONTROL_SCHEMA_VERSION}")
    version = int(found)
    if not OLDEST_READABLE_CONTROL_SCHEMA_VERSION <= version <= CONTROL_SCHEMA_VERSION:
        file = _database_file(connection)
        raise TimeFFormatVersionError(
            FormatVersionSubject.CONTROL_SCHEMA,
            _IN_MEMORY_DATABASE if file is None else file,
            version,
            OLDEST_READABLE_CONTROL_SCHEMA_VERSION,
            CONTROL_SCHEMA_VERSION,
            None if file is None else Path(file),
        )
    return version


def _database_file(connection: duckdb.DuckDBPyConnection) -> str | None:
    """Return the file of the current database of a connection, or ``None`` for an in-memory database."""
    row = connection.execute("SELECT path FROM duckdb_databases() WHERE database_name = current_database()").fetchone()
    return None if row is None else row[0]


@contextmanager
def transaction(connection: duckdb.DuckDBPyConnection) -> Generator[None]:
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
