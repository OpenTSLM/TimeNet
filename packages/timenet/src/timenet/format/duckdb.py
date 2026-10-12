"""DuckDB schema and connection helpers for the TimeF relational control plane."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.control_schema import schema_ddl


CONTROL_FILE = "control.duckdb"
"""Name of the relational control-plane database in a TimeF version."""

CONTROL_SCHEMA_VERSION = 1
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


class ControlSession:
    """Share one DuckDB instance across read-only control databases."""

    def __init__(self) -> None:
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._databases: dict[Path, str] = {}
        self._readers = 0
        self._lock = RLock()

    def __getstate__(self) -> dict:
        return {}

    def __setstate__(self, _state: dict) -> None:
        self.__init__()

    def open(self, path: Path) -> duckdb.DuckDBPyConnection:
        """Open an independent cursor whose default database is ``path``.

        Returns:
            A read-only cursor to release with :meth:`close`.

        Raises:
            duckdb.Error: If the database cannot be opened or attached.
        """
        path = path.resolve()
        with self._lock:
            if self._connection is None:
                # A private instance keeps attachments separate from other readers of the same file.
                self._connection = connect_control(Path(":memory:"))
            connection = None
            try:
                if path not in self._databases:
                    database = f"timenet_parent_{len(self._databases)}"
                    escaped = str(path).replace("'", "''")
                    self._connection.execute(f"ATTACH '{escaped}' AS {database} (READ_ONLY)")
                    self._databases[path] = database
                connection = self._connection.cursor()
                connection.execute(f"USE {self._databases[path]}")
            except duckdb.Error:
                if connection is not None:
                    connection.close()
                self._close_idle_connection()
                raise
            self._readers += 1
            return connection

    def close(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Release a cursor and close the instance after its last reader."""
        with self._lock:
            connection.close()
            self._readers -= 1
            self._close_idle_connection()

    def _close_idle_connection(self) -> None:
        """Release the instance when no reader holds a cursor."""
        if self._readers == 0 and self._connection is not None:
            self._connection.close()
            self._connection = None
            self._databases.clear()


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
        TimeFFormatError: If the schema metadata is missing, malformed, or unsupported.
    """
    try:
        row = connection.execute("SELECT value FROM control_metadata WHERE key = 'schema_version'").fetchone()
    except duckdb.Error as exc:
        raise TimeFFormatError("control.duckdb does not contain valid schema metadata") from exc
    if row is None or row[0] != str(CONTROL_SCHEMA_VERSION):
        found = None if row is None else row[0]
        raise TimeFFormatError(f"unsupported control schema version {found!r}; expected {CONTROL_SCHEMA_VERSION}")
    return int(row[0])


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
