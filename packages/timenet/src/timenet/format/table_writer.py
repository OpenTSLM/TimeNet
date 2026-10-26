"""Insert catalog rows using their shared table definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import duckdb
import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.format.control_schema import Table


if TYPE_CHECKING:
    from timenet.types._model import TimeFModel


_Row = TypeVar("_Row", bound="TimeFModel")


def insert_rows(connection: duckdb.DuckDBPyConnection, table: Table, rows: Sequence[Mapping[str, object]]) -> None:
    """Insert one Arrow batch using the catalog table's schema and column order.

    Raises:
        TimeFValidationError: If the rows cannot be converted or inserted into the table.
    """
    if not rows:
        return
    relation_name = f"_timenet_{table.name}_batch"
    names = ", ".join(table.column_names)
    try:
        connection.register(relation_name, pa.Table.from_pylist(rows, schema=table.arrow_schema))
        try:
            connection.execute(
                f"INSERT INTO {table.name} ({names}) SELECT {names} FROM {relation_name}"  # noqa: S608
            )
        finally:
            connection.unregister(relation_name)
    except (duckdb.Error, pa.ArrowException) as exc:
        raise TimeFValidationError(f"could not write catalog table {table.name!r}: {exc}") from exc


@dataclass(frozen=True)
class ModelTable(Generic[_Row]):
    """Bind validated model rows to one catalog table definition."""

    definition: Table

    def write(self, connection: duckdb.DuckDBPyConnection, rows: Sequence[_Row]) -> None:
        """Serialize model rows and insert them through the shared catalog writer."""
        insert_rows(connection, self.definition, [row.model_dump(mode="json") for row in rows])
