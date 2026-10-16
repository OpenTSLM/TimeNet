"""Cached Parquet reads shared by Signal chunks and packed field values."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


@dataclass(frozen=True)
class _DecodedTable:
    """Keep the table and its resolved columns over the same Arrow buffers."""

    table: pa.Table
    columns: dict[str, pa.Array]


class ParquetReadCache:
    """Cache decoded tables and keep row-group file handles open until close."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._tables: OrderedDict[tuple[str, str, int | None, tuple[str, ...] | None], _DecodedTable] = OrderedDict()
        self._files: dict[tuple[str, str], pq.ParquetFile] = {}

    def __getstate__(self) -> dict:
        return {"_max_entries": self._max_entries, "_tables": OrderedDict(), "_files": {}}

    def read_table(
        self,
        version: "DatasetVersion",
        path: str,
        *,
        row_group: int | None = None,
        columns: tuple[str, ...] | None = None,
    ) -> pa.Table:
        """Read a complete file or projected row group, reusing recently decoded tables.

        Args:
            version: The dataset version that stores the file.
            path: The file path relative to the version root.
            row_group: The row-group index, or None to read the complete file.
            columns: The columns to decode, or None to read every column.

        Returns:
            A cached table with each column's chunks combined once.
        """
        return self._read(version, path, row_group, columns).table

    def read_column(self, version: "DatasetVersion", path: str, row_group: int, column: str) -> pa.Array:
        """Read one row-group column, resolving its Arrow array only on a cache miss.

        Args:
            version: The dataset version that stores the file.
            path: The file path relative to the version root.
            row_group: The row-group index.
            column: The column to decode.

        Returns:
            The cached column array.
        """
        return self._read(version, path, row_group, (column,)).columns[column]

    def _read(
        self, version: "DatasetVersion", path: str, row_group: int | None, columns: tuple[str, ...] | None
    ) -> _DecodedTable:
        """Load and cache one read selection.

        Returns:
            The decoded table and its resolved column arrays.

        Raises:
            TimeFFormatError: If the file, row group, or requested columns cannot be read.
        """
        key = (version.root, path, row_group, columns)
        cached = self._tables.get(key)
        if cached is not None:
            self._tables.move_to_end(key)
            return cached
        try:
            projection = None if columns is None else list(columns)
            if row_group is None:
                with version.filesystem.open_input_file(version.resolve_file_path(path)) as source:
                    table = pq.ParquetFile(source).read(columns=projection, use_threads=False)
            else:
                table = self._open_file(version, path).read_row_group(row_group, columns=projection)
            if columns is not None and any(column not in table.column_names for column in columns):
                raise TimeFFormatError(f"missing Parquet columns {columns} in {version.resolve_file_path(path)}")
            table = table.combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(
                f"cannot read Parquet file {version.resolve_file_path(path)}, row group {row_group}: {exc}"
            ) from exc
        arrays = {
            name: values.chunk(0) if values.num_chunks else pa.array([], type=values.type)
            for name, values in zip(table.column_names, table.columns, strict=True)
        }
        decoded = _DecodedTable(table, arrays)
        self._tables[key] = decoded
        while len(self._tables) > self._max_entries:
            self._tables.popitem(last=False)
        return decoded

    def _open_file(self, version: "DatasetVersion", path: str) -> pq.ParquetFile:
        """Reuse a file handle across projected row-group reads.

        Returns:
            The cached or newly opened Parquet file.
        """
        key = (version.root, path)
        if key not in self._files:
            # Coalesce each row group's column chunks into one filesystem read.
            self._files[key] = pq.ParquetFile(
                version.resolve_file_path(path), filesystem=version.filesystem, pre_buffer=True
            )
        return self._files[key]

    def close(self) -> None:
        """Close cached files and release decoded tables, keeping the cache reusable."""
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._tables.clear()
