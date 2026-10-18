"""Cached Parquet reads shared by Signal chunks and packed field values."""

from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_SCAN_WORKERS = 4


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
        self._scan_executor: ThreadPoolExecutor | None = None
        self._scan_pending: dict[tuple[str, str, int | None, tuple[str, ...] | None], Future[_DecodedTable]] = {}
        self._scan_groups: Iterator[tuple[DatasetVersion, str, int, tuple[str, ...]]] = iter(())

    def __getstate__(self) -> dict:
        return {
            "_max_entries": self._max_entries,
            "_tables": OrderedDict(),
            "_files": {},
            "_scan_executor": None,
            "_scan_pending": {},
            "_scan_groups": iter(()),
        }

    def prepare_scan(
        self,
        version: "DatasetVersion",
        groups: list[tuple[str, int]],
        executor: ThreadPoolExecutor,
        columns: tuple[str, ...],
    ) -> None:
        """Prefetch a bounded window of required projected groups in the supplied order."""
        if not groups or self._scan_executor is not None:
            return
        self._scan_executor = executor
        self._scan_groups = ((version, path, group, columns) for path, group in groups)
        self._fill_scan()

    def _fill_scan(self) -> None:
        """Keep at most four required groups decoding ahead of the consumer."""
        if self._scan_executor is None:
            return
        while len(self._scan_pending) < _SCAN_WORKERS:
            item = next(self._scan_groups, None)
            if item is None:
                break
            version, path, group, columns = item
            key = (version.root, path, group, columns)
            if key not in self._tables and key not in self._scan_pending:
                file = self._open_file(version, path)
                self._scan_pending[key] = self._scan_executor.submit(self._decode, file, group, columns)

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
            pending = self._scan_pending.pop(key, None)
            if pending is not None:
                decoded = pending.result()
            elif row_group is None:
                with version.filesystem.open_input_file(version.resolve_file_path(path)) as source:
                    decoded = self._decode(pq.ParquetFile(source), row_group, columns)
            else:
                decoded = self._decode(self._open_file(version, path), row_group, columns)
            if columns is not None and any(column not in decoded.columns for column in columns):
                raise TimeFFormatError(f"missing Parquet columns {columns} in {version.resolve_file_path(path)}")
        except (OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(
                f"cannot read Parquet file {version.resolve_file_path(path)}, row group {row_group}: {exc}"
            ) from exc
        self._fill_scan()
        self._tables[key] = decoded
        while len(self._tables) > self._max_entries:
            self._tables.popitem(last=False)
        return decoded

    @staticmethod
    def _decode(file: pq.ParquetFile, row_group: int | None, columns: tuple[str, ...] | None) -> _DecodedTable:
        """Decode a selection and resolve its columns once, without nested column workers.

        Returns:
            The decoded table and its column arrays over the same buffers.
        """
        projection = None if columns is None else list(columns)
        table = (
            file.read(columns=projection, use_threads=False)
            if row_group is None
            else file.read_row_group(row_group, columns=projection, use_threads=False)
        ).combine_chunks()
        arrays = {
            name: values.chunk(0) if values.num_chunks else pa.array([], type=values.type)
            for name, values in zip(table.column_names, table.columns, strict=True)
        }
        return _DecodedTable(table, arrays)

    def _open_file(self, version: "DatasetVersion", path: str) -> pq.ParquetFile:
        """Reuse a file handle across projected row-group reads.

        Returns:
            The cached or newly opened Parquet file.
        """
        key = (version.root, path)
        if key not in self._files:
            # Concurrent reads must not replace a shared Arrow range cache.
            self._files[key] = pq.ParquetFile(
                version.resolve_file_path(path), filesystem=version.filesystem, pre_buffer=False
            )
        return self._files[key]

    def finish_scan(self) -> None:
        """Release scan workers and pending groups after successful or failed reads."""
        if self._scan_executor is not None:
            self._scan_executor.shutdown(wait=True, cancel_futures=True)
            self._scan_executor = None
        self._scan_pending.clear()
        self._scan_groups = iter(())

    def close(self) -> None:
        """Close cached files and release decoded tables, keeping the cache reusable."""
        self.finish_scan()
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._tables.clear()
