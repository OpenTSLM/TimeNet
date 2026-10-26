"""Bounded caches for packed payload tables and Signal values row groups."""

from collections import OrderedDict
from typing import TYPE_CHECKING, Self

from cachetools import LRUCache
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


def _describe(schema: pa.Schema) -> str:
    """Return a schema's fields on one line, for an error message."""
    return ", ".join(f"{field.name}: {field.type}{'' if field.nullable else ' not null'}" for field in schema)


class PayloadTableCache:
    """Cache complete packed payload files up to a byte budget, without retaining file handles.

    The least recently used files leave first. The newest file stays even when it alone exceeds
    the budget, so a large file is still read once.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._tables: OrderedDict[tuple[str, str], pa.Table] = OrderedDict()
        self._nbytes = 0

    def __reduce__(self) -> tuple[type[Self], tuple[int]]:
        # Recreate an empty cache in the receiving process.
        return type(self), (self._max_bytes,)

    @property
    def nbytes(self) -> int:
        """Return the decoded bytes of the payload files the cache holds."""
        return self._nbytes

    def read_table(self, version: "DatasetVersion", path: str, schema: pa.Schema) -> pa.Table:
        """Read a payload file once while it remains in the cache.

        The file's schema is checked once, when the file enters the cache.

        Args:
            version: The release that holds the payload file.
            path: The payload file, relative to the release root.
            schema: The exact Arrow schema the payload file must have, including nullability.

        Returns:
            The cached payload table.

        Raises:
            TimeFFormatError: If the payload file cannot be read or has a different schema.
        """
        key = (version.root, path)
        cached = self._tables.get(key)
        if cached is not None:
            self._tables.move_to_end(key)
            return cached
        try:
            with pq.ParquetFile(
                version.resolve_file_path(path), filesystem=version.filesystem, pre_buffer=True
            ) as source:
                found = source.schema_arrow
                if not found.equals(schema):
                    raise TimeFFormatError(
                        f"Parquet file {version.resolve_file_path(path)} has schema ({_describe(found)}), "
                        f"expected ({_describe(schema)})"
                    )
                table = source.read(use_threads=False)
        except (OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(f"cannot read Parquet file {version.resolve_file_path(path)}: {exc}") from exc
        self._tables[key] = table
        self._nbytes += table.nbytes
        while len(self._tables) > 1 and self._nbytes > self._max_bytes:
            _, evicted = self._tables.popitem(last=False)
            self._nbytes -= evicted.nbytes
        return table

    def close(self) -> None:
        """Release decoded payloads, keeping the cache reusable."""
        self._tables.clear()
        self._nbytes = 0


class _OpenParquetFiles(LRUCache[tuple[str, str], pq.ParquetFile]):
    """Close the least recently used file when the handle limit is reached."""

    def popitem(self) -> tuple[tuple[str, str], pq.ParquetFile]:
        key, handle = super().popitem()
        handle.close()
        return key, handle


class SignalRowGroupCache:
    """Cache values arrays and a bounded number of open Signal shard files."""

    def __init__(self, max_entries: int) -> None:
        self._groups: LRUCache[tuple[str, str, int], pa.Array] = LRUCache(max_entries)
        self._files = _OpenParquetFiles(max_entries)

    def __reduce__(self) -> tuple[type[Self], tuple[int]]:
        # Recreate an empty cache in the receiving process.
        return type(self), (int(self._groups.maxsize),)

    def read_values(self, version: "DatasetVersion", path: str, row_group: int) -> pa.Array:
        """Read and resolve the values column once per cached row group.

        Returns:
            The cached values array.

        Raises:
            TimeFFormatError: If the file, row group, or values column cannot be read.
        """
        key = (version.root, path, row_group)
        cached = self._groups.get(key)
        if cached is not None:
            return cached
        try:
            table = self._open_file(version, path).read_row_group(row_group, columns=["values"])
            # Arrow silently omits a requested column if the file does not contain it.
            if "values" not in table.column_names:
                raise TimeFFormatError(
                    f"missing Parquet column 'values' in {version.resolve_file_path(path)}, row group {row_group}"
                )
            values = table.column("values").combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(
                f"cannot read Parquet file {version.resolve_file_path(path)}, row group {row_group}: {exc}"
            ) from exc
        self._groups[key] = values
        return values

    def _open_file(self, version: "DatasetVersion", path: str) -> pq.ParquetFile:
        key = (version.root, path)
        handle = self._files.get(key)
        if handle is None:
            handle = pq.ParquetFile(version.resolve_file_path(path), filesystem=version.filesystem, pre_buffer=True)
            self._files[key] = handle
        return handle

    def close(self) -> None:
        """Close shard files and release decoded groups, keeping the cache reusable."""
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._groups.clear()
