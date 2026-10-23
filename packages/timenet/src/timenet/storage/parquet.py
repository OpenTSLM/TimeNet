"""Bounded caches for Signal values row groups and shard file handles."""

from typing import TYPE_CHECKING, Self

from cachetools import LRUCache
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


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
                    f"missing Parquet column 'values' in {version.path(path)}, row group {row_group}"
                )
            values = table.column("values").combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise TimeFFormatError(
                f"cannot read Parquet file {version.path(path)}, row group {row_group}: {exc}"
            ) from exc
        self._groups[key] = values
        return values

    def _open_file(self, version: "DatasetVersion", path: str) -> pq.ParquetFile:
        key = (version.root, path)
        handle = self._files.get(key)
        if handle is None:
            handle = pq.ParquetFile(version.path(path), filesystem=version.filesystem, pre_buffer=True)
            self._files[key] = handle
        return handle

    def close(self) -> None:
        """Close shard files and release decoded groups, keeping the cache reusable."""
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._groups.clear()
