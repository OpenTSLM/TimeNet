"""The default reader-side values backend: reads float32 values from Parquet shards.

Each shared row group is decoded at most once, since chunks of different series can land in the same
row group.
"""

from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.reader.values import BaseValuesReader


_ROW_GROUP_CACHE_SIZE = 16  # decoded row-group value columns kept, so a shared row group decodes once


class ParquetValuesReader(BaseValuesReader):
    """Reads values from Parquet shards, decoding each shared row group at most once."""

    def __init__(self) -> None:
        """Start with empty (per-process) shard and row-group caches."""
        self._shard_cache: dict[Path, pq.ParquetFile] = {}
        self._row_group_cache: OrderedDict[tuple[Path, str, int], pa.ChunkedArray] = OrderedDict()

    def load(self, root: Path, rows: list[dict]) -> pa.Array:
        """Read a series' chunks from their Parquet row-group locations.

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' 1-D float32 values.
        """
        chunks = [
            self._row_group_values(root, row["shard_path"], row["row_group"])[row["row_offset"]].values for row in rows
        ]
        return pa.concat_arrays([chunk.cast(pa.float32()) for chunk in chunks])

    def close(self) -> None:
        """Close every cached shard file handle and drop cached row groups."""
        for handle in self._shard_cache.values():
            handle.close()
        self._shard_cache.clear()
        self._row_group_cache.clear()

    def _row_group_values(self, root: Path, rel_path: str, row_group: int) -> pa.ChunkedArray:
        """Return a shard row group's ``values`` column, decoding each row group at most once.

        Chunks of different series can share a row group; without this cache every per-series read
        would re-decode the whole column, making value materialization quadratic in chunks per group.

        Args:
            root: The version directory.
            rel_path: The shard's path relative to the version directory.
            row_group: The row-group index within the shard.

        Returns:
            The decoded ``values`` column of the row group.
        """
        key = (root.resolve(), rel_path, row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached
        values = self._shard(root, rel_path).read_row_group(row_group, columns=["values"]).column("values")
        self._row_group_cache[key] = values
        while len(self._row_group_cache) > _ROW_GROUP_CACHE_SIZE:
            self._row_group_cache.popitem(last=False)
        return values

    def _shard(self, root: Path, rel_path: str) -> pq.ParquetFile:
        path = root / rel_path
        if path not in self._shard_cache:
            self._shard_cache[path] = pq.ParquetFile(path)
        return self._shard_cache[path]
