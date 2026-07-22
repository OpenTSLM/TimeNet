"""Reader-side values backends: resolve a series' index rows into its concatenated float32 values.

The inverse of :mod:`timenet.writer.values`. A :class:`ValuesReader` takes the index rows for one series
(sorted by ``chunk_idx``, each carrying the backend's chunk locator) and returns the series' 1-D float32
Arrow array. The :class:`TimeFReader` picks the backend from the manifest's ``values_backend`` tag and
never imports a specific storage library itself.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.values_backends import PARQUET_VALUES_BACKEND


_ROW_GROUP_CACHE_SIZE = 16  # decoded row-group value columns kept, so a shared row group decodes once


class ValuesReader(Protocol):
    """Reads a series' values from the version directory given its index rows."""

    def load(self, root: Path, rows: list[dict]) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``; each holds ``chunk_file``,
                ``chunk_offset0``, and ``chunk_offset1``.

        Returns:
            The series' 1-D float32 values.
        """
        ...

    def close(self) -> None:
        """Release any open handles or caches held for the reader's lifetime."""
        ...


def make_values_reader(name: str) -> ValuesReader:
    """Construct the reader-side values backend named ``name``.

    Args:
        name: The manifest ``values_backend`` tag.

    Returns:
        The constructed reader backend.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    if name == PARQUET_VALUES_BACKEND:
        return ParquetValuesReader()
    raise ValueError(f"unknown values_backend {name!r}; expected {PARQUET_VALUES_BACKEND!r}")


class ParquetValuesReader:
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
            self._row_group_values(root, row["chunk_file"], row["chunk_offset0"])[row["chunk_offset1"]].values
            for row in rows
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
