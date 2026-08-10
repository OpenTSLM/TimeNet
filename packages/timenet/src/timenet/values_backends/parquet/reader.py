"""The default reader-side values backend: reads float32 values from Parquet shards.

Each shared row group is decoded at most once, since chunks of different series can land in the same
row group.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.values_backends.reader import BaseValuesReader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec


_ROW_GROUP_CACHE_SIZE = 16  # decoded row groups kept, so one shared by many series decodes once


class ParquetValuesReader(BaseValuesReader):
    """Reads values from Parquet shards, decoding each shared row group at most once."""

    def __init__(self) -> None:
        """Start with empty (per-process) shard and row-group caches."""
        self._shard_cache: dict[str, pq.ParquetFile] = {}
        self._row_group_cache: OrderedDict[tuple[str, str, int], pa.Table] = OrderedDict()

    def load(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read a series' chunks (``chunk_major_idx`` = row group, ``chunk_minor_idx`` = row offset).

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' 1-D float32 values.
        """
        del spec  # Parquet is scalar float32-only in this format version.
        chunks = [
            self._row_group_values(version, row["chunk_file"], row["chunk_major_idx"])[row["chunk_minor_idx"]].values
            for row in rows
        ]
        return pa.concat_arrays([chunk.cast(pa.float32()) for chunk in chunks])

    def load_time_offsets(self, version: DatasetVersion, rows: list[dict]) -> pa.Array:
        """Read an irregular series' time offsets, which share their values' chunk locators.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If a chunk stores no time offsets, which means the row was tagged irregular
                but written without them.
        """
        chunks = []
        for row in rows:
            cell = self._row_group_time_offsets(version, row["chunk_file"], row["chunk_major_idx"])[
                row["chunk_minor_idx"]
            ]
            if not cell.is_valid:
                raise TimeFFormatError(
                    f"chunk {row['chunk_idx']} of an irregular series stores no time offsets in "
                    f"{row['chunk_file']!r}; the row is tagged irregular but was written without them"
                )
            chunks.append(cell.values)
        return pa.concat_arrays([chunk.cast(pa.int64()) for chunk in chunks])

    def load_range(
        self, version: DatasetVersion, rows: list[dict], start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
        """Read a scalar temporal subsection, trimming chunks at the requested boundaries.

        Returns:
            The requested scalar float32 values.
        """
        del spec  # Parquet is scalar float32-only in this format version.
        total = sum(row["n_values"] for row in rows)
        bounded_stop = min(stop, total)
        if start >= bounded_stop:
            return pa.array([], type=pa.float32())
        parts = []
        cursor = 0
        for row in rows:
            row_stop = cursor + row["n_values"]
            if row_stop > start and cursor < bounded_stop:
                values = self._row_group_values(version, row["chunk_file"], row["chunk_major_idx"])[
                    row["chunk_minor_idx"]
                ].values
                lo = max(start - cursor, 0)
                hi = min(bounded_stop - cursor, row["n_values"])
                parts.append(values.slice(lo, hi - lo).cast(pa.float32()))
            cursor = row_stop
        return pa.concat_arrays(parts)

    def close(self) -> None:
        """Close every cached shard file handle and drop cached row groups."""
        for handle in self._shard_cache.values():
            handle.close()
        self._shard_cache.clear()
        self._row_group_cache.clear()

    def _row_group_values(self, version: DatasetVersion, rel_path: str, row_group: int) -> pa.ChunkedArray:
        """Return a shard row group's ``values`` column, decoding each row group at most once.

        Chunks of different series can share a row group; without this cache every per-series read
        would re-decode the whole column, making value materialization quadratic in chunks per group.

        Args:
            version: The opened version handle.
            rel_path: The shard's path relative to the version directory.
            row_group: The row-group index within the shard.

        Returns:
            The decoded ``values`` column of the row group.
        """
        return self._row_group(version, rel_path, row_group).column("values")

    def _row_group_time_offsets(self, version: DatasetVersion, rel_path: str, row_group: int) -> pa.ChunkedArray:
        """Return a shard row group's ``time_offsets_us`` column, from the same cached decode.

        Args:
            version: The opened version handle.
            rel_path: The shard's path relative to the version directory.
            row_group: The row-group index within the shard.

        Returns:
            The decoded ``time_offsets_us`` column of the row group.
        """
        return self._row_group(version, rel_path, row_group).column("time_offsets_us")

    def _row_group(self, version: DatasetVersion, rel_path: str, row_group: int) -> pa.Table:
        """Return a shard row group's values and time offsets together, decoding it at most once.

        Both columns come back in one read because pyarrow decodes a row group's columns in a single
        pass, so adding the time offsets costs almost nothing: measured -0.4% on an all-regular shard,
        where the column is null, and 9.2% in the worst case where every series is irregular. Two
        separate reads cost 26% more whenever a caller wants both, which for an irregular series is
        nearly always, since its time offsets are what make its values interpretable.

        Args:
            version: The opened version handle.
            rel_path: The shard's path relative to the version directory.
            row_group: The row-group index within the shard.

        Returns:
            The decoded row group.
        """
        key = (version.root, rel_path, row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached
        table = self._shard(version, rel_path).read_row_group(row_group, columns=["values", "time_offsets_us"])
        self._row_group_cache[key] = table
        while len(self._row_group_cache) > _ROW_GROUP_CACHE_SIZE:
            self._row_group_cache.popitem(last=False)
        return table

    def _shard(self, version: DatasetVersion, rel_path: str) -> pq.ParquetFile:
        path = version.path(rel_path)
        if path not in self._shard_cache:
            self._shard_cache[path] = pq.ParquetFile(path, filesystem=version.filesystem)
        return self._shard_cache[path]
