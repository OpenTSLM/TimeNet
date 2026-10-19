"""The default reader-side values backend: reads typed values from Parquet shards.

The reader decodes each shared row group at most once because chunks of different series can land
in the same row group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from timenet.storage.parquet import ParquetReadCache
from timenet.values_backends.reader import BaseValuesReader


if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec
    from timenet.values_backends.types import StoredChunk


_ROW_GROUP_CACHE_SIZE = 64
"""Decoded row groups kept, so one shared by many series decodes once.

A shard writes a signal's chunks across several row groups, so a record with seven signals reaches
about 56 groups. A smaller cache drops groups the same record still needs.
"""


def _target_type(spec: TimeSeriesSpec) -> pa.DataType:
    """Return the Arrow type a spec's values should come back as.

    A ``"str"`` reads back as plain text. An ``"enum"`` reads back as a dictionary array (the shard
    already stores one). Every other dtype maps through NumPy so the in-memory and stored types
    match.

    Args:
        spec: The series' spec.

    Returns:
        The canonical Arrow type for the spec's dtype.
    """
    if spec.dtype == "str":
        return pa.string()
    if spec.dtype == "enum":
        from timenet.values_backends.parquet.writer import _ENUM_LEAF  # noqa: PLC0415

        return _ENUM_LEAF
    return pa.from_numpy_dtype(np.dtype(spec.dtype))


class ParquetValuesReader(BaseValuesReader):
    """Reads values from Parquet shards, decoding each shared row group at most once."""

    def __init__(self) -> None:
        """Start with empty per-process shard, row-group, and target-type caches."""
        self._parquet = ParquetReadCache(_ROW_GROUP_CACHE_SIZE)
        self._target_types: dict[str, pa.DataType] = {}

    def prepare_scan(
        self, version: DatasetVersion, groups: list[tuple[str, int]], executor: ThreadPoolExecutor
    ) -> None:
        """Prefetch required Signal groups through the shared Parquet cache."""
        self._parquet.prepare_scan(version, groups, executor, ("values",))

    def load(self, version: DatasetVersion, rows: list[StoredChunk], spec: TimeSeriesSpec) -> pa.Array:
        """Read a series' chunks using row-group and row coordinates in ``data_index``.

        This method copies each chunk into a new array. A caller that keeps the values therefore
        does not keep the whole decoded row group alive.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.
            spec: The series' spec, which names the Arrow type to return.

        Returns:
            The series' 1-D values in the spec's canonical Arrow type.
        """
        # If the stored type matches the spec, skip conversion.
        target = self._target(spec)
        chunks = []
        for row in rows:
            values = self._parquet.read_column(version, row.chunk_file, row.data_index.major_idx, "values")
            chunks.append(values[row.data_index.minor_idx].values)
        return pa.concat_arrays([chunk if chunk.type == target else chunk.cast(target) for chunk in chunks])

    def _target(self, spec: TimeSeriesSpec) -> pa.DataType:
        """Return the spec's canonical Arrow type, resolved once per dtype.

        Args:
            spec: The series' spec.

        Returns:
            The canonical Arrow type for the spec's dtype.
        """
        target = self._target_types.get(spec.dtype)
        if target is None:
            target = _target_type(spec)
            self._target_types[spec.dtype] = target
        return target

    def load_range(
        self, version: DatasetVersion, rows: list[StoredChunk], start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
        """Read a scalar temporal subsection, trimming chunks at the requested boundaries.

        Returns:
            The requested scalar values in the spec's canonical Arrow type.
        """
        target = self._target(spec)
        total = sum(row.n_values for row in rows)
        bounded_stop = min(stop, total)
        if start >= bounded_stop:
            return pa.array([], type=target)
        parts = []
        cursor = 0
        for row in rows:
            row_stop = cursor + row.n_values
            if row_stop > start and cursor < bounded_stop:
                column = self._parquet.read_column(version, row.chunk_file, row.data_index.major_idx, "values")
                values = column[row.data_index.minor_idx].values
                lo = max(start - cursor, 0)
                hi = min(bounded_stop - cursor, row.n_values)
                parts.append(values.slice(lo, hi - lo).cast(target))
            cursor = row_stop
        return pa.concat_arrays(parts)

    def finish_scan(self) -> None:
        """Release scan workers and pending groups after an eager read, including errors."""
        self._parquet.finish_scan()

    def close(self) -> None:
        """Close every cached shard file handle and drop cached row groups."""
        self._parquet.close()
