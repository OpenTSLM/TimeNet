"""The default reader-side values backend: reads typed values from Parquet shards.

The reader decodes each shared row group at most once because chunks of different series can land
in the same row group.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.values_backends.reader import BaseValuesReader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec


_ROW_GROUP_CACHE_SIZE = 64
"""Decoded row groups kept, so one shared by many series decodes once.

A shard writes a signal's chunks across several row groups, so a record with seven signals reaches
about 56 groups. A smaller cache drops groups the same record still needs.
"""


@dataclass(frozen=True)
class _DecodedGroup:
    """One decoded shard row group, with its two list columns combined once.

    Combining them here once makes each series' read a single list-scalar lookup.
    """

    values: pa.ListArray
    """The group's ``values`` column, one list per stored chunk."""
    time_offsets: pa.ListArray
    """The group's ``time_offsets_us`` column, null for a chunk of a regular series."""


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
        """Start with empty (per-process) shard, row-group, and target-type caches."""
        self._shard_cache: dict[str, pq.ParquetFile] = {}
        self._row_group_cache: OrderedDict[tuple[str, str, int], _DecodedGroup] = OrderedDict()
        self._target_types: dict[str, pa.DataType] = {}

    def load(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read a series' chunks (``chunk_major_idx`` = row group, ``chunk_minor_idx`` = row offset).

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
        chunks = [
            self._row_group(version, row["chunk_file"], row["chunk_major_idx"]).values[row["chunk_minor_idx"]].values
            for row in rows
        ]
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

    def load_time_offsets(self, version: DatasetVersion, rows: list[dict]) -> pa.Array:
        """Read an irregular series' time offsets, which share their values' chunk locators.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFFormatError: If a chunk stores no time offsets. The index tags the row irregular,
                but the writer omitted them.
        """
        chunks = []
        for row in rows:
            group = self._row_group(version, row["chunk_file"], row["chunk_major_idx"])
            cell = group.time_offsets[row["chunk_minor_idx"]]
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
            The requested scalar values in the spec's canonical Arrow type.
        """
        target = self._target(spec)
        total = sum(row["n_values"] for row in rows)
        bounded_stop = min(stop, total)
        if start >= bounded_stop:
            return pa.array([], type=target)
        parts = []
        cursor = 0
        for row in rows:
            row_stop = cursor + row["n_values"]
            if row_stop > start and cursor < bounded_stop:
                group = self._row_group(version, row["chunk_file"], row["chunk_major_idx"])
                values = group.values[row["chunk_minor_idx"]].values
                lo = max(start - cursor, 0)
                hi = min(bounded_stop - cursor, row["n_values"])
                parts.append(values.slice(lo, hi - lo).cast(target))
            cursor = row_stop
        return pa.concat_arrays(parts)

    def close(self) -> None:
        """Close every cached shard file handle and drop cached row groups."""
        for handle in self._shard_cache.values():
            handle.close()
        self._shard_cache.clear()
        self._row_group_cache.clear()

    def _row_group(self, version: DatasetVersion, rel_path: str, row_group: int) -> _DecodedGroup:
        """Return a shard row group's values and time offsets together, decoding it at most once.

        Both columns come back in one read because pyarrow decodes a row group's columns in a single
        pass. The extra time offsets cost almost nothing. Two separate reads cost about 26% more when a
        caller wants both, which is nearly always the case for an irregular series.

        Chunks of different series can share a row group. Without this cache, every per-series read
        will re-decode the whole column. That makes value materialization quadratic in chunks per group.

        Args:
            version: The opened version handle.
            rel_path: The shard's path relative to the version root.
            row_group: The row-group index within the shard.

        Returns:
            The decoded row group, with both list columns combined once.
        """
        key = (version.root, rel_path, row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached
        shard = self._shard(version, rel_path)
        wanted = ["values"] if self._offsets_all_null(shard, row_group) else ["values", "time_offsets_us"]
        table = shard.read_row_group(row_group, columns=wanted)
        values = table.column("values").combine_chunks()
        group = _DecodedGroup(
            values=values,
            time_offsets=(
                table.column("time_offsets_us").combine_chunks()
                if "time_offsets_us" in wanted
                else pa.nulls(len(values), type=pa.list_(pa.int64()))
            ),
        )
        self._row_group_cache[key] = group
        while len(self._row_group_cache) > _ROW_GROUP_CACHE_SIZE:
            self._row_group_cache.popitem(last=False)
        return group

    @staticmethod
    def _offsets_all_null(shard: pq.ParquetFile, row_group: int) -> bool:
        """Return whether a row group stores no time offset at all.

        A regular series stores no time offsets, so on a regular-axis dataset the column is null
        in every row. The Parquet footer already holds the count, so this reads no data.

        The footer counts leaf values, so an empty list counts as absent too. TimeF allows neither
        an empty list nor a null offset, and :meth:`load_time_offsets` rejects such a shard.

        Args:
            shard: The open shard.
            row_group: The row group to check.

        Returns:
            True when the column holds no leaf value, so the read can skip it.
        """
        try:
            group = shard.metadata.row_group(row_group)
        except AttributeError:
            # A shard with no footer cannot say. Read both columns.
            return False
        # Column chunks are indexed by leaf, which is not the position of the Arrow field.
        for index in range(group.num_columns):
            column = group.column(index)
            if column.path_in_schema.split(".", 1)[0] != "time_offsets_us":
                continue
            statistics = column.statistics
            return statistics is not None and statistics.null_count == column.num_values
        return False

    def _shard(self, version: DatasetVersion, rel_path: str) -> pq.ParquetFile:
        path = version.path(rel_path)
        if path not in self._shard_cache:
            # pre_buffer coalesces a row group's column chunks into one read, which is one request
            # instead of one per column on a remote filesystem.
            self._shard_cache[path] = pq.ParquetFile(path, filesystem=version.filesystem, pre_buffer=True)
        return self._shard_cache[path]
