"""Zarr values backend: reader side.

This module resolves index rows to values through per-``spec_type`` Zarr arrays. It is the inverse of
:class:`timenet.values_backends.zarr.writer.ZarrValuesBackend`. Each index row locates a run of values by
``(chunk_file = array path, chunk_major_idx = element start)``. A series usually has one row, so a read is
one contiguous range. For a series with multiple rows, the reader merges the contiguous rows into as few
ranges as possible. The reader serves ranges from an LRU cache of decoded storage chunks. Neighboring
series share storage chunks. Without the cache, each read decodes its full chunks again. This is the same
reason the Parquet reader caches decoded row groups. The reader caches opened arrays for its own lifetime.

This module needs the ``zarr`` extra (``pip install 'timenet[zarr]'``). The module imports it lazily.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.errors import TimeFFormatError
from timenet.values_backends.reader import BaseValuesReader


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec


_CHUNK_CACHE_MAX_BYTES = 64 * 2**20
#: Mirrors the group names in the writer module. See timenet.values_backends.zarr.writer.
_IRREGULAR_GROUP = "_irregular"
_TIME_OFFSETS_GROUP = "_time_offsets"
#: Group holding the per-timestep validity of nullable series, mirroring the writer's layout.
_VALIDITY_GROUP = "_validity"


def _time_offsets_path(values_rel_path: str) -> str:
    """Return the time offsets array path that is parallel to a values array path.

    This function replaces the ``_irregular`` path segment only, not any matching substring. A spec
    type name can contain that text, for example ``foo_irregular_bar``. If the code uses a plain
    string replace, it can match the wrong array instead of failing.

    Args:
        values_rel_path: The values array's path relative to the version directory.

    Returns:
        The parallel time offsets array's path.

    Raises:
        TimeFFormatError: If the path does not sit under the irregular group.
    """
    parts = values_rel_path.split("/")
    if len(parts) < 3 or parts[1] != _IRREGULAR_GROUP:  # noqa: PLR2004 - store dir, group, array name
        raise TimeFFormatError(
            f"expected an irregular series' values under {_IRREGULAR_GROUP!r}, got {values_rel_path!r}; "
            f"the row is tagged irregular but was written without a parallel time offsets array"
        )
    return "/".join([parts[0], _TIME_OFFSETS_GROUP, *parts[2:]])


def _validity_path(values_rel_path: str) -> str:
    """Return the validity array path that is parallel to a values array path.

    The path keeps the values array's own group, so an irregular series' validity sits under
    ``_validity/_irregular/`` and cannot collide with the regular one of the same modality.

    Args:
        values_rel_path: The values array's path relative to the version directory.

    Returns:
        The parallel validity array's path.
    """
    parts = values_rel_path.split("/")
    return "/".join([parts[0], _VALIDITY_GROUP, *parts[1:]])


class ZarrValuesReader(BaseValuesReader):
    """Reads values from per-``spec_type`` Zarr arrays through an LRU cache of decoded chunks."""

    def __init__(self) -> None:
        """Start with empty array and chunk caches for this process."""
        self._array_cache: dict[str, Any] = {}
        self._chunk_cache: OrderedDict[tuple[str, int], Shaped[np.ndarray, " chunk *value"]] = OrderedDict()
        self._chunk_cache_bytes = 0

    def load(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read a series' values. ``chunk_major_idx`` is the element start, and ``n_values`` is the length.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' canonical scalar or fixed-shape tensor Arrow array.
        """
        parts = [self._read_range(version, rel, start, stop) for rel, start, stop in _coalesce_runs(rows)]
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return _to_arrow(combined, spec, self._read_validity(version, rows, spec))

    def _read_validity(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> np.ndarray | None:
        """Read the validity mask covering a series' index rows, or ``None`` when not nullable.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.
            spec: The series' spec. Only a nullable one has a validity array.

        Returns:
            The per-timestep validity (``True`` = present), or ``None``.
        """
        if not spec.nullable:
            return None
        parts = [
            self._read_range(version, _validity_path(rel), start, stop) for rel, start, stop in _coalesce_runs(rows)
        ]
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return combined.astype(bool, copy=False)

    def load_range(
        self, version: DatasetVersion, rows: list[dict], start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
        """Read only the storage chunks that intersect a temporal step range.

        Returns:
            The requested steps in their canonical Arrow representation.
        """
        total = sum(row["n_values"] for row in rows)
        bounded_stop = min(stop, total)
        if start >= bounded_stop:
            if spec.dtype == "enum":
                return pa.DictionaryArray.from_arrays(
                    pa.array([], type=pa.int32()), pa.array(spec.categories, type=pa.string())
                )
            empty = np.empty((0, *spec.value_shape), dtype=spec.dtype)
            return _to_arrow(empty, spec)
        runs = _coalesce_runs(rows)
        parts = []
        validity_parts = []
        cursor = 0
        for rel, run_start, run_stop in runs:
            run_len = run_stop - run_start
            if cursor + run_len > start and cursor < bounded_stop:
                lo = max(start - cursor, 0)
                hi = min(bounded_stop - cursor, run_len)
                parts.append(self._read_range(version, rel, run_start + lo, run_start + hi))
                if spec.nullable:
                    validity_parts.append(
                        self._read_range(version, _validity_path(rel), run_start + lo, run_start + hi)
                    )
            cursor += run_len
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        validity = None
        if validity_parts:
            validity = validity_parts[0] if len(validity_parts) == 1 else np.concatenate(validity_parts)
            validity = validity.astype(bool, copy=False)
        return _to_arrow(combined, spec, validity)

    def load_time_offsets(self, version: DatasetVersion, rows: list[dict]) -> pa.Array:
        """Read an irregular series' time offsets from the array parallel to its values.

        Args:
            version: The opened version handle.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            One int64 microsecond time offset for each value. :func:`_time_offsets_path` raises an error
            if a row's values do not sit under the irregular group. This means the writer tagged
            the row irregular but did not write a parallel time offsets array.
        """
        parts = [
            self._read_range(version, _time_offsets_path(rel), start, stop) for rel, start, stop in _coalesce_runs(rows)
        ]
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return pa.array(combined.astype(np.int64, copy=False))

    def close(self) -> None:
        """Drop cached arrays and decoded chunks. Zarr arrays hold no OS file handles to close."""
        self._array_cache.clear()
        self._chunk_cache.clear()
        self._chunk_cache_bytes = 0

    def _read_range(
        self, version: DatasetVersion, rel_path: str, start: int, stop: int
    ) -> Shaped[np.ndarray, " time *value"]:
        """Assemble ``array[start:stop]`` from cached decoded storage chunks.

        Args:
            version: The opened version handle.
            rel_path: The array's path relative to the version root.
            start: First element of the range.
            stop: One past the last element of the range.

        Returns:
            The range's values with their per-step dimensions preserved.
        """
        array = self._array(version, rel_path)
        chunk_len = array.chunks[0]
        segments = []
        for chunk_idx in range(start // chunk_len, (stop - 1) // chunk_len + 1):
            chunk = self._chunk(rel_path, array, chunk_idx, chunk_len)
            lo = max(start - chunk_idx * chunk_len, 0)
            hi = min(stop - chunk_idx * chunk_len, len(chunk))
            segments.append(chunk[lo:hi])
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def _chunk(self, rel_path: str, array: Any, chunk_idx: int, chunk_len: int) -> Shaped[np.ndarray, " chunk *value"]:
        """Return one decoded storage chunk. The LRU cache decodes each chunk at most once.

        Args:
            rel_path: The array's path (the cache key namespace).
            array: The opened Zarr array.
            chunk_idx: The storage chunk index within the array.
            chunk_len: The array's chunk length in elements.

        Returns:
            The chunk's values with their per-step dimensions preserved.
        """
        key = (rel_path, chunk_idx)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self._chunk_cache.move_to_end(key)
            return cached
        lo = chunk_idx * chunk_len
        data = np.asarray(array[lo : min(lo + chunk_len, array.shape[0])])
        if data.nbytes > _CHUNK_CACHE_MAX_BYTES:
            return data
        self._chunk_cache[key] = data
        self._chunk_cache_bytes += data.nbytes
        while self._chunk_cache_bytes > _CHUNK_CACHE_MAX_BYTES:
            _, evicted = self._chunk_cache.popitem(last=False)
            self._chunk_cache_bytes -= evicted.nbytes
        return data

    def _array(self, version: DatasetVersion, rel_path: str) -> Any:
        """Open the Zarr array for ``rel_path`` through the version's store. Cache the array for later use.

        Args:
            version: The opened version handle.
            rel_path: The array's path relative to the version root.

        Returns:
            The opened read-only Zarr array.

        Raises:
            ImportError: If the ``zarr`` extra is not installed.
        """
        if rel_path not in self._array_cache:
            try:
                import zarr  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "this dataset version stores values in Zarr; install the extra: pip install 'timenet[zarr]'"
                ) from exc

            self._array_cache[rel_path] = zarr.open_array(store=version.store_uri(rel_path), mode="r")
        return self._array_cache[rel_path]


def _coalesce_runs(rows: list[dict]) -> list[tuple[str, int, int]]:
    """Merge adjacent index rows into contiguous ``(array path, start, stop)`` ranges.

    Consecutive placements of one series are contiguous by construction. Because of this, the merge
    usually produces one range for each series.

    Args:
        rows: The series' index rows, sorted by ``chunk_idx``.

    Returns:
        The minimal list of ranges covering the rows, in order.
    """
    runs: list[tuple[str, int, int]] = []
    for row in rows:
        rel, start, stop = row["chunk_file"], row["chunk_major_idx"], row["chunk_major_idx"] + row["n_values"]
        if runs and runs[-1][0] == rel and runs[-1][2] == start:
            runs[-1] = (rel, runs[-1][1], stop)
        else:
            runs.append((rel, start, stop))
    return runs


def _to_arrow(
    values: Shaped[np.ndarray, " time *value"], spec: TimeSeriesSpec, validity: np.ndarray | None = None
) -> pa.Array:
    """Wrap a backend NumPy buffer in the spec's canonical Arrow representation.

    Args:
        values: The dense values read from the store.
        spec: The series' spec.
        validity: Per-timestep validity (``True`` = present), or ``None`` for a non-nullable spec.
            Zarr stores values densely, so this is what turns a stored placeholder back into an
            Arrow null.

    Returns:
        A primitive array for scalar values or a fixed-shape tensor array for N-D values, carrying a
        null at every timestep the validity mask marks absent.
    """
    absent = None if validity is None else ~validity
    if spec.dtype == "str":
        return pa.array(values.tolist(), type=pa.string(), mask=absent)
    if spec.dtype == "enum":
        indices = pa.array(values.ravel().astype(np.int32, copy=False), mask=absent)
        dictionary = pa.array(spec.categories, type=pa.string())
        return pa.DictionaryArray.from_arrays(indices, dictionary)
    contiguous = np.ascontiguousarray(values)
    value_type = pa.from_numpy_dtype(np.dtype(spec.dtype))
    if spec.value_shape:
        dim_names = spec.dimension_names or None
        tensor_type = pa.fixed_shape_tensor(value_type, spec.value_shape, dim_names=dim_names)
        if len(contiguous) == 0:
            storage = pa.FixedSizeListArray.from_arrays(pa.array([], type=value_type), int(np.prod(spec.value_shape)))
            return pa.FixedShapeTensorArray.from_storage(tensor_type, storage)
        # A list-level bitmap needs a pyarrow boolean array, not a NumPy one.
        mask = None if absent is None else pa.array(absent, type=pa.bool_())
        storage = pa.FixedSizeListArray.from_arrays(
            pa.array(contiguous.reshape(-1), type=value_type), int(np.prod(spec.value_shape)), mask=mask
        )
        return pa.FixedShapeTensorArray.from_storage(tensor_type, storage)
    return pa.array(contiguous, type=value_type, mask=absent)
