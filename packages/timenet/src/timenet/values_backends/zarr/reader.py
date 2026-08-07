"""Zarr values backend (reader side): resolve index rows to values via per-``spec_type`` Zarr arrays.

The inverse of :class:`timenet.values_backends.zarr.writer.ZarrValuesBackend`. Each index row locates a run of
values by ``(chunk_file = array path, chunk_major_idx = element start)``. A series is normally one row, so
a read is one contiguous range; multi-row series coalesce contiguous rows into as few ranges as
possible. Ranges are served from an LRU of decoded storage chunks — neighboring series share storage
chunks, and without the cache every read would re-decode its full chunks (the same reason the Parquet
reader caches decoded row groups). Opened arrays are cached for the reader's lifetime.

Requires the ``zarr`` extra (``pip install 'timenet[zarr]'``); imported lazily.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec
from timenet.values_backends.reader import BaseValuesReader


_CHUNK_CACHE_MAX_BYTES = 64 * 2**20


class ZarrValuesReader(BaseValuesReader):
    """Reads values from per-``spec_type`` Zarr arrays through a decoded-chunk LRU."""

    def __init__(self) -> None:
        """Start with empty (per-process) array and chunk caches."""
        self._array_cache: dict[str, Any] = {}
        self._chunk_cache: OrderedDict[tuple[str, int], Shaped[np.ndarray, " chunk *value"]] = OrderedDict()
        self._chunk_cache_bytes = 0

    def load(self, root: Path, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read a series' values (``chunk_major_idx`` = element start; length is ``n_values``).

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' canonical scalar or fixed-shape tensor Arrow array.
        """
        parts = [self._read_range(root, rel, start, stop) for rel, start, stop in _coalesce_runs(rows)]
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return _to_arrow(combined, spec)

    def load_range(self, root: Path, rows: list[dict], start: int, stop: int, spec: TimeSeriesSpec) -> pa.Array:
        """Read only the storage chunks intersecting a temporal step range.

        Returns:
            The requested steps in their canonical Arrow representation.
        """
        total = sum(row["n_values"] for row in rows)
        bounded_stop = min(stop, total)
        if start >= bounded_stop:
            empty = np.empty((0, *spec.value_shape), dtype=spec.dtype)
            return _to_arrow(empty, spec)
        runs = _coalesce_runs(rows)
        parts = []
        cursor = 0
        for rel, run_start, run_stop in runs:
            run_len = run_stop - run_start
            if cursor + run_len > start and cursor < bounded_stop:
                lo = max(start - cursor, 0)
                hi = min(bounded_stop - cursor, run_len)
                parts.append(self._read_range(root, rel, run_start + lo, run_start + hi))
            cursor += run_len
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        return _to_arrow(combined, spec)

    def load_time_offsets(self, root: Path, rows: list[dict]) -> pa.Array:
        """Refuse: this backend cannot store per-value time offsets yet, so it can never have written any.

        Args:
            root: The version directory.
            rows: The series' index rows.

        Raises:
            TimeFFormatError: Always.
        """
        del self, root, rows
        raise TimeFFormatError(
            "the Zarr values backend cannot store per-value time offsets, so this artifact should not "
            "contain an irregular series; it was written by a different or newer writer"
        )

    def close(self) -> None:
        """Drop cached arrays and decoded chunks (Zarr arrays hold no OS file handles to close)."""
        self._array_cache.clear()
        self._chunk_cache.clear()
        self._chunk_cache_bytes = 0

    def _read_range(self, root: Path, rel_path: str, start: int, stop: int) -> Shaped[np.ndarray, " time *value"]:
        """Assemble ``array[start:stop]`` from cached decoded storage chunks.

        Args:
            root: The version directory.
            rel_path: The array's path relative to the version directory.
            start: First element of the range.
            stop: One past the last element of the range.

        Returns:
            The range's values with their per-step dimensions preserved.
        """
        array = self._array(root, rel_path)
        chunk_len = array.chunks[0]
        segments = []
        for chunk_idx in range(start // chunk_len, (stop - 1) // chunk_len + 1):
            chunk = self._chunk(rel_path, array, chunk_idx, chunk_len)
            lo = max(start - chunk_idx * chunk_len, 0)
            hi = min(stop - chunk_idx * chunk_len, len(chunk))
            segments.append(chunk[lo:hi])
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def _chunk(self, rel_path: str, array: Any, chunk_idx: int, chunk_len: int) -> Shaped[np.ndarray, " chunk *value"]:
        """Return one decoded storage chunk, decoding each chunk at most once (LRU).

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

    def _array(self, root: Path, rel_path: str) -> Any:
        """Open (and cache) the Zarr array at ``root/rel_path``.

        Args:
            root: The version directory.
            rel_path: The array's path relative to the version directory.

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

            self._array_cache[rel_path] = zarr.open_array(store=root / rel_path, mode="r")
        return self._array_cache[rel_path]


def _coalesce_runs(rows: list[dict]) -> list[tuple[str, int, int]]:
    """Merge adjacent index rows into contiguous ``(array path, start, stop)`` ranges.

    Consecutive placements of one series are contiguous by construction, so this normally collapses to a
    single range per series.

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


def _to_arrow(values: Shaped[np.ndarray, " time *value"], spec: TimeSeriesSpec) -> pa.Array:
    """Wrap a backend NumPy buffer in the spec's canonical Arrow representation.

    Returns:
        A primitive array for scalar values or a fixed-shape tensor array for N-D values.
    """
    contiguous = np.ascontiguousarray(values)
    value_type = pa.from_numpy_dtype(np.dtype(spec.dtype))
    if spec.value_shape:
        dim_names = spec.dimension_names or None
        if len(contiguous) == 0:
            # pa.FixedShapeTensorArray.from_numpy_ndarray rejects a 0-length ndarray, which an empty
            # range read (e.g. read_steps(n, n)) produces; build the empty tensor array from storage.
            tensor_type = pa.fixed_shape_tensor(value_type, spec.value_shape, dim_names=dim_names)
            storage = pa.FixedSizeListArray.from_arrays(pa.array([], type=value_type), int(np.prod(spec.value_shape)))
            return pa.FixedShapeTensorArray.from_storage(tensor_type, storage)
        return pa.FixedShapeTensorArray.from_numpy_ndarray(contiguous, dim_names=dim_names)
    return pa.array(contiguous, type=value_type)
