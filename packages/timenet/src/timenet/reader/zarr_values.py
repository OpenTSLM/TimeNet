"""Zarr values backend (reader side): resolve index rows to values via per-``spec_type`` Zarr arrays.

The inverse of :class:`timenet.writer.zarr_values.ZarrValuesBackend`. Each index row locates a run of
values by ``(chunk_file = array path, chunk_offset0 = element start)``. A series is normally one row, so
a read is one contiguous range; multi-row series coalesce contiguous rows into as few ranges as
possible. Ranges are served from an LRU of decoded storage chunks — neighboring series share storage
chunks, and without the cache every read would re-decode its full chunks (the same reason the Parquet
reader caches decoded row groups). Opened arrays are cached for the reader's lifetime.

Requires the ``zarr`` extra (``pip install 'timenet[zarr]'``); imported lazily.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


_CHUNK_CACHE_SIZE = 64  # decoded storage chunks kept (~64 MiB at the default 1 MiB chunk size)


class ZarrValuesReader:
    """Reads values from per-``spec_type`` Zarr arrays through a decoded-chunk LRU."""

    def __init__(self) -> None:
        """Start with empty (per-process) array and chunk caches."""
        self._array_cache: dict[str, Any] = {}
        self._chunk_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def load(self, root: Path, rows: list[dict]) -> pa.Array:
        """Read a series' values (``chunk_offset0`` = element start; length is ``n_values``).

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' 1-D float32 values.
        """
        parts = [self._read_range(root, rel, start, stop) for rel, start, stop in _coalesce_runs(rows)]
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return pa.array(combined, type=pa.float32())

    def close(self) -> None:
        """Drop cached arrays and decoded chunks (Zarr arrays hold no OS file handles to close)."""
        self._array_cache.clear()
        self._chunk_cache.clear()

    def _read_range(self, root: Path, rel_path: str, start: int, stop: int) -> np.ndarray:
        """Assemble ``array[start:stop]`` from cached decoded storage chunks.

        Args:
            root: The version directory.
            rel_path: The array's path relative to the version directory.
            start: First element of the range.
            stop: One past the last element of the range.

        Returns:
            The range's float32 values.
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

    def _chunk(self, rel_path: str, array: Any, chunk_idx: int, chunk_len: int) -> np.ndarray:
        """Return one decoded storage chunk, decoding each chunk at most once (LRU).

        Args:
            rel_path: The array's path (the cache key namespace).
            array: The opened Zarr array.
            chunk_idx: The storage chunk index within the array.
            chunk_len: The array's chunk length in elements.

        Returns:
            The chunk's float32 values.
        """
        key = (rel_path, chunk_idx)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self._chunk_cache.move_to_end(key)
            return cached
        lo = chunk_idx * chunk_len
        data = np.asarray(array[lo : min(lo + chunk_len, array.shape[0])], dtype=np.float32)
        self._chunk_cache[key] = data
        while len(self._chunk_cache) > _CHUNK_CACHE_SIZE:
            self._chunk_cache.popitem(last=False)
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
                import zarr
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
        rel, start, stop = row["chunk_file"], row["chunk_offset0"], row["chunk_offset0"] + row["n_values"]
        if runs and runs[-1][0] == rel and runs[-1][2] == start:
            runs[-1] = (rel, runs[-1][1], stop)
        else:
            runs.append((rel, start, stop))
    return runs
