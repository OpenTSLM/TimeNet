"""Write signal values into a Zarr values plane, and read them back by locator.

This is the second backend behind the seam in :mod:`timenet.control_plane.values`, and it is
deliberately plainer than the Parquet one: no per-modality encoding choice, no byte-budgeted shard
rotation, no self-check on the finished file. It writes one chunked typed array per modality and
addresses a signal by where it starts in that array.

The layout follows what the earlier revision of this project measured. Series of one modality are
appended along time to one array, so a modality's values sit contiguously and Zarr's own chunking
decides the storage granularity. Arrays are partitioned by ``(spec_type, stores_time_offsets)``
rather than by ``spec_type`` alone: a series with per-value time offsets writes to a values array
under ``_irregular/`` and to a parallel ``_time_offsets/`` array, and both advance together only if
every series in the partition writes to both. Partitioning by modality alone lets the two drift, and
then one element offset no longer addresses both.

A locator is ``(array path, element offset)``. ``chunk_minor_idx`` stays ``None``: a Zarr array has
no second level of addressing, and ``n_values`` already gives the length.

This module needs the ``zarr`` extra (``pip install 'timenet[zarr]'``) and imports it lazily, so the
Parquet core never pays for it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote

import numpy as np

from timenet.control_plane.values import ZARR, ChunkLocator, PendingSignal
from timenet.errors import TimeFValidationError
from timenet.format.constants import DEFAULT_CHUNK_MAX_BYTES, DEFAULT_SHARD_TARGET_BYTES


STORE_DIR = "time_series.zarr"
"""The Zarr group every array of a version lives in, relative to the version directory."""

_IRREGULAR_GROUP = "_irregular"
_TIME_OFFSETS_GROUP = "_time_offsets"
_BYTES_PER_TIME_OFFSET = 8
_BLOSC_CNAMES = frozenset({"zstd", "lz4", "lz4hc", "zlib", "blosclz"})


def array_name(spec_type: str) -> str:
    """Encode a modality tag as one filesystem-safe Zarr path segment.

    Percent-encoding every character outside the URL-unreserved set keeps a path separator inside a
    spec type from splitting the path, and is reversible, so distinct modalities stay distinct.

    Args:
        spec_type: The modality tag.

    Returns:
        The physical array name.
    """
    return quote(spec_type, safe="")


def values_array_path(spec_type: str, stores_time_offsets: bool) -> str:
    """Return the version-relative path of a partition's values array.

    Args:
        spec_type: The modality tag.
        stores_time_offsets: Whether this partition's series carry per-value time offsets.

    Returns:
        The array path, which is what a locator's ``chunk_file`` holds.
    """
    name = array_name(spec_type)
    inner = f"{_IRREGULAR_GROUP}/{name}" if stores_time_offsets else name
    return f"{STORE_DIR}/{inner}"


def time_offsets_array_path(spec_type: str) -> str:
    """Return the version-relative path of the time offsets array parallel to a values array.

    Args:
        spec_type: The modality tag.

    Returns:
        The array path.
    """
    return f"{STORE_DIR}/{_TIME_OFFSETS_GROUP}/{array_name(spec_type)}"


class ZarrValuesPlaneWriter:
    """Appends each signal to its modality's Zarr array and reports where it landed."""

    def __init__(
        self,
        staging_dir: Path,
        *,
        chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
        shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        """Bind the writer to its staging directory and its chunking budget.

        Args:
            staging_dir: The version's staging directory. The store goes under it.
            chunk_max_bytes: How large one Zarr chunk may be, uncompressed.
            shard_target_bytes: How much to buffer before writing. Appends are flushed on a chunk
                boundary, so a chunk object is written once rather than rewritten per signal.
            compression: The Blosc codec name.
            compression_level: The Blosc level, 0-9.

        Raises:
            TimeFValidationError: If ``compression`` is not a Blosc codec.
        """
        if compression not in _BLOSC_CNAMES:
            raise TimeFValidationError(
                f"unknown compression {compression!r} for the zarr values backend; it compresses "
                f"with Blosc, whose codecs are {', '.join(sorted(_BLOSC_CNAMES))}"
            )
        self._staging_dir = Path(staging_dir)
        self._chunk_max_bytes = chunk_max_bytes
        self._buffer_target_bytes = shard_target_bytes
        self._compression = compression
        self._compression_level = compression_level
        self._group: Any = None
        self._partitions: dict[tuple[str, bool], _Partition] = {}

    @property
    def parts(self) -> list[str]:
        """Every file the store holds, as version-relative paths, for the manifest.

        Returns:
            The files, sorted. A Zarr store is a directory of many small objects, so this list is
            far longer than the Parquet backend's.
        """
        store = self._staging_dir / STORE_DIR
        if not store.exists():
            return []
        return sorted(path.relative_to(self._staging_dir).as_posix() for path in store.rglob("*") if path.is_file())

    @property
    def artifacts(self) -> list[tuple[str, str]]:
        """Every array a locator may name, tagged with this backend.

        Returns:
            One ``(array path, 'zarr')`` pair per array written, including the time offsets arrays.
        """
        found: list[tuple[str, str]] = []
        for (spec_type, irregular), partition in self._partitions.items():
            found.append((values_array_path(spec_type, irregular), ZARR))
            if partition.time_offsets is not None:
                found.append((time_offsets_array_path(spec_type), ZARR))
        return found

    @property
    def value_encoding(self) -> dict[str, str]:
        """Empty: this backend makes no per-modality encoding choice to record.

        Returns:
            An empty mapping.
        """
        return {}

    def add(self, signal: PendingSignal) -> list[ChunkLocator]:
        """Append one signal to its partition and return its locator.

        Unlike the Parquet backend, the offset is known as soon as the values are handed over, so
        the locator comes back straight away rather than when a row group flushes.

        Args:
            signal: The signal to write.

        Returns:
            The signal's one locator.
        """
        irregular = signal.time_offsets_us is not None
        key = (signal.spec_type, irregular)
        partition = self._partitions.get(key)
        if partition is None:
            partition = self._open_partition(signal, irregular)
            self._partitions[key] = partition
        start = partition.append(signal.values, signal.time_offsets_us)
        return [
            ChunkLocator(
                signal_id=signal.signal_id,
                chunk_idx=0,
                chunk_file=values_array_path(signal.spec_type, irregular),
                chunk_major_idx=start,
                chunk_minor_idx=None,
                n_values=len(signal.values),
            )
        ]

    def finish(self) -> list[ChunkLocator]:
        """Flush every partition's trailing buffer.

        Returns:
            No locators. Every locator was already returned by :meth:`add`.
        """
        for partition in self._partitions.values():
            partition.finish()
        return []

    def _open_partition(self, signal: PendingSignal, irregular: bool) -> "_Partition":
        """Create a partition's arrays from the first signal that lands on it.

        Args:
            signal: The first signal of the partition, for its dtype.
            irregular: Whether the partition carries per-value time offsets.

        Returns:
            The partition.

        Raises:
            ImportError: If the ``zarr`` extra is not installed.
        """
        try:
            import zarr  # noqa: PLC0415
            from zarr.codecs import BloscCodec  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("the zarr values backend needs the zarr extra: pip install 'timenet[zarr]'") from exc

        if self._group is None:
            self._group = zarr.open_group(store=self._staging_dir / STORE_DIR, mode="w")
        codec = BloscCodec(
            cname=cast(Literal["zstd", "lz4", "lz4hc", "zlib", "blosclz"], self._compression),
            clevel=self._compression_level,
            shuffle="bitshuffle",
        )
        itemsize = np.dtype(signal.dtype).itemsize
        values = _Appender(
            self._group.create_array(
                name=values_array_path(signal.spec_type, irregular).removeprefix(f"{STORE_DIR}/"),
                shape=(0,),
                dtype=signal.dtype,
                chunks=(max(1, self._chunk_max_bytes // itemsize),),
                compressors=codec,
            ),
            self._buffer_target_bytes // itemsize,
        )
        offsets = None
        if irregular:
            offsets = _Appender(
                self._group.create_array(
                    name=time_offsets_array_path(signal.spec_type).removeprefix(f"{STORE_DIR}/"),
                    shape=(0,),
                    dtype="int64",
                    chunks=(max(1, self._chunk_max_bytes // _BYTES_PER_TIME_OFFSET),),
                    compressors=codec,
                ),
                self._buffer_target_bytes // _BYTES_PER_TIME_OFFSET,
            )
        return _Partition(values=values, time_offsets=offsets)


@dataclass
class _Partition:
    """One modality's values array, and the time offsets array that advances beside it."""

    values: "_Appender"
    time_offsets: "_Appender | None" = None

    def append(self, values: np.ndarray, time_offsets: np.ndarray | None) -> int:
        """Append one signal and report where its values start.

        Args:
            values: The signal's values.
            time_offsets: Its time offsets, or ``None`` for a regular partition.

        Returns:
            The element offset the values were written at.

        Raises:
            TimeFValidationError: If the signal disagrees with what this partition stores.
        """
        if (time_offsets is None) != (self.time_offsets is None):
            raise TimeFValidationError(
                "a signal's time offsets must match its partition: an irregular partition takes "
                "time offsets from every signal and any other takes none"
            )
        start = self.values.length
        self.values.append(values)
        if self.time_offsets is not None and time_offsets is not None:
            self.time_offsets.append(time_offsets)
        return start

    def finish(self) -> None:
        """Flush both arrays."""
        self.values.finish()
        if self.time_offsets is not None:
            self.time_offsets.finish()


class _Appender:
    """Buffers appends to one Zarr array and writes them on a chunk boundary.

    Without the buffer every signal resizes the array and rewrites the trailing chunk, so a corpus
    of N short signals rewrites the same chunk N times.
    """

    def __init__(self, array: Any, buffer_len: int) -> None:
        self._array = array
        self._chunk_len = array.chunks[0]
        self._buffer_len = max(self._chunk_len, buffer_len)
        self._written = 0
        self._buffer: list[np.ndarray] = []
        self._pending = 0

    @property
    def length(self) -> int:
        """How many elements the array logically holds, buffered ones included."""
        return self._written + self._pending

    def append(self, values: np.ndarray) -> None:
        """Buffer one signal's values, writing whole chunks once enough have piled up.

        Args:
            values: The values to append.
        """
        self._buffer.append(values)
        self._pending += len(values)
        if self._pending >= self._buffer_len:
            self._flush((self._pending // self._chunk_len) * self._chunk_len)

    def finish(self) -> None:
        """Write the trailing partial chunk."""
        self._flush(self._pending)

    def _flush(self, count: int) -> None:
        """Write the first ``count`` buffered elements and keep the rest.

        Args:
            count: How many elements to write.
        """
        if count == 0:
            return
        data = self._buffer[0] if len(self._buffer) == 1 else np.concatenate(self._buffer)
        self._array.resize((self._written + count,))
        self._array[self._written : self._written + count] = data[:count]
        self._written += count
        rest = data[count:]
        self._buffer = [rest] if len(rest) else []
        self._pending = len(rest)


def read_chunks(root: Path, locators: Sequence[ChunkLocator]) -> np.ndarray:
    """Read one signal's values back by locator.

    Each locator names an array and the element it starts at, so a read asks Zarr for one contiguous
    slice. Zarr fetches only the storage chunks that slice touches; nothing scans the array.

    Args:
        root: The version directory holding the store.
        locators: The signal's chunk locators, in chunk order.

    Returns:
        The signal's values, in the array's own dtype.

    Raises:
        ImportError: If the ``zarr`` extra is not installed.
    """
    try:
        import zarr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("this version stores values in Zarr; install the extra: pip install 'timenet[zarr]'") from exc

    opened: dict[str, Any] = {}
    pieces: list[np.ndarray] = []
    for locator in sorted(locators, key=lambda item: item.chunk_idx):
        array = opened.get(locator.chunk_file)
        if array is None:
            array = zarr.open_array(store=root / locator.chunk_file, mode="r")
            opened[locator.chunk_file] = array
        start = locator.chunk_major_idx
        pieces.append(np.asarray(array[start : start + locator.n_values]))
    return np.concatenate(pieces) if pieces else np.asarray([])
