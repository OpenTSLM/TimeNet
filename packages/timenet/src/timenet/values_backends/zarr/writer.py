"""Zarr values backend (writer side): one Zarr array per ``(spec_type, stores_time_offsets)`` partition.

An alternative to the default Parquet shard store, laid out for Zarr's strengths rather than mirroring
Parquet's:

- Every series of a modality is appended along time to one typed array inside a ``time_series.zarr``
  group. Zarr chunks the storage itself, so a series is **one index row** (one placement spanning its
  full length), not a run of ``chunk_max_bytes`` logical chunks.
- Arrays are partitioned by ``(spec_type, stores_time_offsets)``, not by ``spec_type`` alone. A series
  whose time offsets are stored writes its values under ``_irregular/`` and its int64 time offsets under
  ``_time_offsets/``, and the two advance in lockstep because every series in that partition contributes
  to both. That is what lets the index's single ``chunk_major_idx`` element offset address either one.
  Partitioning by ``spec_type`` alone would leave the time offsets array receiving only some of the
  series, so the offsets would drift apart with nothing to notice.
- Appends are buffered per partition and flushed at shard-aligned boundaries, so every Zarr shard
  object is written exactly once — a naive ``resize()``-per-series append re-writes (read-modify-write)
  the trailing shard for every series. The backend sorts series by partition, so only one partition is
  open at a time. An irregular partition keeps its values and its time offsets appender open together, so
  it buffers near two shards rather than one.

A chunk is located by ``(array path, element start)``; the shared index carries that in its
backend-agnostic ``chunk_file`` / ``chunk_major_idx`` locator (``chunk_minor_idx`` is unused).

Requires the ``zarr`` extra (``pip install 'timenet[zarr]'``); imported lazily so the Parquet core never
needs it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.values_backends import ValuesBackend
from timenet.values_backends.writer import (
    BaseValuesBackend,
    ChunkDataIndex,
    ChunkPlacement,
    ValuesWriteResult,
)
from timenet.values_backends.zarr.config import ZarrValuesConfig


_STORE_DIR = "time_series.zarr"
_BLOSC_CNAMES = frozenset({"zstd", "lz4", "lz4hc", "zlib", "blosclz"})
# One placement normally spans a whole series; split only to keep n_values inside int32 (the index
# column type). 2^30 values = 4 GiB of float32 per placement.
_MAX_PLACEMENT_VALUES = 2**30
_BYTES_PER_TIME_OFFSET = 8


#: Group holding the values of series that store per-value time offsets, kept apart from the regular
#: arrays so a partition's values and time offsets advance together and one offset addresses both.
_IRREGULAR_GROUP = "_irregular"
#: Group holding those series' int64 time offsets, one array per spec type, parallel to _IRREGULAR_GROUP.
_TIME_OFFSETS_GROUP = "_time_offsets"


def _value_array_path(spec_type: str, stores_time_offsets: bool) -> str:
    """Return the array path holding one partition's values.

    Args:
        spec_type: The modality tag.
        stores_time_offsets: Whether the partition's series store per-value time offsets.

    Returns:
        The path relative to the Zarr store root.
    """
    name = _array_name(spec_type)
    return f"{_IRREGULAR_GROUP}/{name}" if stores_time_offsets else name


def _time_offsets_array_path(spec_type: str) -> str:
    """Return the array path holding one partition's time offsets.

    Args:
        spec_type: The modality tag.

    Returns:
        The path relative to the Zarr store root.
    """
    return f"{_TIME_OFFSETS_GROUP}/{_array_name(spec_type)}"


def _array_name(spec_type: str) -> str:
    """Encode a logical spec type as one filesystem-safe Zarr path segment.

    Percent-encodes every character outside the URL-unreserved set, so path separators never leak into
    the path and distinct spec types map to distinct single segments (the encoding is reversible, hence
    injective). ``.`` and ``..`` are the only unreserved-yet-unsafe segments; :class:`TimeSeriesSpec`
    rejects them.

    Returns:
        The percent-encoded physical array name.
    """
    return quote(spec_type, safe="")


class ZarrValuesBackend(BaseValuesBackend):
    """Streams each series into a per-partition typed N-D Zarr array (chunked + sharded)."""

    name = ValuesBackend.ZARR

    def __init__(self, config: ZarrValuesConfig) -> None:
        """Configure the Zarr backend.

        Args:
            config: Typed Zarr backend options.
        """
        self._staging_dir = config.staging_dir
        self._chunk_max_bytes = config.chunk_max_bytes
        self._shard_target_bytes = config.shard_target_bytes
        self._cname = config.compression if config.compression in _BLOSC_CNAMES else "zstd"
        self._clevel = config.compression_level

    def _value_appender(self, group: Any, ts: TimeSeries, array_path: str, codec: Any) -> "_ArrayAppender":
        """Create the values array for one partition and wrap it in an appender.

        Args:
            group: The open Zarr group.
            ts: The first series of the partition, for its dtype and per-step shape.
            array_path: Where the array goes inside the group.
            codec: The Blosc compressor.

        Returns:
            The appender for that array.
        """
        bytes_per_step = np.dtype(ts.spec.dtype).itemsize * max(1, int(np.prod(ts.spec.value_shape)))
        chunk_len = max(1, self._chunk_max_bytes // bytes_per_step)
        shard_len = max(1, (self._shard_target_bytes // bytes_per_step) // chunk_len) * chunk_len
        trailing = ts.spec.value_shape
        return _ArrayAppender(
            group.create_array(
                name=array_path,
                shape=(0, *trailing),
                dtype=ts.spec.dtype,
                chunks=(chunk_len, *trailing),
                shards=(shard_len, *trailing),
                compressors=codec,
            ),
            shard_len,
        )

    def _time_offsets_appender(self, group: Any, spec_type: str, codec: Any, delta: Any) -> "_ArrayAppender":
        """Create the time offsets array parallel to a partition's values, and wrap it in an appender.

        The Delta filter is what makes stored time offsets cheap: they are monotonic, so the deltas are
        small and the Blosc bitshuffle then has little left to do. Measured at 15-17% over bitshuffle
        alone on realistic irregular spacing.

        Args:
            group: The open Zarr group.
            spec_type: The modality tag naming the array.
            codec: The Blosc compressor.
            delta: The Delta codec class, imported lazily with the rest of zarr.

        Returns:
            The appender for that array.
        """
        chunk_len = max(1, self._chunk_max_bytes // _BYTES_PER_TIME_OFFSET)
        # Rounded down to a whole number of chunks, exactly as the values array is: Zarr requires the
        # shard shape to be a multiple of the chunk shape, and an unrounded value aborts create_array
        # for any byte target where the two do not divide.
        shard_len = max(1, (self._shard_target_bytes // _BYTES_PER_TIME_OFFSET) // chunk_len) * chunk_len
        return _ArrayAppender(
            group.create_array(
                name=_time_offsets_array_path(spec_type),
                shape=(0,),
                dtype="int64",
                chunks=(chunk_len,),
                shards=(shard_len,),
                filters=[delta(dtype="int64")],
                compressors=codec,
            ),
            shard_len,
        )

    def write_series(  # noqa: PLR0914
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        read_time_offsets: Callable[[TimeSeries], pa.Array | None],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Append every series to its spec-type array and return per-series placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series against its dtype and shape contract.
            read_time_offsets: Loads an irregular series' int64 time offsets, or ``None`` for other shapes.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(arrays_finalized)`` as each partition closes,
                counting a partition's values array plus its time offsets array when it has one.

        Returns:
            The placements and the Zarr store's files (relative to the staging directory).

        Raises:
            ImportError: If the ``zarr`` extra is not installed.
        """
        try:
            import zarr  # noqa: PLC0415
            from zarr.codecs import BloscCname, BloscCodec, BloscShuffle  # noqa: PLC0415
            from zarr.codecs.numcodecs import Delta  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("the zarr values backend needs the zarr extra: pip install 'timenet[zarr]'") from exc

        store_path = self._staging_dir / _STORE_DIR
        group = zarr.open_group(store=store_path, mode="w")
        codec = BloscCodec(cname=BloscCname(self._cname), clevel=self._clevel, shuffle=BloscShuffle.bitshuffle)

        partitions: dict[tuple[str, bool], _Partition] = {}
        placements: dict[tuple[str, int], ChunkPlacement] = {}
        closed = 0
        active: tuple[str, bool] | None = None  # one partition is open at a time, see the sort below
        total = len(unique_series)
        # Stable, so the caller's (spec_type, channel, time_series_id) order survives within a
        # partition. Grouping by whether a series stores time offsets is what keeps its values array and
        # its time offsets array the same length: every series in an irregular partition contributes to
        # both, so one element offset addresses either.
        ordered = sorted(unique_series, key=lambda ts: (ts.spec.spec_type, ts.time_offsets_loader is not None))
        for completed, ts in enumerate(ordered, start=1):
            arrow_values = read_and_validate(ts)
            values = (
                arrow_values.to_numpy_ndarray()
                if isinstance(arrow_values, pa.FixedShapeTensorArray)
                else arrow_values.to_numpy(zero_copy_only=False)
            )
            arrow_time_offsets = read_time_offsets(ts)
            spec_type = ts.spec.spec_type
            stores_time_offsets = arrow_time_offsets is not None
            partition = (spec_type, stores_time_offsets)
            array_path = _value_array_path(spec_type, stores_time_offsets)
            if partition != active:
                if active is not None:
                    closed += partitions[active].finish()
                    on_file_done(closed)
                if partition not in partitions:
                    partitions[partition] = _Partition(
                        values=self._value_appender(group, ts, array_path, codec),
                        time_offsets=self._time_offsets_appender(group, spec_type, codec, Delta)
                        if stores_time_offsets
                        else None,
                    )
                active = partition
            base = partitions[partition].append(
                values,
                None if arrow_time_offsets is None else arrow_time_offsets.to_numpy(zero_copy_only=False),
            )
            rel = f"{_STORE_DIR}/{array_path}"
            for chunk_idx, start in enumerate(range(0, len(values), _MAX_PLACEMENT_VALUES)):
                n = min(_MAX_PLACEMENT_VALUES, len(values) - start)
                placements[ts.time_series_id, chunk_idx] = ChunkPlacement(
                    chunk_file=rel,
                    data_index=ChunkDataIndex(major_idx=base + start, minor_idx=None),
                    spec_type=spec_type,
                    channel=ts.channel,
                    n_values=n,
                )
            on_series_done(completed, total)
        if active is not None:
            closed += partitions[active].finish()
            on_file_done(closed)
        files = [
            path.relative_to(self._staging_dir).as_posix() for path in sorted(store_path.rglob("*")) if path.is_file()
        ]
        return ValuesWriteResult(placements=placements, files=files)


@dataclass
class _Partition:
    """One ``(spec_type, stores_time_offsets)`` partition: a values array and, when irregular, its time offsets.

    Holding the pair together is what keeps them the same length. Every series in an irregular
    partition contributes to both, so one element offset addresses either, and a caller cannot append
    to one and forget the other.
    """

    values: "_ArrayAppender"
    time_offsets: "_ArrayAppender | None" = None

    def append(self, values: np.ndarray, time_offsets: np.ndarray | None) -> int:
        """Append one series to the partition and report where its values landed.

        Args:
            values: The series values.
            time_offsets: Its int64 time offsets, or ``None`` for a partition that stores none.

        Returns:
            The element offset the values were written at.

        Raises:
            TimeFValidationError: If ``time_offsets`` disagrees with what this partition stores.
        """
        if (time_offsets is None) != (self.time_offsets is None):
            raise TimeFValidationError(
                "a series' time offsets must match its partition: an irregular partition takes time offsets "
                "from every series and any other takes none"
            )
        base = self.values.logical_len
        self.values.append(values)
        if self.time_offsets is not None and time_offsets is not None:
            self.time_offsets.append(time_offsets)
        return base

    def finish(self) -> int:
        """Flush the partition's arrays.

        Returns:
            How many Zarr arrays this partition finalized: two when it stores time offsets, else one.
        """
        self.values.finish()
        if self.time_offsets is None:
            return 1
        self.time_offsets.finish()
        return 2


class _ArrayAppender:
    """Buffers appends to one Zarr array and flushes at shard-aligned boundaries.

    Writing only whole, aligned shards means each shard object is created exactly once; the single
    trailing partial shard is written by :meth:`finish`. Without this, every per-series append would
    re-write (read-modify-write) the trailing shard.
    """

    def __init__(self, array: Any, shard_len: int) -> None:
        self._array = array
        self._shard_len = shard_len
        self._written = 0
        self._buffer: list[Shaped[np.ndarray, " time *value"]] = []
        self._buffer_len = 0

    @property
    def logical_len(self) -> int:
        """The array's length including not-yet-flushed values (the next append's base offset)."""
        return self._written + self._buffer_len

    def append(self, values: Shaped[np.ndarray, " time *value"]) -> None:
        """Buffer one series' values, flushing every completed shard.

        Args:
            values: The series values with their per-step dimensions.
        """
        self._buffer.append(values)
        self._buffer_len += len(values)
        if self._buffer_len >= self._shard_len:
            self._flush((self._buffer_len // self._shard_len) * self._shard_len)

    def finish(self) -> None:
        """Flush the trailing partial shard. Safe to call on an already-finished appender."""
        self._flush(self._buffer_len)

    def _flush(self, n: int) -> None:
        if n == 0:
            return
        data = self._buffer[0] if len(self._buffer) == 1 else np.concatenate(self._buffer)
        self._array.resize((self._written + n, *self._array.shape[1:]))
        self._array[self._written : self._written + n] = data[:n]
        self._written += n
        remainder = data[n:]
        self._buffer = [remainder] if len(remainder) else []
        self._buffer_len = len(remainder)
