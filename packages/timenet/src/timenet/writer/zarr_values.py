"""Zarr values backend (writer side): stream series into one Zarr array per ``spec_type``.

An alternative to the default Parquet shard store, laid out for Zarr's strengths rather than mirroring
Parquet's:

- Every series of a modality is appended along time to one typed array named for its ``spec_type``
  inside a ``time_series.zarr`` group. Zarr chunks the storage itself, so a series is **one index row**
  (one placement spanning its full length), not a run of ``chunk_max_bytes`` logical chunks.
- Appends are buffered per ``spec_type`` and flushed at shard-aligned boundaries, so every Zarr shard
  object is written exactly once — a naive ``resize()``-per-series append re-writes (read-modify-write)
  the trailing shard for every series. The writer feeds series sorted by ``spec_type``, so one appender
  is active at a time and buffered memory stays bounded near one shard.

A chunk is located by ``(array path, element start)``; the shared index carries that in its
backend-agnostic ``chunk_file`` / ``chunk_major_idx`` locator (``chunk_minor_idx`` is unused).

Requires the ``zarr`` extra (``pip install 'timenet[zarr]'``); imported lazily so the Parquet core never
needs it.
"""

from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.values_backends import ValuesBackend
from timenet.writer.values import (
    BaseValuesBackend,
    ChunkDataIndex,
    ChunkPlacement,
    ValuesWriteResult,
    ZarrValuesConfig,
)


_STORE_DIR = "time_series.zarr"
_BLOSC_CNAMES = frozenset({"zstd", "lz4", "lz4hc", "zlib", "blosclz"})
# One placement normally spans a whole series; split only to keep n_values inside int32 (the index
# column type). 2^30 values = 4 GiB of float32 per placement.
_MAX_PLACEMENT_VALUES = 2**30


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
    """Streams each series into a per-``spec_type`` typed N-D Zarr array (chunked + sharded)."""

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

    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Append every series to its spec-type array and return per-series placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series against its dtype and shape contract.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(arrays_finalized)`` as each spec-type array closes.

        Returns:
            The placements and the Zarr store's files (relative to the staging directory).

        Raises:
            ImportError: If the ``zarr`` extra is not installed.
        """
        try:
            import zarr
            from zarr.codecs import BloscCname, BloscCodec, BloscShuffle
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("the zarr values backend needs the zarr extra: pip install 'timenet[zarr]'") from exc

        store_path = self._staging_dir / _STORE_DIR
        group = zarr.open_group(store=store_path, mode="w")
        codec = BloscCodec(cname=BloscCname(self._cname), clevel=self._clevel, shuffle=BloscShuffle.bitshuffle)

        appenders: dict[str, _ArrayAppender] = {}
        placements: dict[tuple[str, int], ChunkPlacement] = {}
        active: str | None = None  # series arrive sorted by spec_type, so one appender is active at a time
        total = len(unique_series)
        for completed, ts in enumerate(unique_series, start=1):
            arrow_values = read_and_validate(ts)
            values = (
                arrow_values.to_numpy_ndarray()
                if isinstance(arrow_values, pa.FixedShapeTensorArray)
                else arrow_values.to_numpy(zero_copy_only=False)
            )
            spec_type = ts.spec.spec_type
            array_name = _array_name(spec_type)
            if spec_type != active:
                if active is not None:
                    appenders[active].finish()
                    on_file_done(len(appenders))
                if spec_type not in appenders:
                    bytes_per_step = np.dtype(ts.spec.dtype).itemsize * max(1, int(np.prod(ts.spec.value_shape)))
                    chunk_len = max(1, self._chunk_max_bytes // bytes_per_step)
                    shard_chunks = max(1, (self._shard_target_bytes // bytes_per_step) // chunk_len)
                    shard_len = shard_chunks * chunk_len
                    trailing = ts.spec.value_shape
                    appenders[spec_type] = _ArrayAppender(
                        group.create_array(
                            name=array_name,
                            shape=(0, *trailing),
                            dtype=ts.spec.dtype,
                            chunks=(chunk_len, *trailing),
                            shards=(shard_len, *trailing),
                            compressors=codec,
                        ),
                        shard_len,
                    )
                active = spec_type
            appender = appenders[spec_type]
            base = appender.logical_len
            appender.append(values)
            rel = f"{_STORE_DIR}/{array_name}"
            for chunk_idx, start in enumerate(range(0, len(values), _MAX_PLACEMENT_VALUES)):
                n = min(_MAX_PLACEMENT_VALUES, len(values) - start)
                placements[ts.time_series_id, chunk_idx] = ChunkPlacement(
                    chunk_file=rel,
                    data_index=ChunkDataIndex(major_idx=base + start, minor_idx=None),
                    spec_type=spec_type,
                    channel=ts.channel,
                    t_start_s=ts.t_start_s + start / ts.sampling_rate_hz,
                    n_values=n,
                    sampling_rate_hz=ts.sampling_rate_hz,
                )
            on_series_done(completed, total)
        if active is not None:
            appenders[active].finish()
            on_file_done(len(appenders))
        files = [
            path.relative_to(self._staging_dir).as_posix() for path in sorted(store_path.rglob("*")) if path.is_file()
        ]
        return ValuesWriteResult(placements=placements, files=files)


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
