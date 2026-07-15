"""Writer-side values backends: pluggable storage for the time-series values plane.

The values plane (the float32 waveform of every series) is the one part of a TimeF version whose
on-disk representation is swappable. Everything else — samples, annotations, tasks, and the
time-series *index* that locates each chunk — is backend-agnostic. A :class:`ValuesBackend` takes the
deduped, sorted series and writes their values however it likes, returning a generic
:class:`ChunkPlacement` per chunk plus the list of value files to record in the manifest. The core
writer stays ignorant of shards, row groups, or arrays.

The default :class:`ParquetValuesBackend` streams ``list<float32>`` chunks into rotating Parquet shards
(BYTE_STREAM_SPLIT + zstd).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types.ids import id_to_bytes
from timenet.values_backends import PARQUET_VALUES_BACKEND
from timenet.writer import encodings
from timenet.writer.constants import SHARD_TEMPLATE
from timenet.writer.schemas import IdTypes, shard_schema


MAX_ELEMENTS_PER_ROW_GROUP = 2**31
_BYTES_PER_FLOAT32 = 4


@dataclass
class ChunkPlacement:
    """Where one chunk of a series landed, plus the metadata the index needs.

    The backend-neutral locator identifies a chunk within a values file.
    """

    chunk_file: str
    """Values file (relative to the staging directory) holding this chunk."""
    chunk_offset0: int
    """First backend-defined coordinate (the Parquet row group)."""
    chunk_offset1: int
    """Second backend-defined coordinate (the Parquet row offset)."""
    spec_type: str
    """Spec type of the source series."""
    channel: str
    """Channel name of the source series."""
    t_start_s: float
    """Start time of the chunk in seconds."""
    n_values: int
    """Number of values in the chunk."""
    sampling_rate_hz: float
    """Sampling rate of the series in Hz."""


@dataclass
class ValuesWriteResult:
    """The outcome of writing every series' values with a backend."""

    placements: dict[tuple[str, int], ChunkPlacement]
    """``(time_series_id, chunk_idx)`` -> its on-disk placement."""
    files: list[str] = field(default_factory=list)
    """Value files produced, relative to the staging directory, for ``manifest.files.time_series``."""


@dataclass(frozen=True)
class ParquetValuesConfig:
    """Typed construction options for the Parquet values backend."""

    staging_dir: Path
    """Version staging directory; shards are written beneath it."""
    tsid_uuid16: bool
    """Whether ``time_series_id`` is stored as ``binary(16)``."""
    id_types: IdTypes
    """Resolved logical-id storage types."""
    shard_target_bytes: int
    """Target size for rotating shards."""
    row_group_target_bytes: int
    """Target size for flushing row groups."""
    chunk_max_bytes: int
    """Maximum uncompressed values size of one logical chunk."""
    compression: str
    """Parquet compression codec."""
    compression_level: int
    """Parquet compression level."""


ValuesBackendConfig = ParquetValuesConfig


def make_values_backend(config: ValuesBackendConfig) -> "ValuesBackend":
    """Construct the values backend described by ``config``.

    Args:
        config: Backend-specific typed construction options.

    Returns:
        The constructed backend.
    """
    return ParquetValuesBackend(config)


class ValuesBackend(Protocol):
    """Writes the values plane of a dataset and reports where each chunk landed."""

    name: str
    """Manifest ``values_backend`` tag identifying this backend on read-back."""

    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Write every series' values and return their placements plus the produced files.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series' float32 values (the writer's contract
                check).
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(files_finalized)`` after each value file closes.

        Returns:
            The chunk placements and the value files written.
        """
        ...


class ParquetValuesBackend:
    """Streams series into rotating Parquet shards of ``list<float32>`` chunks (the default backend)."""

    name = PARQUET_VALUES_BACKEND

    def __init__(self, config: ParquetValuesConfig) -> None:
        """Configure the Parquet shard backend.

        Args:
            config: Typed Parquet backend options.
        """
        self._staging_dir = config.staging_dir
        self._tsid_uuid16 = config.tsid_uuid16
        self._shard_schema = shard_schema(config.id_types)
        self._shard_target_bytes = config.shard_target_bytes
        self._row_group_target_bytes = config.row_group_target_bytes
        self._chunk_max_bytes = config.chunk_max_bytes
        self._compression = config.compression
        self._compression_level = config.compression_level
        self._bss_checked = False

    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Stream all series into rotating shard files and return chunk placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series' float32 values.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(files_finalized)`` after each shard closes.

        Returns:
            The chunk placements and the shard files written.
        """
        stream = _ShardStream(self, self._row_group_target_bytes, self._shard_target_bytes, on_file_done)
        total = len(unique_series)
        for completed, ts in enumerate(unique_series, start=1):
            values = read_and_validate(ts)
            for chunk in _plan_chunks(ts, values, self._chunk_max_bytes):
                stream.add(chunk)
            on_series_done(completed, total)
        stream.finish()
        return ValuesWriteResult(placements=stream.placements, files=list(stream.shard_paths))

    def _new_shard_writer(self, rel_path: str) -> pq.ParquetWriter:
        path = self._staging_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(
            path,
            self._shard_schema,
            **encodings.parquet_kwargs(
                dictionary_columns=encodings.SHARD_DICTIONARY,
                column_encoding=encodings.SHARD_ENCODING,
                compression=self._compression,
                compression_level=self._compression_level,
            ),
        )

    def _verify_bss_once(self, rel_path: str) -> None:
        """Verify BYTE_STREAM_SPLIT was applied on a finalized shard (guards silent path mismatch).

        Args:
            rel_path: The finalized shard's path relative to the staging directory.

        Raises:
            TimeFValidationError: If the values column is not BYTE_STREAM_SPLIT encoded.
        """
        if self._bss_checked:
            return
        path = self._staging_dir / rel_path
        applied = encodings.values_encoding_of(str(path))
        if not applied:  # empty shard (no values written yet); nothing to verify
            return
        if "BYTE_STREAM_SPLIT" not in applied:
            raise TimeFValidationError(f"expected BYTE_STREAM_SPLIT on shard values, got {sorted(applied)}")
        self._bss_checked = True


@dataclass
class _Chunk:
    """One sub-chunk of a series buffered for the current row group."""

    time_series_id: str
    spec_type: str
    channel: str
    chunk_idx: int
    t_start_s: float
    n_values: int
    sampling_rate_hz: float
    values: pa.Array


class _ShardStream:
    """Buffers chunks into rotating shard files and records each chunk's on-disk placement."""

    def __init__(
        self,
        backend: ParquetValuesBackend,
        row_group_target_bytes: int,
        shard_target_bytes: int,
        on_file_done: Callable[[int], None],
    ) -> None:
        """Bind the stream to its backend and the byte targets that trigger flush/rotation.

        Args:
            backend: The owning backend, used to open shards and verify encoding.
            row_group_target_bytes: Flush a row group once buffered values exceed this.
            shard_target_bytes: Rotate to a new shard once a shard's written values exceed this.
            on_file_done: Progress callback invoked with the finalized shard count.
        """
        self._backend = backend
        self._row_group_target_bytes = row_group_target_bytes
        self._shard_target_bytes = shard_target_bytes
        self._on_file_done = on_file_done
        self._shard: pq.ParquetWriter | None = None
        self._shard_idx = 0
        self._row_group = 0
        self._shard_bytes = 0
        self._buffer: list[_Chunk] = []
        self._buffer_bytes = 0
        self.shard_paths: list[str] = []
        self.placements: dict[tuple[str, int], ChunkPlacement] = {}

    def add(self, chunk: _Chunk) -> None:
        """Buffer one chunk, flushing a row group and rotating shards as the byte targets are reached."""
        self._buffer.append(chunk)
        self._buffer_bytes += chunk.n_values * _BYTES_PER_FLOAT32
        if self._buffer_bytes >= self._row_group_target_bytes:
            self._flush()
            if self._shard_bytes >= self._shard_target_bytes:
                self._close_shard()
                self._shard_idx += 1
                self._row_group = 0
                self._shard_bytes = 0

    def finish(self) -> None:
        """Flush any remaining buffered chunks and close the final shard."""
        self._flush()
        self._close_shard()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if sum(chunk.n_values for chunk in self._buffer) >= MAX_ELEMENTS_PER_ROW_GROUP:
            raise TimeFValidationError("row group would exceed the 2^31 element limit")
        shard = self._shard
        if shard is None:
            rel = SHARD_TEMPLATE.format(self._shard_idx)
            self.shard_paths.append(rel)
            shard = self._backend._new_shard_writer(rel)
            self._shard = shard
        shard.write_table(_shard_table(self._buffer, self._backend._shard_schema, self._backend._tsid_uuid16))
        shard_path = self.shard_paths[self._shard_idx]
        for offset, chunk in enumerate(self._buffer):
            self.placements[chunk.time_series_id, chunk.chunk_idx] = ChunkPlacement(
                chunk_file=shard_path,
                chunk_offset0=self._row_group,
                chunk_offset1=offset,
                spec_type=chunk.spec_type,
                channel=chunk.channel,
                t_start_s=chunk.t_start_s,
                n_values=chunk.n_values,
                sampling_rate_hz=chunk.sampling_rate_hz,
            )
        self._row_group += 1
        self._shard_bytes += self._buffer_bytes
        self._buffer = []
        self._buffer_bytes = 0

    def _close_shard(self) -> None:
        if self._shard is None:
            return
        self._shard.close()
        self._backend._verify_bss_once(self.shard_paths[self._shard_idx])
        self._on_file_done(self._shard_idx + 1)
        self._shard = None


def _shard_table(buffer: list[_Chunk], schema: pa.Schema, tsid_uuid16: bool) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "time_series_id": [id_to_bytes(c.time_series_id) for c in buffer]
            if tsid_uuid16
            else [c.time_series_id for c in buffer],
            "spec_type": [c.spec_type for c in buffer],
            "channel": [c.channel for c in buffer],
            "chunk_idx": [c.chunk_idx for c in buffer],
            "t_start_s": [c.t_start_s for c in buffer],
            "n_values": [c.n_values for c in buffer],
            "sampling_rate_hz": [c.sampling_rate_hz for c in buffer],
            "values": _values_column([c.values for c in buffer]),
        },
        schema=schema,
    )


def _plan_chunks(ts: TimeSeries, values: pa.Array, chunk_max_bytes: int) -> list[_Chunk]:
    """Split a validated series into backend-independent logical chunks.

    Args:
        ts: Series metadata used for chunk identity and timing.
        values: Validated float32 values.
        chunk_max_bytes: Maximum uncompressed values bytes in one chunk.

    Returns:
        Logical chunks in series order.
    """
    max_values = max(1, chunk_max_bytes // _BYTES_PER_FLOAT32)
    chunks: list[_Chunk] = []
    for chunk_idx, start in enumerate(range(0, len(values), max_values)):
        sub = values.slice(start, max_values)
        chunks.append(
            _Chunk(
                time_series_id=ts.time_series_id,
                spec_type=ts.spec.spec_type,
                channel=ts.channel,
                chunk_idx=chunk_idx,
                t_start_s=ts.t_start_s + start / ts.sampling_rate_hz,
                n_values=len(sub),
                sampling_rate_hz=ts.sampling_rate_hz,
                values=sub,
            )
        )
    return chunks


def _values_column(chunks: list[pa.Array]) -> pa.ListArray:
    """Pack per-chunk float32 arrays into one ``list<float32>`` column without boxing to Python floats.

    Args:
        chunks: The per-row float32 value arrays (one per chunk in the row group).

    Returns:
        A ``list<float32>`` array with one row per chunk.
    """
    offsets = np.zeros(len(chunks) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(chunk) for chunk in chunks])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.concat_arrays(chunks))
