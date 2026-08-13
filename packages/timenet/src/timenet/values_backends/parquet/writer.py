"""The default values backend: streams ``list<float32>`` chunks into rotating Parquet shards.

Shards are zstd-compressed and single-modality, each carrying the values encoding chosen for its
``spec_type`` (see :mod:`timenet.writer.value_encoding`). Each chunk's placement is recorded in the
backend-neutral time-series index as a ``chunk_file`` plus a
:class:`~timenet.values_backends.writer.ChunkDataIndex` — for Parquet, the shard path, the row group
(``major_idx``), and the row offset (``minor_idx``).
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.format.constants import SHARD_TEMPLATE
from timenet.format.schemas import IdCodec, shard_schema
from timenet.values_backends import ValuesBackend
from timenet.values_backends.parquet.config import ParquetValuesConfig
from timenet.values_backends.writer import (
    BaseValuesBackend,
    ChunkDataIndex,
    ChunkPlacement,
    ValuesWriteResult,
)
from timenet.writer import encodings
from timenet.writer.value_encoding import (
    DICT_MAX_CARDINALITY,
    ValueEncoding,
    distinct_bit_patterns,
    encoding_for_cardinality,
    sample_values,
)


MAX_ELEMENTS_PER_ROW_GROUP = 2**31
_BYTES_PER_FLOAT32 = 4
_BYTES_PER_TIME_OFFSET = 8
_LOG = logging.getLogger(__name__)


def _step_bytes(stores_time_offsets: bool) -> int:
    """Return the uncompressed bytes one step of a series costs.

    An irregular series carries an int64 time offset beside each float32 value, so its step costs three
    times a regular one. Chunk sizing, row-group flushing and shard rotation all budget in these
    units; charging every series the float32 rate would let an irregular one overrun each target
    threefold.

    Args:
        stores_time_offsets: Whether the series stores one time offset per value.

    Returns:
        Bytes per step.
    """
    return _BYTES_PER_FLOAT32 + (_BYTES_PER_TIME_OFFSET if stores_time_offsets else 0)


class ParquetValuesBackend(BaseValuesBackend):
    """Streams series into rotating Parquet shards of ``list<float32>`` chunks (the default backend)."""

    name = ValuesBackend.PARQUET

    def __init__(self, config: ParquetValuesConfig) -> None:
        """Configure the Parquet shard backend.

        Args:
            config: Typed Parquet backend options.
        """
        self._staging_dir = config.staging_dir
        self._codec = config.codec
        self._shard_schema = shard_schema(config.id_types)
        self._shard_target_bytes = config.shard_target_bytes
        self._row_group_target_bytes = config.row_group_target_bytes
        self._chunk_max_bytes = config.chunk_max_bytes
        self._compression = config.compression
        self._compression_level = config.compression_level
        self._forced_encoding = config.value_encoding
        self._time_offsets_checked = False

    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        read_time_offsets: Callable[[TimeSeries], pa.Array | None],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Stream all series into rotating shard files and return chunk placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series' float32 values.
            read_time_offsets: Loads an irregular series' int64 time offsets, or ``None`` for other shapes.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(files_finalized)`` after each shard closes.

        Returns:
            The chunk placements, the shard files written, and the encoding applied per ``spec_type``.

        Raises:
            TimeFValidationError: If a spec uses an N-D shape or non-float32 dtype.
        """
        unsupported = [ts.spec.spec_type for ts in unique_series if ts.spec.value_shape or ts.spec.dtype != "float32"]
        if unsupported:
            raise TimeFValidationError(
                "the Parquet values backend currently supports scalar float32 series only; "
                f"use values_backend='zarr' for N-D/dtyped specs: {sorted(set(unsupported))}"
            )
        stream = _ShardStream(self, self._row_group_target_bytes, self._shard_target_bytes, on_file_done)
        total = len(unique_series)
        for completed, ts in enumerate(unique_series, start=1):
            values = read_and_validate(ts)
            for chunk in _plan_chunks(ts, values, self._chunk_max_bytes, read_time_offsets(ts)):
                stream.add(chunk)
            on_series_done(completed, total)
        stream.finish()
        return ValuesWriteResult(
            placements=stream.placements,
            files=list(stream.shard_paths),
            value_encoding={spec_type: str(encoding) for spec_type, encoding in stream.encodings.items()},
        )

    def encoding_for(self, spec_type: str, buffered: list[pa.Array]) -> ValueEncoding:
        """Return the encoding to use for a modality, deciding it on first sight and logging the choice.

        The caller passes the values it has buffered for the modality's first row group, so the
        decision costs a distinct-value count over data already in memory: no second pass over the
        series, no extra loader calls, and the buffer stays bounded by ``row_group_target_bytes``. The
        decision is logged at INFO under this module's logger, so a curator can see per ``spec_type``
        what was chosen and, for the auto path, the cardinality that drove it.

        Args:
            spec_type: The modality the decision is for, named in the log line.
            buffered: The buffered chunks' values, used only the first time a modality appears.

        Returns:
            The forced encoding when the caller set one, else the encoding selected from the sample.
        """
        if self._forced_encoding is not None:
            _LOG.info("values encoding for %r: %s (forced)", spec_type, self._forced_encoding.value)
            return self._forced_encoding
        arrays = [np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32) for values in buffered]
        sample = sample_values(arrays)
        distinct = distinct_bit_patterns(sample)
        encoding = encoding_for_cardinality(distinct)
        _LOG.info(
            "values encoding for %r: %s (auto; %d distinct in %d sampled values, dictionary up to %d)",
            spec_type,
            encoding.value,
            distinct,
            sample.size,
            DICT_MAX_CARDINALITY,
        )
        return encoding

    def _new_shard_writer(self, rel_path: str, value_encoding: ValueEncoding) -> pq.ParquetWriter:
        path = self._staging_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(
            path,
            self._shard_schema,
            **encodings.parquet_kwargs(
                dictionary_columns=encodings.shard_dictionary(value_encoding),
                column_encoding=encodings.shard_encoding(value_encoding),
                compression=self._compression,
                compression_level=self._compression_level,
            ),
        )

    def _verify_values_encoding(self, rel_path: str, value_encoding: ValueEncoding) -> None:
        """Verify a finalized shard's values column carries the encoding that was selected for it.

        pyarrow drops a column encoding silently when the column path does not match, so without this
        the size the selection was made for would not be the size on disk. Parquet's own
        dictionary-to-plain fallback is permitted: it happens when a dictionary outgrows its page
        limit, is lossless, and is not something the writer chose.

        Args:
            rel_path: The finalized shard's path relative to the staging directory.
            value_encoding: The encoding the shard was opened with.

        Raises:
            TimeFValidationError: If the values column does not carry the selected encoding, or a
                shard that carries time offsets does not encode them DELTA_BINARY_PACKED.
        """
        path = self._staging_dir / rel_path
        applied = encodings.values_encoding_of(str(path))
        if not applied:  # empty shard (no values written yet); nothing to verify
            return
        if not encodings.applied_matches(value_encoding, applied):
            raise TimeFValidationError(
                f"expected {value_encoding} on the shard values column of {rel_path}, got {sorted(applied)}"
            )
        if not self._time_offsets_checked:
            time_offsets = encodings.values_encoding_of(str(path), encodings.TIME_OFFSETS_COLUMN)
            # An all-regular shard carries only RLE definition levels here and has no time offsets to
            # encode, so it settles nothing and the check stays armed for a later shard.
            if time_offsets and time_offsets != {"RLE"}:
                if "DELTA_BINARY_PACKED" not in time_offsets:
                    raise TimeFValidationError(
                        f"expected DELTA_BINARY_PACKED on shard time offsets, got {sorted(time_offsets)}"
                    )
                self._time_offsets_checked = True


@dataclass
class _Chunk:
    """One sub-chunk of a series buffered for the current row group."""

    time_series_id: str
    spec_type: str
    channel: str
    chunk_idx: int
    n_values: int
    values: pa.Array
    time_offsets: pa.Array | None = None
    """This chunk's int64 time offsets, sliced at the same boundary as its values, or ``None``."""

    @property
    def n_bytes(self) -> int:
        """Uncompressed bytes this chunk contributes to the row-group and shard budgets."""
        return self.n_values * _step_bytes(self.time_offsets is not None)


class _ShardStream:
    """Buffers chunks into rotating shard files and records each chunk's on-disk placement.

    Shards are single-modality: the writer hands chunks over sorted by ``spec_type``, and the stream
    rotates whenever that changes. A shard fixes its column encodings when it is opened, so keeping
    one modality per shard is what lets each modality carry the encoding chosen for it.
    """

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
        self._shard_encoding: ValueEncoding | None = None
        self._shard_idx = 0
        self._row_group = 0
        self._shard_bytes = 0
        self._spec_type: str | None = None
        self._buffer: list[_Chunk] = []
        self._buffer_bytes = 0
        self.shard_paths: list[str] = []
        self.placements: dict[tuple[str, int], ChunkPlacement] = {}
        self.encodings: dict[str, ValueEncoding] = {}
        """The encoding settled on for each ``spec_type``, in first-seen order."""

    def add(self, chunk: _Chunk) -> None:
        """Buffer one chunk, flushing a row group and rotating shards as the boundaries are reached."""
        if self._spec_type is not None and chunk.spec_type != self._spec_type:
            self._flush()
            self._rotate()
        self._spec_type = chunk.spec_type
        self._buffer.append(chunk)
        self._buffer_bytes += chunk.n_bytes
        if self._buffer_bytes >= self._row_group_target_bytes:
            self._flush()
            if self._shard_bytes >= self._shard_target_bytes:
                self._rotate()

    def finish(self) -> None:
        """Flush any remaining buffered chunks and close the final shard."""
        self._flush()
        self._close_shard()

    def _rotate(self) -> None:
        """Close the open shard and start counting a fresh one.

        A no-op when no shard is open: the previous rotation already advanced the index, and
        advancing again would leave a gap that no file is ever written for.
        """
        if self._shard is None:
            return
        self._close_shard()
        self._shard_idx += 1
        self._row_group = 0
        self._shard_bytes = 0

    def _flush(self) -> None:
        if not self._buffer:
            return
        if sum(chunk.n_values for chunk in self._buffer) >= MAX_ELEMENTS_PER_ROW_GROUP:
            raise TimeFValidationError("row group would exceed the 2^31 element limit")
        shard = self._shard
        if shard is None:
            spec_type = self._buffer[0].spec_type
            if spec_type not in self.encodings:
                self.encodings[spec_type] = self._backend.encoding_for(
                    spec_type, [chunk.values for chunk in self._buffer]
                )
            self._shard_encoding = self.encodings[spec_type]
            rel = SHARD_TEMPLATE.format(self._shard_idx)
            self.shard_paths.append(rel)
            shard = self._backend._new_shard_writer(rel, self._shard_encoding)
            self._shard = shard
        shard.write_table(_shard_table(self._buffer, self._backend._shard_schema, self._backend._codec))
        shard_path = self.shard_paths[self._shard_idx]
        for offset, chunk in enumerate(self._buffer):
            self.placements[chunk.time_series_id, chunk.chunk_idx] = ChunkPlacement(
                chunk_file=shard_path,
                data_index=ChunkDataIndex(major_idx=self._row_group, minor_idx=offset),
                spec_type=chunk.spec_type,
                channel=chunk.channel,
                n_values=chunk.n_values,
            )
        self._row_group += 1
        self._shard_bytes += self._buffer_bytes
        self._buffer = []
        self._buffer_bytes = 0

    def _close_shard(self) -> None:
        if self._shard is None or self._shard_encoding is None:
            return
        self._shard.close()
        self._backend._verify_values_encoding(self.shard_paths[self._shard_idx], self._shard_encoding)
        self._on_file_done(self._shard_idx + 1)
        self._shard = None
        self._shard_encoding = None


def _shard_table(buffer: list[_Chunk], schema: pa.Schema, codec: IdCodec) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "time_series_id": codec.encode_list("time_series_id", [c.time_series_id for c in buffer]),
            "spec_type": [c.spec_type for c in buffer],
            "channel": [c.channel for c in buffer],
            "chunk_idx": [c.chunk_idx for c in buffer],
            "n_values": [c.n_values for c in buffer],
            "values": _values_column([c.values for c in buffer]),
            "time_offsets_us": _time_offsets_column([c.time_offsets for c in buffer]),
        },
        schema=schema,
    )


def _plan_chunks(
    ts: TimeSeries, values: pa.Array, chunk_max_bytes: int, time_offsets: pa.Array | None = None
) -> list[_Chunk]:
    """Split a validated series into backend-independent logical chunks.

    Values and time offsets are cut at identical boundaries, which is what lets one chunk locator address
    both: time offset ``k`` is always in the same chunk as value ``k``.

    Args:
        ts: Series metadata used for chunk identity and timing.
        values: Validated float32 values.
        chunk_max_bytes: Maximum uncompressed bytes in one chunk.
        time_offsets: Validated int64 time offsets, one per value, or ``None``.

    Returns:
        Logical chunks in series order.
    """
    max_values = max(1, chunk_max_bytes // _step_bytes(time_offsets is not None))
    chunks: list[_Chunk] = []
    for chunk_idx, start in enumerate(range(0, len(values), max_values)):
        sub = values.slice(start, max_values)
        chunks.append(
            _Chunk(
                time_series_id=ts.time_series_id,
                spec_type=ts.spec.spec_type,
                channel=ts.channel,
                chunk_idx=chunk_idx,
                n_values=len(sub),
                values=sub,
                time_offsets=None if time_offsets is None else time_offsets.slice(start, max_values),
            )
        )
    return chunks


def _time_offsets_column(chunks: list[pa.Array | None]) -> pa.ListArray:
    """Pack per-chunk int64 time offsets into one ``list<int64>`` column, null where a chunk has none.

    A null cell is deliberate: an empty list would be indistinguishable from a zero-length chunk and
    would still cost an offset.

    Args:
        chunks: The per-row time offset arrays, ``None`` for a chunk whose series stores none.

    Returns:
        A ``list<int64>`` array with one row per chunk.
    """
    present = [c for c in chunks if c is not None]
    if not present:
        return pa.nulls(len(chunks), type=pa.list_(pa.int64()))
    offsets = np.zeros(len(chunks) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([0 if c is None else len(c) for c in chunks])
    mask = pa.array([c is None for c in chunks], type=pa.bool_())
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.concat_arrays(present), mask=mask)


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
