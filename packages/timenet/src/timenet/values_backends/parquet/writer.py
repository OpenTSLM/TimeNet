"""The default values backend. It streams typed chunks into rotating Parquet shards.

Each shard uses zstd compression and holds one modality. Each shard carries the values encoding
chosen for its ``spec_type`` (see :mod:`timenet.writer.value_encoding`). The backend-neutral
time-series index records each chunk's placement. This record supplies a ``chunk_file`` plus a
:class:`~timenet.values_backends.writer.ChunkDataIndex`. For Parquet, this index gives the
row group (``major_idx``) and the row offset (``minor_idx``).
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.format.constants import SHARD_TEMPLATE, part_path
from timenet.format.schemas import IdCodec, shard_schema
from timenet.types import TimeSeriesSpec
from timenet.values_backends import ValuesBackend
from timenet.values_backends.parquet.config import ParquetValuesConfig
from timenet.values_backends.writer import (
    BaseValuesBackend,
    ChunkDataIndex,
    ChunkPlacement,
    ValuesWriteResult,
)
from timenet.writer import encodings
from timenet.writer.sharded import RotatingPartWriter
from timenet.writer.value_encoding import (
    DICT_MAX_CARDINALITY,
    ENCODING_SAMPLE_SOURCE_BYTES,
    ValueEncoding,
    distinct_bit_patterns,
    encoding_for_cardinality,
    sample_values,
)


MAX_ELEMENTS_PER_ROW_GROUP = 2**31
_BYTES_PER_TIME_OFFSET = 8
_PYARROW_DEFAULT_PAGE_BYTES = 1 << 20
_LOG = logging.getLogger(__name__)


def _bytes_per_value(spec: TimeSeriesSpec, values: pa.Array) -> int:
    """Return the bytes one stored value costs, used to size chunks, row groups, and shards.

    Fixed-width dtypes charge their declared itemsize. A string's width is data-dependent, so its
    buffer bytes per element is the honest budget.

    Args:
        spec: The series' spec, for its dtype.
        values: The series' values, whose buffer size fixes a string's per-value cost.

    Returns:
        The uncompressed bytes one value costs.
    """
    if spec.dtype in {"str", "enum"}:
        return max(1, values.nbytes // max(1, len(values)))
    return np.dtype(spec.dtype).itemsize


def _step_bytes(stores_time_offsets: bool, bytes_per_value: int) -> int:
    """Return the uncompressed bytes that one step of a series costs.

    An irregular series stores an int64 time offset next to each value. So its step costs eight
    bytes more than a regular step. Chunk sizing, row-group flushing, and shard rotation all
    use this unit as their budget. If the code charged every series at the float32 rate, an
    irregular series would overrun each target by three times.

    Args:
        stores_time_offsets: True if the series stores one time offset for each value.
        bytes_per_value: The bytes one value costs for this series' dtype.

    Returns:
        The number of bytes for one step.
    """
    return bytes_per_value + (_BYTES_PER_TIME_OFFSET if stores_time_offsets else 0)


#: Arrow leaf type for ``"enum"`` values. Each shard carries a self-contained dictionary so it stays
#: readable on its own.
_ENUM_LEAF = pa.dictionary(pa.int32(), pa.string())


def _leading_sample_source(buffered: list[pa.Array], budget_bytes: int) -> list[np.ndarray]:
    """Return the leading buffered chunks (as NumPy) that fit within ``budget_bytes``.

    At least one chunk is always returned.
    """
    arrays: list[np.ndarray] = []
    used = 0
    for values in buffered:
        # Drop nulls before sampling: they carry no bit pattern of their own, and leaving them in
        # makes NumPy's distinct-value sort compare None against the real values and fail.
        sampled = values.drop_null() if values.null_count else values
        arrays.append(np.asarray(sampled.to_numpy(zero_copy_only=False)))
        used += values.nbytes
        if used >= budget_bytes:
            break
    return arrays


class ParquetValuesBackend(BaseValuesBackend):
    """The default values backend. It streams series into rotating Parquet shards of typed chunks."""

    name = ValuesBackend.PARQUET

    def _schema_for(self, dtype: str) -> pa.Schema:
        """Return the shard schema whose values element matches a spec dtype.

        Args:
            dtype: The spec's dtype tag.

        Returns:
            The shard Arrow schema with ``values`` as ``list<element>`` for that dtype. A ``"str"``
            dtype stores variable-width text; an ``"enum"`` dtype uses a dictionary leaf.
        """
        if dtype == "str":
            return shard_schema(self._id_types, pa.string())
        if dtype == "enum":
            return shard_schema(self._id_types, _ENUM_LEAF)
        return shard_schema(self._id_types, pa.from_numpy_dtype(np.dtype(dtype)))

    def __init__(self, config: ParquetValuesConfig) -> None:
        """Configure the Parquet values backend.

        Args:
            config: Typed Parquet backend options.
        """
        self._staging_dir = config.staging_dir
        self._codec = config.codec
        self._id_types = config.id_types
        self._shard_target_bytes = config.shard_target_bytes
        self._row_group_target_bytes = config.row_group_target_bytes
        self._chunk_max_bytes = config.chunk_max_bytes
        self._compression = config.compression
        self._compression_level = config.compression_level
        if config.data_page_size is not None:
            self._data_page_size = config.data_page_size
        elif config.row_group_target_bytes > _PYARROW_DEFAULT_PAGE_BYTES:
            self._data_page_size = config.row_group_target_bytes
        else:
            self._data_page_size = None
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
        """Stream all series into rotating shard files and return the chunk placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series' values against its dtype and shape.
            read_time_offsets: Loads an irregular series' int64 time offsets. Returns ``None`` for
                other shapes.
            on_series_done: Progress callback. This method calls it with ``(completed, total)``
                after each series.
            on_file_done: Progress callback. This method calls it with ``(files_finalized)`` after
                each shard closes.

        Returns:
            The chunk placements, the shard files written, and the encoding chosen for each
            ``spec_type``.

        Raises:
            TimeFValidationError: If a spec uses an N-D shape.
        """
        unsupported = [ts.spec.spec_type for ts in unique_series if ts.spec.value_shape]
        if unsupported:
            raise TimeFValidationError(
                "the Parquet values backend currently supports scalar series only; "
                f"use values_backend='zarr' for N-D specs: {sorted(set(unsupported))}"
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

    def encoding_for(self, spec_type: str, dtype: str, buffered: list[pa.Array]) -> ValueEncoding:
        """Return the encoding to use for a modality. Decide it on first sight and log the choice.

        The caller passes the values it buffered for the modality's first row group. So the
        decision costs only a distinct-value count over data already in memory. The backend does
        not read the series a second time and does not call the loader again. The buffer size stays
        within ``row_group_target_bytes``. The backend logs the decision at INFO level under this
        module's logger. So a builder can see, for each ``spec_type``, what encoding the backend
        chose. For the auto path, the log also shows the cardinality that drove the choice.

        Args:
            spec_type: The modality for this decision. The log line names it.
            dtype: The modality's values dtype. It fixes the legal encodings: BYTE_STREAM_SPLIT is
                floats-only.
            buffered: The buffered chunks' values. The backend uses them only the first time a
                modality appears.

        Returns:
            The forced encoding, if the caller set one and it is legal for ``dtype``. Otherwise,
            the encoding chosen from the sample.

        Raises:
            TimeFValidationError: If a forced ``byte_stream_split`` targets a non-float dtype.
        """
        forced = self._forced_encoding
        if dtype == "bool":
            return ValueEncoding.PLAIN
        if dtype == "enum":
            _LOG.info("values encoding for %r: dictionary (enum)", spec_type)
            return ValueEncoding.DICTIONARY
        if forced is not None:
            if forced is ValueEncoding.BYTE_STREAM_SPLIT and not encodings.byte_stream_split_supported(dtype):
                raise TimeFValidationError(
                    f"BYTE_STREAM_SPLIT is not supported for dtype {dtype!r} on spec {spec_type!r}"
                )
            _LOG.info("values encoding for %r: %s (forced)", spec_type, forced.value)
            return forced
        arrays = _leading_sample_source(buffered, ENCODING_SAMPLE_SOURCE_BYTES)
        sample = sample_values(arrays)
        distinct = distinct_bit_patterns(sample)
        encoding = encoding_for_cardinality(distinct, dtype=dtype)
        _LOG.info(
            "values encoding for %r: %s (auto; %d distinct in %d sampled values, dictionary up to %d)",
            spec_type,
            encoding.value,
            distinct,
            sample.size,
            DICT_MAX_CARDINALITY,
        )
        return encoding

    def _shard_parquet_kwargs(self, value_encoding: ValueEncoding) -> dict:
        return encodings.parquet_kwargs(
            dictionary_columns=encodings.shard_dictionary(value_encoding),
            column_encoding=encodings.shard_encoding(value_encoding),
            compression=self._compression,
            compression_level=self._compression_level,
            data_page_size=self._data_page_size,
        )

    def _verify_values_encoding(self, rel_path: str, value_encoding: ValueEncoding) -> None:
        """Verify a finalized shard's values column carries the encoding chosen for it.

        pyarrow can silently drop a column encoding when the column path does not match. Without
        this check, the size on disk can differ from the size that the choice expected. The code
        allows Parquet's own dictionary-to-plain fallback. This fallback happens when a dictionary
        outgrows its page limit. The fallback is lossless, and the writer does not choose it.

        Args:
            rel_path: The finalized shard's path, relative to the staging directory.
            value_encoding: The encoding used to open the shard.

        Raises:
            TimeFValidationError: If the values column does not carry the chosen encoding, or a
                shard that carries time offsets does not encode them as DELTA_BINARY_PACKED.
        """
        path = self._staging_dir / rel_path
        applied = encodings.values_encoding_of(str(path))
        if not applied:  # empty shard (no values written yet), so there is nothing to verify
            return
        if not encodings.applied_matches(value_encoding, applied):
            raise TimeFValidationError(
                f"expected {value_encoding} on the shard values column of {rel_path}, got {sorted(applied)}"
            )
        if not self._time_offsets_checked:
            time_offsets = encodings.values_encoding_of(str(path), encodings.TIME_OFFSETS_COLUMN)
            # An all-regular shard has only RLE definition levels here. It has no time offsets to
            # encode. So this case proves nothing, and the check stays active for a later shard.
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
    signal: str
    chunk_idx: int
    n_values: int
    dtype: str
    bytes_per_value: int
    values: pa.Array
    time_offsets: pa.Array | None = None
    """This chunk's int64 time offsets, sliced at the same boundary as its values, or ``None``."""

    @property
    def n_bytes(self) -> int:
        """Uncompressed bytes this chunk contributes to the row-group and shard budgets."""
        return self.n_values * _step_bytes(self.time_offsets is not None, self.bytes_per_value)


class _ShardStream:
    """Buffers chunks into row groups and streams each modality through its own rotating-part writer.

    Each shard holds only one modality. The stream receives chunks sorted by ``spec_type``. It
    starts a fresh :class:`RotatingPartWriter` each time the modality changes. So each modality's
    shards carry the values encoding chosen for it. Part numbers stay globally unique because each
    modality's writer starts counting after the shards already written.
    """

    def __init__(
        self,
        backend: ParquetValuesBackend,
        row_group_target_bytes: int,
        shard_target_bytes: int,
        on_file_done: Callable[[int], None],
    ) -> None:
        """Bind the stream to its backend and to the byte targets that trigger a flush or rotation.

        Args:
            backend: The owning backend. The stream uses it for the shard schema, the codec, the
                encoding choice, and the self-check.
            row_group_target_bytes: If buffered values pass this number of bytes, the stream flushes
                a row group.
            shard_target_bytes: If a shard's written values pass this number of bytes, the stream
                rotates to a new shard.
            on_file_done: Progress callback. The stream calls it with the count of finalized shards.
        """
        self._backend = backend
        self._row_group_target_bytes = row_group_target_bytes
        self._shard_target_bytes = shard_target_bytes
        self._on_file_done = on_file_done
        self._core: RotatingPartWriter | None = None
        self._schema: pa.Schema | None = None
        self._spec_type: str | None = None
        self._shard_base = 0
        self._buffer: list[_Chunk] = []
        self._buffer_bytes = 0
        self._shard_paths: list[str] = []
        self.placements: dict[tuple[str, int], ChunkPlacement] = {}
        self.encodings: dict[str, ValueEncoding] = {}
        """The encoding chosen for each ``spec_type``, in first-seen order."""

    @property
    def shard_paths(self) -> list[str]:
        """The shard part paths written so far, in order."""
        return self._shard_paths

    def add(self, chunk: _Chunk) -> None:
        """If the modality changes, close the current modality's shards. Then buffer the chunk."""
        if self._spec_type is not None and chunk.spec_type != self._spec_type:
            self._flush()
            self._close_modality()
        self._spec_type = chunk.spec_type
        self._buffer.append(chunk)
        self._buffer_bytes += chunk.n_bytes
        if self._buffer_bytes >= self._row_group_target_bytes:
            self._flush()

    def finish(self) -> None:
        """Flush any remaining buffered chunks and close the final shard."""
        self._flush()
        self._close_modality()

    def _open_modality(self) -> RotatingPartWriter:
        """Open a rotating-part writer for the buffered modality. Choose its encoding on first sight.

        Returns:
            The rotating-part writer bound to this modality's encoding, schema, and shard-index offset.
        """
        spec_type = self._buffer[0].spec_type
        dtype = self._buffer[0].dtype
        if spec_type not in self.encodings:
            self.encodings[spec_type] = self._backend.encoding_for(
                spec_type, dtype, [chunk.values for chunk in self._buffer]
            )
        encoding = self.encodings[spec_type]
        base = self._shard_base

        def on_part_closed(rel_path: str, parts_written: int) -> None:
            self._backend._verify_values_encoding(rel_path, encoding)
            self._on_file_done(base + parts_written)

        self._schema = self._backend._schema_for(dtype)
        self._core = RotatingPartWriter(
            self._backend._staging_dir,
            self._schema,
            lambda index: part_path(SHARD_TEMPLATE, base + index),
            part_target_bytes=self._shard_target_bytes,
            parquet_kwargs=self._backend._shard_parquet_kwargs(encoding),
            on_part_closed=on_part_closed,
        )
        return self._core

    def _close_modality(self) -> None:
        """Close the current modality's writer. Advance the global shard index past its parts."""
        if self._core is None:
            return
        self._core.finish()
        self._shard_paths.extend(self._core.parts)
        self._shard_base += len(self._core.parts)
        self._core = None
        self._schema = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        if sum(chunk.n_values for chunk in self._buffer) >= MAX_ELEMENTS_PER_ROW_GROUP:
            raise TimeFValidationError("row group would exceed the 2^31 element limit")
        core = self._core if self._core is not None else self._open_modality()
        table = _shard_table(self._buffer, self._schema, self._backend._codec)
        shard_path, row_group = core.write(table, self._buffer_bytes)
        for offset, chunk in enumerate(self._buffer):
            self.placements[chunk.time_series_id, chunk.chunk_idx] = ChunkPlacement(
                chunk_file=shard_path,
                data_index=ChunkDataIndex(major_idx=row_group, minor_idx=offset),
                spec_type=chunk.spec_type,
                signal=chunk.signal,
                n_values=chunk.n_values,
            )
        self._buffer = []
        self._buffer_bytes = 0


def _shard_table(buffer: list[_Chunk], schema: pa.Schema, codec: IdCodec) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "time_series_id": codec.encode_list("time_series_id", [c.time_series_id for c in buffer]),
            "spec_type": [c.spec_type for c in buffer],
            "signal": [c.signal for c in buffer],
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

    The code cuts values and time offsets at the same boundaries. This lets one chunk locator
    address both. So time offset ``k`` is always in the same chunk as value ``k``.

    Args:
        ts: Series metadata used for chunk identity and timing.
        values: Validated values in the spec's canonical Arrow type.
        chunk_max_bytes: Maximum uncompressed bytes in one chunk.
        time_offsets: Validated int64 time offsets, one per value, or ``None``.

    Returns:
        Logical chunks in series order.
    """
    bytes_per_value = _bytes_per_value(ts.spec, values)
    max_values = max(1, chunk_max_bytes // _step_bytes(time_offsets is not None, bytes_per_value))
    chunks: list[_Chunk] = []
    for chunk_idx, start in enumerate(range(0, len(values), max_values)):
        sub = values.slice(start, max_values)
        if ts.spec.dtype == "enum":
            sub = sub.cast(_ENUM_LEAF)
        chunks.append(
            _Chunk(
                time_series_id=ts.time_series_id,
                spec_type=ts.spec.spec_type,
                signal=ts.signal,
                chunk_idx=chunk_idx,
                n_values=len(sub),
                dtype=ts.spec.dtype,
                bytes_per_value=bytes_per_value,
                values=sub,
                time_offsets=None if time_offsets is None else time_offsets.slice(start, max_values),
            )
        )
    return chunks


def _time_offsets_column(chunks: list[pa.Array | None]) -> pa.ListArray:
    """Pack per-chunk int64 time offsets into one ``list<int64>`` column. Use null where a chunk has none.

    The null cell is deliberate. An empty list looks identical to a zero-length chunk, and it still
    costs one offset.

    Args:
        chunks: The per-row time offset arrays. ``None`` marks a chunk whose series stores no offsets.

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
    """Pack per-chunk arrays into one ``list<element>`` column without boxing to Python values.

    Args:
        chunks: The per-row element arrays, one per chunk in the row group. Each array has the
            modality's value type.

    Returns:
        A ``list<element>`` array with one row per chunk.
    """
    offsets = np.zeros(len(chunks) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(chunk) for chunk in chunks])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.concat_arrays(chunks))
