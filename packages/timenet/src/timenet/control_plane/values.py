"""Write signal values into the Parquet values plane, and read them back by locator.

Nothing here is new. It calls the same :class:`~timenet.parquet.sharded.RotatingPartWriter`, the same
:mod:`~timenet.parquet.encodings`, and the same :mod:`~timenet.parquet.value_encoding` the current
writer calls, and it does the same self-check on the finished shard. That is the point: the control
plane moves to a database, and the bulk numeric data keeps the byte-budgeted row groups and the
measured per-modality encoding choice exactly as they are.

The only thing that changes is where a chunk's locator is recorded. It used to go into the
``time_series_index`` Parquet table. It now goes into the ``signal_chunks`` table of the database.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset.axis import IrregularAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFValidationError
from timenet.format.constants import (
    DEFAULT_CHUNK_MAX_BYTES,
    DEFAULT_COMPRESSION,
    DEFAULT_ROW_GROUP_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    SHARD_TEMPLATE,
    part_path,
)
from timenet.parquet import encodings
from timenet.parquet.sharded import RotatingPartWriter
from timenet.parquet.value_encoding import ValueEncoding, select_value_encoding


# How many of a modality's arrays to look at when choosing its encoding. The production writer caps
# the sample by bytes off the head of its buffer; this caps it by array count, which is the same
# idea against an in-memory hierarchy.
_ENCODING_SAMPLE_ARRAYS = 64


@dataclass(frozen=True)
class ChunkLocator:
    """Where one chunk of a signal's values landed in the values plane."""

    signal_id: str
    chunk_idx: int
    chunk_file: str
    row_group: int
    row_offset: int
    n_values: int


@dataclass(frozen=True)
class PendingSignal:
    """One signal's values, waiting to be written, with the facts the shard row needs."""

    signal_id: str
    name: str
    spec_type: str
    dtype: str
    values: np.ndarray
    time_offsets_us: np.ndarray | None


def plan_chunks(n_values: int, itemsize: int, chunk_max_bytes: int) -> list[tuple[int, int]]:
    """Split a signal into chunks that each stay under the byte cap.

    Args:
        n_values: How many values the signal has.
        itemsize: Bytes per value.
        chunk_max_bytes: The largest a chunk may be, uncompressed.

    Returns:
        One ``(start, stop)`` pair per chunk, covering the whole signal.
    """
    per_chunk = max(1, chunk_max_bytes // max(1, itemsize))
    return [(start, min(start + per_chunk, n_values)) for start in range(0, max(n_values, 1), per_chunk)][
        : max(1, -(-n_values // per_chunk))
    ]


def shard_schema(dtype: str) -> pa.Schema:
    """Return the Arrow schema of a values shard for one modality.

    Args:
        dtype: The modality's value dtype, as its string name.

    Returns:
        The shard schema. ``values`` is a list of the modality's dtype; ``time_offsets_us`` is null
        for every row of a regular or ordinal series.
    """
    return pa.schema(
        [
            ("signal_id", pa.string()),
            ("chunk_idx", pa.int32()),
            ("spec_type", pa.string()),
            ("signal", pa.string()),
            ("values", pa.list_(pa.type_for_alias(dtype))),
            ("time_offsets_us", pa.list_(pa.int64())),
        ]
    )


class ValuesPlaneWriter:
    """Streams signal values into rotating, byte-budgeted Parquet shards, one stream per modality."""

    def __init__(  # noqa: PLR0913
        self,
        staging_dir: Path,
        *,
        shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
        row_group_target_bytes: int = DEFAULT_ROW_GROUP_TARGET_BYTES,
        chunk_max_bytes: int = DEFAULT_CHUNK_MAX_BYTES,
        compression: str = DEFAULT_COMPRESSION,
        compression_level: int | None = None,
    ) -> None:
        """Bind the writer to its staging directory and byte budgets.

        Args:
            staging_dir: The version's staging directory. Shards go under it.
            shard_target_bytes: Rotate to a new shard once one reaches this size.
            row_group_target_bytes: Flush a row group once buffered chunks reach this size.
            chunk_max_bytes: The largest one logical chunk may be, uncompressed.
            compression: The Parquet codec.
            compression_level: The codec level, used by zstd. ``None`` takes the format's default.
        """
        self._staging_dir = staging_dir
        self._shard_target_bytes = shard_target_bytes
        self._row_group_target_bytes = row_group_target_bytes
        self._chunk_max_bytes = chunk_max_bytes
        self._compression = compression
        self._compression_level = compression_level
        self._parts: list[str] = []
        self._encodings: dict[str, ValueEncoding] = {}

    @property
    def parts(self) -> list[str]:
        """The relative path of every shard written, in order."""
        return self._parts

    @property
    def value_encoding(self) -> dict[str, str]:
        """The encoding chosen for each modality, for the manifest's provenance block."""
        return {spec_type: str(encoding) for spec_type, encoding in self._encodings.items()}

    def write(self, signals: Sequence[PendingSignal]) -> list[ChunkLocator]:
        """Write every signal, grouped by modality, and return where each chunk landed.

        Args:
            signals: The signals to write. They are grouped by ``spec_type`` so one shard holds one
                modality, which keeps a shard's values column homogeneous enough to encode well.

        Returns:
            One :class:`ChunkLocator` per chunk written.
        """
        by_modality: dict[str, list[PendingSignal]] = {}
        for signal in signals:
            by_modality.setdefault(signal.spec_type, []).append(signal)

        locators: list[ChunkLocator] = []
        part_index = 0
        for spec_type in sorted(by_modality):
            modality = by_modality[spec_type]
            written, part_index = self._write_modality(spec_type, modality, part_index)
            locators.extend(written)
        return locators

    def _write_modality(
        self, spec_type: str, signals: list[PendingSignal], part_index: int
    ) -> tuple[list[ChunkLocator], int]:
        """Write one modality's signals into its own rotating shard stream.

        Returns:
            Where every chunk landed, and the next free part index.
        """
        dtype = signals[0].dtype
        sample = [signal.values for signal in signals[:_ENCODING_SAMPLE_ARRAYS]]
        encoding = select_value_encoding(sample, dtype=dtype)
        self._encodings[spec_type] = encoding

        schema = shard_schema(dtype)
        start = part_index
        # ParquetEncoding carries the format's compression-level default, so the values plane keeps
        # exactly the write options the production writer uses.
        options = encodings.ParquetEncoding(
            dictionary_columns=encodings.shard_dictionary(encoding),
            column_encoding=encodings.shard_encoding(encoding),
            compression=self._compression,
            **({} if self._compression_level is None else {"compression_level": self._compression_level}),
        )
        writer = RotatingPartWriter(
            self._staging_dir,
            schema,
            lambda index, start=start: part_path(SHARD_TEMPLATE, start + index),
            part_target_bytes=self._shard_target_bytes,
            parquet_kwargs=encodings.parquet_kwargs(
                dictionary_columns=options.dictionary_columns,
                column_encoding=options.column_encoding,
                compression=options.compression,
                compression_level=options.compression_level,
            ),
        )

        locators: list[ChunkLocator] = []
        buffer: list[dict] = []
        buffered_bytes = 0
        pending: list[tuple[str, int, int]] = []  # signal_id, chunk_idx, n_values

        def flush() -> None:
            nonlocal buffer, buffered_bytes, pending
            if not buffer:
                return
            table = pa.Table.from_pylist(buffer, schema=schema)
            chunk_file, row_group = writer.write(table, table.nbytes)
            for row_offset, (signal_id, chunk_idx, n_values) in enumerate(pending):
                locators.append(
                    ChunkLocator(
                        signal_id=signal_id,
                        chunk_idx=chunk_idx,
                        chunk_file=chunk_file,
                        row_group=row_group,
                        row_offset=row_offset,
                        n_values=n_values,
                    )
                )
            buffer = []
            buffered_bytes = 0
            pending = []

        for signal in signals:
            for chunk_idx, (begin, end) in enumerate(
                plan_chunks(len(signal.values), signal.values.dtype.itemsize, self._chunk_max_bytes)
            ):
                piece = signal.values[begin:end]
                offsets = None if signal.time_offsets_us is None else signal.time_offsets_us[begin:end].tolist()
                buffer.append(
                    {
                        "signal_id": signal.signal_id,
                        "chunk_idx": chunk_idx,
                        "spec_type": signal.spec_type,
                        "signal": signal.name,
                        "values": piece.tolist(),
                        "time_offsets_us": offsets,
                    }
                )
                pending.append((signal.signal_id, chunk_idx, len(piece)))
                buffered_bytes += piece.nbytes
                if buffered_bytes >= self._row_group_target_bytes:
                    flush()
        flush()

        parts = writer.finish()
        self._parts.extend(parts)
        for part in parts:
            self._verify_encoding(part, encoding)
        return locators, start + len(parts)

    def _verify_encoding(self, part: str, encoding: ValueEncoding) -> None:
        """Confirm Parquet applied the encoding that was asked for.

        A wrong column path makes pyarrow drop an encoding request silently, which costs the whole
        benefit with nothing to show for it. The production writer checks this on every shard, so
        this one does too.

        Args:
            part: The shard's version-relative path.
            encoding: The encoding the writer selected for this modality.

        Raises:
            TimeFValidationError: If the finished shard does not carry the requested encoding.
        """
        applied = encodings.values_encoding_of(str(self._staging_dir / part))
        if not encodings.applied_matches(encoding, applied):
            raise TimeFValidationError(f"{part}: asked for {encoding}, Parquet applied {sorted(applied)}")


def read_chunks(root: Path, locators: Sequence[ChunkLocator]) -> np.ndarray:
    """Read one signal's values back by locator, one row group at a time.

    This is the same access pattern the current values reader uses: open the shard, read the one row
    group the locator names, and slice the row inside it. Nothing scans a whole file.

    Args:
        root: The version directory holding the shards.
        locators: The signal's chunk locators, in chunk order.

    Returns:
        The signal's values, concatenated back into one array.
    """
    pieces: list[np.ndarray] = []
    for locator in sorted(locators, key=lambda item: item.chunk_idx):
        parquet_file = pq.ParquetFile(root / locator.chunk_file)
        group = parquet_file.read_row_group(locator.row_group, columns=["values"])
        # Take the list scalar's child array rather than as_py(): a Python list of floats would come
        # back as float64 whatever the column's dtype, silently widening a float32 signal.
        chunk = group.column("values")[locator.row_offset].values
        pieces.append(chunk.to_numpy(zero_copy_only=False))
    return np.concatenate(pieces) if pieces else np.asarray([])


def axis_columns(axis: TimeAxis) -> dict[str, int | None]:
    """Return the ``axes`` row columns for one axis, whatever its shape.

    Args:
        axis: The axis object. Its own type says which shape it is.

    Returns:
        The shape-specific columns. The columns that do not apply to this shape are ``None``.
    """
    columns: dict[str, int | None] = {
        "period_numerator_us": None,
        "period_denominator": None,
        "start_index": None,
        "first_us": None,
        "last_us": None,
    }
    if isinstance(axis, RegularAxis):
        columns["period_numerator_us"] = axis.period_us.numerator
        columns["period_denominator"] = axis.period_us.denominator
        columns["start_index"] = axis.start_index
    elif isinstance(axis, IrregularAxis):
        columns["first_us"] = axis.first_us
        columns["last_us"] = axis.last_us
    return columns


def iter_signal_arrays(signals: Sequence[PendingSignal]) -> Iterator[np.ndarray]:
    """Yield each signal's values array.

    Args:
        signals: The signals to walk.

    Yields:
        Each signal's values.
    """
    for signal in signals:
        yield signal.values
