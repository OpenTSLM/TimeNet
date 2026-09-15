"""Write signal values into the Parquet values plane, read them back by locator, and pick a backend.

The Parquet path is unchanged. It calls the same :class:`~timenet.parquet.sharded.RotatingPartWriter`,
the same :mod:`~timenet.parquet.encodings`, and the same :mod:`~timenet.parquet.value_encoding` the
current writer calls, and it does the same self-check on the finished shard. The control plane moves
to a database, and the bulk numeric data keeps the byte-budgeted row groups and the measured
per-modality encoding choice exactly as they are.

Two things are new. A chunk's locator is recorded in the ``signal_chunks`` table of the database
rather than in the ``time_series_index`` Parquet table, and the locator no longer speaks Parquet:
``chunk_major_idx`` and ``chunk_minor_idx`` mean whatever the backend that wrote the artifact says
they mean. Parquet reads them as a row group and a row inside it. Zarr reads the major index as an
element offset and leaves the minor index unset. :data:`VALUES_BACKENDS` names the backends, and
:func:`values_writer` and :func:`read_values` are the seam both sides go through.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset.axis import IrregularAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
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

PARQUET: Final = "parquet"
"""The default values backend: byte-budgeted Parquet shards with per-modality encodings."""

ZARR: Final = "zarr"
"""The Zarr values backend: one chunked typed array per modality."""

VALUES_BACKENDS: Final = (PARQUET, ZARR)
"""Every backend a writer may target and a reader can resolve."""


@dataclass(frozen=True)
class ChunkLocator:
    """Where one chunk of a signal's values landed, in terms its backend interprets.

    ``signal_id`` is the integer surrogate the control-plane writer assigned to the signal.

    ``chunk_file`` names an artifact: a Parquet shard, or a Zarr array inside the store. It stays a
    path here. The values plane knows nothing about ``values_artifacts``, so the control-plane
    writer is what turns the path into an ``artifact_id``. The two indexes address a run of values
    inside the artifact. Parquet uses both (row group, then row). Zarr uses only the major one (the
    element offset along the array) and leaves ``chunk_minor_idx`` ``None``.
    """

    signal_id: int
    chunk_idx: int
    chunk_file: str
    chunk_major_idx: int
    chunk_minor_idx: int | None
    n_values: int


@dataclass(frozen=True)
class ChunkSlice:
    """One chunk a step window crosses, and the run of that chunk the window asks for.

    ``start`` and ``stop`` are offsets inside the chunk, not steps of the signal. A window that
    opens partway through chunk 4 and closes inside chunk 5 plans two slices: the first starts where
    the window opens, the second starts at 0.
    """

    locator: ChunkLocator
    start: int
    stop: int
    request: int = 0
    """Which of the caller's windows asked for this run.

    A batch of items cut from one recording asks for many windows of the same signal, and the point
    of reading them together is that they share a row group and so cost one decode between them. A
    result keyed by signal could hold only one of them, so the key names the window instead.
    """


@dataclass(frozen=True)
class PendingSignal:
    """One signal's values, waiting to be written, with the facts the shard row needs.

    ``signal_id`` is the integer surrogate the control-plane writer assigned to the signal, not the
    caller's own id. It is what a shard row carries and what a locator comes back with.
    """

    signal_id: int
    name: str
    spec_type: str
    dtype: str
    values: np.ndarray
    time_offsets_us: np.ndarray | None


class ValuesWriter(Protocol):
    """What the control-plane writer needs from a values backend.

    A backend takes signals one at a time and hands back locators as they reach disk, names the
    artifacts it wrote so the write-time validation can check the chunk rows against them, and lists
    the files it produced so the manifest can checksum them.
    """

    @property
    def parts(self) -> list[str]:
        """Every file written, as version-relative paths, for the manifest."""
        ...

    @property
    def artifacts(self) -> list[tuple[str, str]]:
        """Every ``(chunk_file, backend)`` a locator may name, for ``values_artifacts``."""
        ...

    @property
    def value_encoding(self) -> dict[str, str]:
        """The encoding chosen per modality, or empty for a backend that chooses none."""
        ...

    def add(self, signal: "PendingSignal") -> list[ChunkLocator]:
        """Take one signal and return the locators of any chunks that reached disk."""
        ...

    def finish(self) -> list[ChunkLocator]:
        """Flush everything still buffered and return its locators."""
        ...


def values_writer(backend: str, staging_dir: Path, **options: Any) -> ValuesWriter:
    """Open the values-plane writer for one backend.

    Args:
        backend: One of :data:`VALUES_BACKENDS`.
        staging_dir: The version's staging directory.
        options: Byte budgets and codec settings, forwarded to the backend.

    Returns:
        The writer.

    Raises:
        TimeFValidationError: If ``backend`` is not a known values backend.
    """
    if backend == PARQUET:
        return ValuesPlaneWriter(staging_dir, **options)
    if backend == ZARR:
        # Imported here because zarr is an optional extra; the Parquet core never needs it.
        from timenet.control_plane.zarr_values import ZarrValuesPlaneWriter  # noqa: PLC0415

        return ZarrValuesPlaneWriter(staging_dir, **options)
    raise TimeFValidationError(f"unknown values backend {backend!r}; known: {', '.join(VALUES_BACKENDS)}")


def read_values(root: Path, locators: Sequence[ChunkLocator], backend_of: Mapping[str, str]) -> np.ndarray:
    """Read one signal's values, sending each locator to the backend that wrote its artifact.

    Args:
        root: The version directory.
        locators: The signal's chunk locators, in any order.
        backend_of: Which backend wrote each artifact, from ``values_artifacts``.

    Returns:
        The signal's values, concatenated back into one array.

    Raises:
        TimeFFormatError: If a locator names an artifact no backend declared.
    """  # noqa: DOC502 - raised by _runs_by_backend
    pieces: list[np.ndarray] = []
    for backend, run in _runs_by_backend(sorted(locators, key=lambda item: item.chunk_idx), backend_of):
        if backend == ZARR:
            from timenet.control_plane.zarr_values import read_chunks as read_zarr_chunks  # noqa: PLC0415

            pieces.append(read_zarr_chunks(root, run))
        else:
            pieces.append(read_chunks(root, run))
    return np.concatenate(pieces) if pieces else np.asarray([])


def read_many_values(
    root: Path, locators: Sequence[ChunkLocator], backend_of: Mapping[str, str]
) -> dict[int, np.ndarray]:
    """Read many signals' values in one pass, sending each artifact to the backend that wrote it.

    Args:
        root: The version directory.
        locators: Chunk locators for any number of signals, in any order.
        backend_of: Which backend wrote each artifact, from ``values_artifacts``.

    Returns:
        Each signal's values, keyed by signal id.

    Raises:
        TimeFFormatError: If a locator names an artifact no backend declared.
    """
    grouped: dict[str, list[ChunkLocator]] = {}
    for locator in locators:
        backend = backend_of.get(locator.chunk_file)
        if backend is None:
            raise TimeFFormatError(f"no backend declared for values artifact {locator.chunk_file!r}")
        grouped.setdefault(backend, []).append(locator)

    found: dict[int, np.ndarray] = {}
    for backend, run in grouped.items():
        if backend == ZARR:
            # Zarr addresses an element offset along one array, so there is no row group to share.
            # Each signal is still one addressed read; only the Parquet path gains from grouping.
            from timenet.control_plane.zarr_values import read_chunks as read_zarr_chunks  # noqa: PLC0415

            by_signal: dict[int, list[ChunkLocator]] = {}
            for locator in run:
                by_signal.setdefault(locator.signal_id, []).append(locator)
            for signal_id, signal_locators in by_signal.items():
                found[signal_id] = read_zarr_chunks(root, sorted(signal_locators, key=lambda it: it.chunk_idx))
        else:
            found.update(read_many_chunks(root, run))
    return found


def plan_window(locators: Sequence[ChunkLocator], start: int, stop: int, request: int = 0) -> list[ChunkSlice]:
    """Work out which chunks a step window crosses, without reading any values.

    A chunk row states how many values its chunk holds, so a signal's chunk boundaries are known
    from the control plane alone: chunk k covers the steps ``[the lengths before it, plus its own)``.
    A window is planned by that arithmetic and only the chunks it lands in are ever read, which is
    the whole point of a window read. Reading the signal to find its boundaries would cost the thing
    the window is avoiding.

    A ``stop`` past the signal's last step is clamped to it, and a window that starts past the end
    plans nothing. A scored window can overrun the signals it scores, and a corpus keeps those
    scorings, so a clamp delivers the steps that exist instead of refusing the read.

    Args:
        locators: The signal's chunk locators, in any order.
        start: The window's first step, inclusive.
        stop: The window's last step, exclusive.
        request: Which of the caller's windows this is, carried onto every slice it plans so the
            read can key the result by window rather than by signal.

    Returns:
        One slice per chunk the window crosses, in chunk order. An empty window plans nothing.

    Raises:
        TimeFValidationError: If the window runs backwards or starts before the first step.
    """
    if start < 0 or stop < start:
        raise TimeFValidationError(f"expected 0 <= start <= stop, got start={start}, stop={stop}")
    ordered = sorted(locators, key=lambda item: item.chunk_idx)
    bounded = min(stop, sum(item.n_values for item in ordered))
    if start >= bounded:
        return []
    planned: list[ChunkSlice] = []
    cursor = 0
    for locator in ordered:
        chunk_stop = cursor + locator.n_values
        if chunk_stop > start:
            planned.append(
                ChunkSlice(
                    locator=locator,
                    start=max(start - cursor, 0),
                    stop=min(bounded - cursor, locator.n_values),
                    request=request,
                )
            )
        cursor = chunk_stop
        if cursor >= bounded:
            break
    return planned


def read_many_windows(
    root: Path,
    locators: Sequence[ChunkLocator],
    windows: Sequence[tuple[int, int, int]],
    backend_of: Mapping[str, str],
) -> dict[int, np.ndarray]:
    """Read many step windows in one pass, touching only the chunks those windows cross.

    Several windows may name the same signal. That is the case this is for: a batch of items cut
    from one recording asks for one window per item per signal, and those windows sit in the same
    row groups, so read together they cost one decode between them instead of one each.

    Args:
        root: The version directory.
        locators: Chunk locators for any number of signals, in any order.
        windows: One ``(signal_id, start, stop)`` per window, with ``stop`` exclusive.
        backend_of: Which backend wrote each artifact, from ``values_artifacts``.

    Returns:
        Each window's values, keyed by its position in ``windows``. A window covering no value is
        left out: nothing was read, so nothing here knows which dtype an empty array should carry.
        The reader fills those in from the control plane, which does know.

    Raises:
        TimeFFormatError: If a locator names an artifact no backend declared.
    """
    by_signal: dict[int, list[ChunkLocator]] = {}
    for locator in locators:
        by_signal.setdefault(locator.signal_id, []).append(locator)

    planned: dict[str, list[ChunkSlice]] = {}
    for request, (signal_id, start, stop) in enumerate(windows):
        for piece in plan_window(by_signal.get(signal_id, []), start, stop, request):
            backend = backend_of.get(piece.locator.chunk_file)
            if backend is None:
                raise TimeFFormatError(f"no backend declared for values artifact {piece.locator.chunk_file!r}")
            planned.setdefault(backend, []).append(piece)

    found: dict[int, np.ndarray] = {}
    for backend, pieces in planned.items():
        if backend == ZARR:
            from timenet.control_plane.zarr_values import read_chunk_slices as read_zarr_slices  # noqa: PLC0415

            found.update(read_zarr_slices(root, pieces))
        else:
            found.update(read_many_chunk_slices(root, pieces))
    return found


def _runs_by_backend(
    locators: Sequence[ChunkLocator], backend_of: Mapping[str, str]
) -> list[tuple[str, list[ChunkLocator]]]:
    """Split a signal's locators into consecutive runs that share a backend.

    Args:
        locators: The locators, in chunk order.
        backend_of: Which backend wrote each artifact.

    Returns:
        One ``(backend, locators)`` pair per run. A signal normally yields exactly one.

    Raises:
        TimeFFormatError: If a locator names an artifact no backend declared.
    """
    runs: list[tuple[str, list[ChunkLocator]]] = []
    for locator in locators:
        backend = backend_of.get(locator.chunk_file)
        if backend is None:
            raise TimeFFormatError(f"no backend declared for values artifact {locator.chunk_file!r}")
        if runs and runs[-1][0] == backend:
            runs[-1][1].append(locator)
        else:
            runs.append((backend, [locator]))
    return runs


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
        The shard schema. ``signal_id`` is the control plane's integer surrogate; ``values`` is a
        list of the modality's dtype; ``time_offsets_us`` is null for every row of a regular or
        ordinal series.
    """
    # An enum value is one of its spec's category labels, so the shard stores the label and needs no
    # codebook of its own. Arrow has no alias of that name, and the writer dictionary-encodes a
    # low-cardinality column anyway, so a label costs one copy per shard however often it repeats.
    value_type = pa.string() if dtype == "enum" else pa.type_for_alias(dtype)
    return pa.schema(
        [
            # Matches the control plane's surrogate id width, so a corpus the control plane can
            # address cannot overflow the column naming its values.
            ("signal_id", pa.uint32()),
            ("chunk_idx", pa.int32()),
            ("spec_type", pa.string()),
            ("signal", pa.string()),
            ("values", pa.list_(value_type)),
            ("time_offsets_us", pa.list_(pa.int64())),
        ]
    )


class ValuesPlaneWriter:
    """Streams signal values into rotating, byte-budgeted Parquet shards, one stream per modality.

    Signals arrive one at a time, so a build never holds the whole values plane in memory. Each
    modality keeps its own open shard stream, opened the first time that modality is seen, and every
    stream draws from one part counter so the shard numbers stay unique across modalities.

    A modality's encoding is chosen from the first signals that arrive on it, since the choice has to
    be made before the first row group is written and cannot be revised afterwards.
    """

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
        self._streams: dict[str, _ModalityStream] = {}
        self._next_part = 0

    @property
    def parts(self) -> list[str]:
        """The relative path of every shard written, in order."""
        return self._parts

    @property
    def artifacts(self) -> list[tuple[str, str]]:
        """Every shard a locator may name, tagged with this backend.

        Returns:
            One ``(chunk_file, 'parquet')`` pair per shard. A Parquet shard is one file, so the
            artifact list and the file list happen to coincide here; they do not for Zarr.
        """
        return [(part, PARQUET) for part in self._parts]

    @property
    def value_encoding(self) -> dict[str, str]:
        """The encoding chosen for each modality, for the manifest's provenance block."""
        return {spec_type: str(encoding) for spec_type, encoding in self._encodings.items()}

    def _claim_part(self) -> str:
        """Return the next shard path, so every modality's shards share one numbering.

        Returns:
            The new shard's version-relative path.
        """
        path = part_path(SHARD_TEMPLATE, self._next_part)
        self._next_part += 1
        self._parts.append(path)
        return path

    def add(self, signal: PendingSignal) -> list[ChunkLocator]:
        """Buffer one signal's values, writing a row group if that fills one.

        Args:
            signal: The signal to write.

        Returns:
            The locators of any chunks that reached disk because of this signal. Usually empty: a
            locator is only known once the row group holding it has been written.
        """
        stream = self._streams.get(signal.spec_type)
        if stream is None:
            stream = _ModalityStream(self, signal.spec_type, signal.dtype)
            self._streams[signal.spec_type] = stream
        return stream.add(signal)

    def write(self, signals: Sequence[PendingSignal]) -> list[ChunkLocator]:
        """Write every signal and return where each chunk landed.

        Args:
            signals: The signals to write.

        Returns:
            One :class:`ChunkLocator` per chunk written.
        """
        locators: list[ChunkLocator] = []
        for signal in signals:
            locators.extend(self.add(signal))
        locators.extend(self.finish())
        return locators

    def finish(self) -> list[ChunkLocator]:
        """Flush every open modality stream and close its shards.

        Returns:
            The locators of every chunk still buffered when this was called.
        """
        locators: list[ChunkLocator] = []
        for stream in self._streams.values():
            locators.extend(stream.finish())
        self._streams.clear()
        return locators


def _list_array(pieces: Sequence[np.ndarray | None], value_type: pa.DataType) -> pa.Array:
    """Build a list column from typed buffers, without going through Python objects.

    Each piece becomes one list element and a ``None`` piece becomes a null element. The values are
    concatenated once and handed to Arrow as a single buffer, so a float stays a float the whole way
    rather than being unpacked into Python floats and repacked. Measured on 4,000 arrays of 1,000
    float32: 1.4 ms here against 114.6 ms for the round trip through ``tolist``, for byte-identical
    output.

    Args:
        pieces: One array per row, or ``None`` where the row has no list.
        value_type: The Arrow type of the list's elements.

    Returns:
        The list array.
    """
    present = [piece for piece in pieces if piece is not None]
    lengths = [0 if piece is None else len(piece) for piece in pieces]
    bounds = np.zeros(len(pieces) + 1, dtype=np.int32)
    np.cumsum(lengths, out=bounds[1:])
    flat = np.concatenate(present) if present else np.empty(0, dtype=value_type.to_pandas_dtype())
    # A null in the offsets marks that element null. The trailing bound is never itself an element.
    missing = np.array([piece is None for piece in pieces] + [False])
    offsets = pa.array(bounds, type=pa.int32(), mask=missing)
    return pa.ListArray.from_arrays(offsets, pa.array(flat, type=value_type))


class _ModalityStream:
    """One modality's open shard stream: its buffer, its chosen encoding, and its rotating writer."""

    def __init__(self, owner: ValuesPlaneWriter, spec_type: str, dtype: str) -> None:
        self._owner = owner
        self._spec_type = spec_type
        self._dtype = dtype
        self._schema = shard_schema(dtype)
        self._writer: RotatingPartWriter | None = None
        self._sample: list[np.ndarray] = []
        # One list per column rather than a list of row dicts. A row dict would have to hold the
        # values as Python objects, and converting a typed buffer into Python floats and back into
        # Arrow costs 82x what building the Arrow array from the buffer directly costs.
        self._signal_ids: list[int] = []
        self._chunk_idxs: list[int] = []
        self._names: list[str] = []
        self._values: list[np.ndarray] = []
        self._offsets: list[np.ndarray | None] = []
        self._pending: list[tuple[int, int, int]] = []
        self._buffered_bytes = 0

    def _open(self) -> RotatingPartWriter:
        """Choose this modality's encoding from what has arrived, and open its shard stream.

        Returns:
            The rotating writer this modality's row groups go through.
        """
        encoding = select_value_encoding(self._sample, dtype=self._dtype)
        self._owner._encodings[self._spec_type] = encoding
        options = encodings.ParquetEncoding(
            dictionary_columns=encodings.shard_dictionary(encoding),
            column_encoding=encodings.shard_encoding(encoding),
            compression=self._owner._compression,
            **({} if self._owner._compression_level is None else {"compression_level": self._owner._compression_level}),
        )
        self._writer = RotatingPartWriter(
            self._owner._staging_dir,
            self._schema,
            lambda _index: self._owner._claim_part(),
            part_target_bytes=self._owner._shard_target_bytes,
            parquet_kwargs=encodings.parquet_kwargs(
                dictionary_columns=options.dictionary_columns,
                column_encoding=options.column_encoding,
                compression=options.compression,
                compression_level=options.compression_level,
            ),
        )
        return self._writer

    def add(self, signal: PendingSignal) -> list[ChunkLocator]:
        """Buffer one signal, flushing a row group each time the buffer fills.

        The budget is checked per chunk rather than per signal, because a row group is the unit a
        reader decodes. Checked once per signal, a signal longer than the budget lands in one row
        group of its own whatever the budget says: an 11 MB channel writes an 11 MB row group, and
        then a reader that wants 30 seconds of it decodes the whole channel.

        Returns:
            The locators of any chunks this call pushed to disk.
        """
        if self._writer is None and len(self._sample) < _ENCODING_SAMPLE_ARRAYS:
            self._sample.append(signal.values)
        locators: list[ChunkLocator] = []
        for chunk_idx, (begin, end) in enumerate(
            plan_chunks(len(signal.values), signal.values.dtype.itemsize, self._owner._chunk_max_bytes)
        ):
            piece = signal.values[begin:end]
            self._signal_ids.append(signal.signal_id)
            self._chunk_idxs.append(chunk_idx)
            self._names.append(signal.name)
            self._values.append(piece)
            self._offsets.append(None if signal.time_offsets_us is None else signal.time_offsets_us[begin:end])
            self._pending.append((signal.signal_id, chunk_idx, len(piece)))
            self._buffered_bytes += piece.nbytes
            if self._buffered_bytes >= self._owner._row_group_target_bytes:
                locators.extend(self._flush())
        return locators

    def _flush(self) -> list[ChunkLocator]:
        """Write the buffered chunks as one row group and return where they landed.

        Returns:
            One locator per chunk in the row group.
        """
        if not self._values:
            return []
        writer = self._writer if self._writer is not None else self._open()
        rows = len(self._values)
        table = pa.Table.from_arrays(
            [
                pa.array(np.asarray(self._signal_ids, dtype=np.uint32), pa.uint32()),
                pa.array(np.asarray(self._chunk_idxs, dtype=np.int32), pa.int32()),
                pa.array([self._spec_type] * rows, pa.string()),
                pa.array(self._names, pa.string()),
                _list_array(self._values, self._schema.field("values").type.value_type),
                _list_array(self._offsets, pa.int64()),
            ],
            schema=self._schema,
        )
        chunk_file, row_group = writer.write(table, table.nbytes)
        locators = [
            ChunkLocator(
                signal_id=signal_id,
                chunk_idx=chunk_idx,
                chunk_file=chunk_file,
                chunk_major_idx=row_group,
                chunk_minor_idx=row_offset,
                n_values=n_values,
            )
            for row_offset, (signal_id, chunk_idx, n_values) in enumerate(self._pending)
        ]
        self._signal_ids = []
        self._chunk_idxs = []
        self._names = []
        self._values = []
        self._offsets = []
        self._pending = []
        self._buffered_bytes = 0
        return locators

    def finish(self) -> list[ChunkLocator]:
        """Flush what is left, close the shards, and check the encoding Parquet actually applied.

        Returns:
            The locators of the chunks still buffered when this was called.

        Raises:
            TimeFValidationError: If a finished shard does not carry the requested encoding.
        """
        locators = self._flush()
        if self._writer is None:
            return locators
        for part in self._writer.finish():
            applied = encodings.values_encoding_of(str(self._owner._staging_dir / part))
            encoding = self._owner._encodings[self._spec_type]
            if not encodings.applied_matches(encoding, applied):
                raise TimeFValidationError(f"{part}: asked for {encoding}, Parquet applied {sorted(applied)}")
        return locators


def read_many_chunks(root: Path, locators: Sequence[ChunkLocator]) -> dict[int, np.ndarray]:
    """Read many signals' whole chunks, touching each row group once however many signals it holds.

    A whole chunk is the run ``[0, n_values)`` of itself, so this is :func:`read_many_chunk_slices`
    with nothing cut off either end. The only difference is the key: a whole-signal read wants its
    result keyed by signal, so each locator names its own signal as the request.

    Args:
        root: The version directory holding the shards.
        locators: Chunk locators for any number of signals, in any order.

    Returns:
        Each signal's values, keyed by signal id and concatenated back in chunk order.

    Raises:
        TimeFFormatError: If a locator carries no row offset, which no Parquet shard can address.
    """  # noqa: DOC502 - raised by read_many_chunk_slices
    return read_many_chunk_slices(
        root,
        [ChunkSlice(locator, 0, locator.n_values, locator.signal_id) for locator in locators],
    )


def read_many_chunk_slices(root: Path, slices: Sequence[ChunkSlice]) -> dict[int, np.ndarray]:
    """Read the planned run of each chunk a set of windows crosses, decoding each row group once.

    Reading run by run re-opens the shard and re-decodes the row group every time, which is the
    wrong shape as soon as a caller wants a batch: a series shared by several records is read once
    per use, and neighbouring signals almost always sit in the row group just decoded. Grouping by
    ``(artifact, row group)`` first turns that into one read per row group.

    A request answered by one run is copied out of the decoded group. Arrow's slice is a view of the
    group's own buffer, and keeping a 12 KB window that way would hold a 4 MB row group alive behind
    it. The copy is also what makes the result writable, so a caller can scale or fill the array it
    gets back. A request spanning several runs is not copied run by run: the concatenate below
    already allocates an array of its own, and copying first would allocate the whole result twice.

    Args:
        root: The version directory holding the shards.
        slices: The planned chunk runs, for any number of windows, in any order.

    Returns:
        Each window's values, keyed by its :attr:`ChunkSlice.request` and concatenated back in chunk
        order.

    Raises:
        TimeFFormatError: If a locator carries no row offset, which no Parquet shard can address.
    """
    by_group: dict[tuple[str, int], list[ChunkSlice]] = {}
    runs_of: Counter[int] = Counter()
    for piece in slices:
        if piece.locator.chunk_minor_idx is None:
            raise TimeFFormatError(f"{piece.locator.chunk_file}: a Parquet locator needs a row offset, got none")
        by_group.setdefault((piece.locator.chunk_file, piece.locator.chunk_major_idx), []).append(piece)
        runs_of[piece.request] += 1

    pieces: dict[int, list[tuple[int, np.ndarray]]] = {}
    open_files: dict[str, pq.ParquetFile] = {}
    for (chunk_file, row_group), group_slices in sorted(by_group.items()):
        parquet_file = open_files.get(chunk_file)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(root / chunk_file)
            open_files[chunk_file] = parquet_file
        column = parquet_file.read_row_group(row_group, columns=["values"]).column("values")
        for piece in group_slices:
            # The list scalar's child array, not as_py(): a Python list of floats would come back as
            # float64 whatever the column's dtype, silently widening a float32 signal.
            chunk = column[piece.locator.chunk_minor_idx].values.slice(piece.start, piece.stop - piece.start)
            run = chunk.to_numpy(zero_copy_only=False)
            pieces.setdefault(piece.request, []).append(
                (piece.locator.chunk_idx, run.copy() if runs_of[piece.request] == 1 else run)
            )
    return {
        request: np.concatenate([chunk for _, chunk in sorted(parts)]) if len(parts) > 1 else parts[0][1]
        for request, parts in pieces.items()
    }


def read_chunks(root: Path, locators: Sequence[ChunkLocator]) -> np.ndarray:
    """Read one signal's values back by locator, one row group at a time.

    This is the same access pattern the current values reader uses: open the shard, read the one row
    group ``chunk_major_idx`` names, and slice row ``chunk_minor_idx`` inside it. Nothing scans a
    whole file.

    Args:
        root: The version directory holding the shards.
        locators: The signal's chunk locators, in chunk order.

    Returns:
        The signal's values, concatenated back into one array.

    Raises:
        TimeFFormatError: If a locator carries no row offset, which no Parquet shard can address.
    """
    pieces: list[np.ndarray] = []
    for locator in sorted(locators, key=lambda item: item.chunk_idx):
        if locator.chunk_minor_idx is None:
            raise TimeFFormatError(f"{locator.chunk_file}: a Parquet locator needs a row offset, got none")
        parquet_file = pq.ParquetFile(root / locator.chunk_file)
        group = parquet_file.read_row_group(locator.chunk_major_idx, columns=["values"])
        # Take the list scalar's child array rather than as_py(): a Python list of floats would come
        # back as float64 whatever the column's dtype, silently widening a float32 signal.
        chunk = group.column("values")[locator.chunk_minor_idx].values
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
