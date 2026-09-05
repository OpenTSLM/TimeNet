"""Zarr values backend (writer side): one Zarr array per ``(spec_type, stores_time_offsets)`` partition.

This backend is an alternative to the default Parquet shard store. It uses Zarr's own layout:

- The writer appends every series of a modality along time to one typed array in a ``time_series.zarr``
  group. Zarr chunks the storage itself, so a series is one index row (one placement spanning its full
  length), not a run of ``chunk_max_bytes`` logical chunks.
- The writer partitions arrays by ``(spec_type, stores_time_offsets)``, not by ``spec_type`` alone. A series
  that stores time offsets writes its values under ``_irregular/`` and its int64 time offsets under
  ``_time_offsets/``. The two arrays advance together because every series in that partition writes to
  both. That is what lets the index's single ``chunk_major_idx`` element offset address either one.
  Partitioning by ``spec_type`` alone gives the time offsets array only some of the series, so the two
  arrays drift apart with nothing to notice.
- The writer buffers appends per partition and flushes them at shard-aligned boundaries, so it writes
  every Zarr shard object exactly once. A ``resize()``-per-series append rewrites the trailing shard for
  every series.
  The backend sorts series by partition, so only one partition is open at a time. An irregular partition
  keeps its values appender and its time offsets appender open together, so it buffers near two shards.

The ``(array path, element start)`` pair locates a chunk. The shared index carries that location in its
backend-agnostic ``chunk_file`` / ``chunk_major_idx`` locator (``chunk_minor_idx`` is unused).

This backend needs the ``zarr`` extra (``pip install 'timenet[zarr]'``). This module imports zarr lazily,
so the Parquet core never needs it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import quote

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timenet.dataset import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec
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
_BLOSC_MAX_CLEVEL = 9
# One placement normally spans a whole series. The writer splits only to keep n_values inside int32 (the
# index column type). 2^30 values is 4 GiB of float32 per placement.
_MAX_PLACEMENT_VALUES = 2**30
_BYTES_PER_TIME_OFFSET = 8


#: Group that holds the values of series that store per-value time offsets. The writer keeps it apart
#: from the regular arrays so a partition's values and time offsets advance together and one offset
#: addresses both.
_IRREGULAR_GROUP = "_irregular"
#: Group that holds those series' int64 time offsets, one array per spec type, parallel to _IRREGULAR_GROUP.
_TIME_OFFSETS_GROUP = "_time_offsets"
#: Group that holds the per-timestep validity of nullable series. One boolean array sits under it for
#: each values array, so a validity array sits parallel to the values it describes. Zarr has no null
#: concept of its own, so a nullable series stores its values densely and its missing timesteps here.
#: A non-nullable spec writes nothing under it.
_VALIDITY_GROUP = "_validity"


def _scalar_bytes(dtype: str) -> int:
    """Return the byte cost of one scalar value, for chunk and shard sizing.

    Args:
        dtype: The spec's dtype tag.

    Returns:
        Bytes per value.
    """
    if dtype == "str":
        return np.dtypes.StringDType().itemsize
    if dtype == "enum":
        return np.dtype(np.int32).itemsize
    return np.dtype(dtype).itemsize


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


def _validity_array_path(values_path: str) -> str:
    """Return the validity array path that sits parallel to a values array path.

    The result keeps the values path's own group, so an irregular partition's validity lands under
    ``_validity/_irregular/`` rather than colliding with the regular partition of the same modality.

    Args:
        values_path: The values array's path, as :func:`_value_array_path` returns it.

    Returns:
        The path relative to the Zarr store root.
    """
    return f"{_VALIDITY_GROUP}/{values_path}"


def _array_name(spec_type: str) -> str:
    """Encode a logical spec type as one filesystem-safe Zarr path segment.

    This function percent-encodes every character outside the URL-unreserved set. Path separators never
    leak into the path, and distinct spec types map to distinct single segments. The encoding is
    reversible, so it is injective. ``.`` and ``..`` are the only unreserved but unsafe segments.
    :class:`TimeSeriesSpec` rejects them.

    Returns:
        The percent-encoded physical array name.
    """
    return quote(spec_type, safe="")


#: What to write into a null slot. Zarr is dense, so an absent timestep still occupies storage; this
#: value is meaningless and only the parallel validity array says the timestep is missing. Each entry
#: is the zero-equivalent its dtype accepts, since ``fill_null`` refuses a cross-type fill.
_NULL_PLACEHOLDER: dict[str, object] = {"bool": False, "str": "", "enum": 0}


def _dense_and_validity(
    arrow_values: pa.Array, spec: TimeSeriesSpec
) -> tuple[Shaped[np.ndarray, " time *value"], np.ndarray | None]:
    """Return a series' values densely, plus its per-timestep validity when the spec is nullable.

    Zarr stores dense arrays, so an absent timestep still occupies its slot. The value written there
    is a zero-equivalent placeholder and means nothing by itself: the validity array is what says the
    timestep is missing. A nullable series always yields a validity array, even with no nulls, so the
    values and validity arrays stay the same length and one element offset addresses both.

    Args:
        arrow_values: The validated Arrow values.
        spec: The series' spec, for its dtype, per-step shape, and nullability.

    Returns:
        The dense values and the validity mask (``True`` = present), or ``(values, None)`` for a
        non-nullable spec.
    """
    if spec.dtype == "enum":
        # index_in maps each label to its codebook position in C. This replaces a to_pylist() plus
        # a Python lookup for every value in the series.
        categories = pa.array(spec.categories, type=arrow_values.type.value_type)
        codes = pc.index_in(arrow_values, value_set=categories)  # ty: ignore[unresolved-attribute]
        values = codes.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int32)
    elif isinstance(arrow_values, pa.FixedShapeTensorArray):
        # The tensor's storage already holds a value at every slot, absent timesteps included.
        values = arrow_values.to_numpy_ndarray()
    else:
        # A nullable primitive would otherwise convert to float64 with NaN at every null, which
        # cannot be written to an integer or boolean array. Fill with the zero-equivalent for the
        # dtype first, and let the mask carry the meaning.
        filled = arrow_values
        if arrow_values.null_count:
            filled = arrow_values.fill_null(_NULL_PLACEHOLDER.get(spec.dtype, 0))
        values = filled.to_numpy(zero_copy_only=False)
    if not spec.nullable:
        return values, None
    # Build the mask only when the spec wants one. A nullable series gets a full mask even with no
    # nulls, so the values and validity arrays stay the same length.
    present = (
        arrow_values.is_valid().to_numpy(zero_copy_only=False)
        if arrow_values.null_count
        else np.ones(len(arrow_values), dtype=bool)
    )
    return values, present


class ZarrValuesBackend(BaseValuesBackend):
    """Streams each series into a per-partition typed N-D Zarr array (chunked + sharded)."""

    name = ValuesBackend.ZARR

    def __init__(self, config: ZarrValuesConfig) -> None:
        """Configure the Zarr backend.

        Args:
            config: Typed Zarr backend options.

        Raises:
            TimeFValidationError: If ``compression`` is not a Blosc codec the backend can apply.
        """
        if config.compression not in _BLOSC_CNAMES:
            raise TimeFValidationError(
                f"unknown compression {config.compression!r} for the zarr values backend; it "
                f"compresses with Blosc, whose codecs are {', '.join(sorted(_BLOSC_CNAMES))}"
            )
        self._staging_dir = config.staging_dir
        self._chunk_max_bytes = config.chunk_max_bytes
        self._shard_target_bytes = config.shard_target_bytes
        self._cname = config.compression
        if not 0 <= config.compression_level <= _BLOSC_MAX_CLEVEL:
            raise TimeFValidationError(
                f"Blosc compression level must be 0-{_BLOSC_MAX_CLEVEL}, got {config.compression_level}"
            )
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
        bytes_per_step = _scalar_bytes(ts.spec.dtype) * max(1, int(np.prod(ts.spec.value_shape)))
        chunk_len = max(1, self._chunk_max_bytes // bytes_per_step)
        shard_len = max(1, (self._shard_target_bytes // bytes_per_step) // chunk_len) * chunk_len
        trailing = ts.spec.value_shape
        zarr_dtype = "int32" if ts.spec.dtype == "enum" else ts.spec.dtype
        return _ArrayAppender(
            group.create_array(
                name=array_path,
                shape=(0, *trailing),
                dtype=zarr_dtype,
                chunks=(chunk_len, *trailing),
                shards=(shard_len, *trailing),
                compressors=codec,
            ),
            shard_len,
        )

    def _time_offsets_appender(self, group: Any, spec_type: str, codec: Any, delta: Any) -> "_ArrayAppender":
        """Create the time offsets array parallel to a partition's values, and wrap it in an appender.

        The Delta filter makes stored time offsets cheap. Time offsets are monotonic, so the deltas are
        small and the Blosc bitshuffle then has little left to do. This saves 15-17% over bitshuffle
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
        # Round down to a whole number of chunks, the same as the values array. Zarr requires the shard
        # shape to be a multiple of the chunk shape. An unrounded value aborts create_array for any byte
        # target where the two do not divide.
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

    def _validity_appender(self, group: Any, array_path: str, codec: Any) -> "_ArrayAppender":
        """Create the validity array parallel to a partition's values, and wrap it in an appender.

        Zarr stores dense arrays and has no null of its own, so a nullable series records its missing
        timesteps here: one boolean per timestep, ``True`` where the timestep is present. The values
        array still holds a zero-equivalent placeholder at an absent timestep, which is what lets one
        element offset address both arrays. A non-nullable series writes nothing here at all.

        Args:
            group: The open Zarr group.
            array_path: The values array's path. The validity path is derived from it, so an
                irregular partition does not collide with the regular one of the same modality.
            codec: The Blosc compressor.

        Returns:
            The appender for that array.
        """
        chunk_len = max(1, self._chunk_max_bytes)  # one byte per timestep
        shard_len = max(1, self._shard_target_bytes // chunk_len) * chunk_len
        return _ArrayAppender(
            group.create_array(
                name=_validity_array_path(array_path),
                shape=(0,),
                dtype="bool",
                chunks=(chunk_len,),
                shards=(shard_len,),
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
            from zarr.codecs import BloscCodec  # noqa: PLC0415
            from zarr.codecs.numcodecs import Delta  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("the zarr values backend needs the zarr extra: pip install 'timenet[zarr]'") from exc

        store_path = self._staging_dir / _STORE_DIR
        group = zarr.open_group(store=store_path, mode="w")
        codec = BloscCodec(
            cname=cast(Literal["zstd", "lz4", "lz4hc", "zlib", "blosclz"], self._cname),
            clevel=self._clevel,
            shuffle="bitshuffle",
        )

        partitions: dict[tuple[str, bool], _Partition] = {}
        placements: dict[tuple[str, int], ChunkPlacement] = {}
        closed = 0
        active: tuple[str, bool] | None = None  # one partition is open at a time, see the sort below
        total = len(unique_series)
        # The sort is stable, so the caller's (spec_type, signal, time_series_id) order survives within a
        # partition. The group by whether a series stores time offsets keeps its values array and its time
        # offsets array the same length. Every series in an irregular partition writes to both, so one
        # element offset addresses either.
        ordered = sorted(unique_series, key=lambda ts: (ts.spec.spec_type, ts.time_offsets_loader is not None))
        for completed, ts in enumerate(ordered, start=1):
            arrow_values = read_and_validate(ts)
            values, validity = _dense_and_validity(arrow_values, ts.spec)
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
                        validity=self._validity_appender(group, array_path, codec) if ts.spec.nullable else None,
                    )
                active = partition
            base = partitions[partition].append(
                values,
                None if arrow_time_offsets is None else arrow_time_offsets.to_numpy(zero_copy_only=False),
                validity,
            )
            rel = f"{_STORE_DIR}/{array_path}"
            for chunk_idx, start in enumerate(range(0, len(values), _MAX_PLACEMENT_VALUES)):
                n = min(_MAX_PLACEMENT_VALUES, len(values) - start)
                placements[ts.time_series_id, chunk_idx] = ChunkPlacement(
                    chunk_file=rel,
                    data_index=ChunkDataIndex(major_idx=base + start, minor_idx=None),
                    spec_type=spec_type,
                    signal=ts.signal,
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
    """Hold one ``(spec_type, stores_time_offsets)`` partition: a values array and, when irregular, time offsets.

    The dataclass holds them together to keep the arrays the same length. Every series writes to the
    arrays its partition has, so one element offset addresses all of them. A caller cannot append to
    one and forget the other.
    """

    values: "_ArrayAppender"
    time_offsets: "_ArrayAppender | None" = None
    validity: "_ArrayAppender | None" = None

    def append(self, values: np.ndarray, time_offsets: np.ndarray | None, validity: np.ndarray | None = None) -> int:
        """Append one series to the partition and report where its values landed.

        Args:
            values: The series values.
            time_offsets: Its int64 time offsets, or ``None`` for a partition that stores none.
            validity: Its per-timestep validity, or ``None`` for a non-nullable partition.

        Returns:
            The element offset the values were written at.

        Raises:
            TimeFValidationError: If ``time_offsets`` or ``validity`` disagrees with what this
                partition stores.
        """
        if (validity is None) != (self.validity is None):
            raise TimeFValidationError(
                "a series' validity must match its partition: a nullable spec writes a validity mask "
                "for every series, even one with no nulls, and any other writes none"
            )
        if (time_offsets is None) != (self.time_offsets is None):
            raise TimeFValidationError(
                "a series' time offsets must match its partition: an irregular partition takes time offsets "
                "from every series and any other takes none"
            )
        base = self.values.logical_len
        self.values.append(values)
        if self.time_offsets is not None and time_offsets is not None:
            self.time_offsets.append(time_offsets)
        if self.validity is not None and validity is not None:
            self.validity.append(validity)
        return base

    def finish(self) -> int:
        """Flush the partition's arrays.

        Returns:
            How many Zarr arrays this partition finalized: one for the values, plus one each for the
            time offsets and the validity mask where the partition has them.
        """
        self.values.finish()
        if self.time_offsets is not None:
            self.time_offsets.finish()
        if self.validity is not None:
            self.validity.finish()
        return 1 + (self.time_offsets is not None) + (self.validity is not None)


class _ArrayAppender:
    """Buffer appends to one Zarr array and flush at shard-aligned boundaries.

    The appender writes only whole, aligned shards, so it creates each shard object exactly once.
    :meth:`finish` writes the single trailing partial shard. Without this buffer, every per-series append
    rewrites the trailing shard.
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
