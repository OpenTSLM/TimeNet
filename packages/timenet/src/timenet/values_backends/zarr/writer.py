"""Zarr values backend (writer side): one Zarr values array per ``spec_type``.

This backend is an alternative to the default Parquet shard store. It uses Zarr's own layout:

- The writer appends every series of a modality along time to one typed array in a ``time_series.zarr``
  group. Zarr chunks the storage itself, so a series is one index row (one placement spanning its full
  length), not a run of ``chunk_max_bytes`` logical chunks.
- Irregular time offsets live once in the shared control-plane axis. They are not repeated beside each
  Signal's values.
- The writer buffers appends per partition and flushes them at shard-aligned boundaries, so it writes
  every Zarr shard object exactly once. A ``resize()``-per-series append rewrites the trailing shard for
  every series.
  The backend sorts series by specification, so only one partition is open at a time.

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

from timenet.dataset import Signal
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
#: Group that stores one boolean array for each nullable values array.
#: Each boolean marks whether the corresponding timestep is present.
#: Zarr does not represent nulls directly. A non-nullable spec writes nothing here.
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


def _value_array_path(spec_type: str) -> str:
    """Return the array path holding one specification's values.

    Args:
        spec_type: The modality tag.

    Returns:
        The path relative to the Zarr store root.
    """
    return _array_name(spec_type)


def _validity_array_path(values_path: str) -> str:
    """Return the path of the validity array for a values array.

    The path preserves any groups in the values path below ``_validity``.

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


#: Fill values for missing timesteps. The validity array distinguishes these from observations.
#: Each fill value matches its dtype because fill_null rejects other types.
_NULL_PLACEHOLDER: dict[str, object] = {"bool": False, "str": "", "enum": 0}


def _dense_and_validity(
    arrow_values: pa.Array, spec: TimeSeriesSpec
) -> tuple[Shaped[np.ndarray, " time *value"], np.ndarray | None]:
    """Return values and, for nullable specs, a mask that marks present timesteps.

    Missing timesteps still occupy space in Zarr. Their fill values are not observations.
    The validity array marks which timesteps are present. Nullable series always return this array,
    even without nulls. Values and validity arrays have the same length, with matching positions.

    Args:
        arrow_values: The validated Arrow values.
        spec: The series' spec, for its dtype, per-step shape, and nullability.

    Returns:
        The dense values and the validity mask (``True`` = present), or ``(values, None)`` for a
        non-nullable spec.
    """
    if spec.dtype == "enum":
        # index_in finds each label's position in the categories using C code.
        # This avoids a Python lookup for every value.
        categories = pa.array(spec.categories, type=arrow_values.type.value_type)
        codes = pc.index_in(arrow_values, value_set=categories)  # ty: ignore[unresolved-attribute]
        values = codes.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int32)
    elif isinstance(arrow_values, pa.FixedShapeTensorArray):
        # The tensor's storage already holds a value at every slot, absent timesteps included.
        values = arrow_values.to_numpy_ndarray()
    else:
        # Converting nulls to NaN does not work for integer or boolean arrays.
        # Fill missing positions with a value of the same dtype. The mask marks them as missing.
        filled = arrow_values
        if arrow_values.null_count:
            filled = arrow_values.fill_null(_NULL_PLACEHOLDER.get(spec.dtype, 0))
        values = filled.to_numpy(zero_copy_only=False)
    if not spec.nullable:
        return values, None
    # Nullable series always get a mask, even without nulls.
    # Values and validity arrays must keep matching positions.
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

    def _value_appender(self, group: Any, ts: Signal, array_path: str, codec: Any) -> "_ArrayAppender":
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

    def _validity_appender(self, group: Any, array_path: str, codec: Any) -> "_ArrayAppender":
        """Create an array that marks present timesteps and return its appender.

        Each boolean is ``True`` where the timestep is present. Missing positions still occupy
        space in the values array, so the two arrays have matching positions.
        Non-nullable series do not use this array.

        Args:
            group: The open Zarr group.
            array_path: The values array's path. The derived validity path keeps regular and
                irregular arrays separate for the same spec type.
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
        unique_series: list[Signal],
        *,
        read_and_validate: Callable[[Signal], pa.Array],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Append every series to its spec-type array and return per-series placements.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series against its dtype and shape contract.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(arrays_finalized)`` as each partition closes,
                counting a partition's values and optional validity arrays.

        Returns:
            The placements and the Zarr store's files (relative to the staging directory).

        Raises:
            ImportError: If the ``zarr`` extra is not installed.
        """
        try:
            import zarr  # noqa: PLC0415
            from zarr.codecs import BloscCodec  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("the zarr values backend needs the zarr extra: pip install 'timenet[zarr]'") from exc

        store_path = self._staging_dir / _STORE_DIR
        group = zarr.open_group(store=store_path, mode="w")
        codec = BloscCodec(
            cname=cast(Literal["zstd", "lz4", "lz4hc", "zlib", "blosclz"], self._cname),
            clevel=self._clevel,
            shuffle="bitshuffle",
        )

        partitions: dict[str, _Partition] = {}
        placements: dict[tuple[str, int], ChunkPlacement] = {}
        closed = 0
        active: str | None = None
        total = len(unique_series)
        for completed, ts in enumerate(unique_series, start=1):
            arrow_values = read_and_validate(ts)
            values, validity = _dense_and_validity(arrow_values, ts.spec)
            spec_type = ts.spec.spec_type
            array_path = _value_array_path(spec_type)
            if spec_type != active:
                if active is not None:
                    closed += partitions[active].finish()
                    on_file_done(closed)
                if spec_type not in partitions:
                    partitions[spec_type] = _Partition(
                        values=self._value_appender(group, ts, array_path, codec),
                        validity=self._validity_appender(group, array_path, codec) if ts.spec.nullable else None,
                    )
                active = spec_type
            base = partitions[spec_type].append(values, validity)
            rel = f"{_STORE_DIR}/{array_path}"
            for chunk_idx, start in enumerate(range(0, len(values), _MAX_PLACEMENT_VALUES)):
                n = min(_MAX_PLACEMENT_VALUES, len(values) - start)
                placements[ts.id, chunk_idx] = ChunkPlacement(
                    chunk_file=rel,
                    data_index=ChunkDataIndex(major_idx=base + start, minor_idx=None),
                    spec_type=spec_type,
                    signal=ts.name,
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
    """Keep a specification partition's values and validity arrays together.

    Each series writes to all arrays in its partition. This keeps their lengths and positions aligned.
    """

    values: "_ArrayAppender"
    validity: "_ArrayAppender | None" = None

    def append(self, values: np.ndarray, validity: np.ndarray | None = None) -> int:
        """Append one series to the partition and report where its values landed.

        Args:
            values: The series values.
            validity: Its per-timestep validity, or ``None`` for a non-nullable partition.

        Returns:
            The element offset the values were written at.

        Raises:
            TimeFValidationError: If ``validity`` disagrees with what this partition stores.
        """
        if (validity is None) != (self.validity is None):
            raise TimeFValidationError(
                "a series' validity must match its partition: a nullable spec writes a validity mask "
                "for every series, even one with no nulls, and any other writes none"
            )
        base = self.values.logical_len
        self.values.append(values)
        if self.validity is not None and validity is not None:
            self.validity.append(validity)
        return base

    def finish(self) -> int:
        """Flush the partition's arrays.

        Returns:
            The number of completed arrays. This includes values and an optional validity array.
        """
        self.values.finish()
        if self.validity is not None:
            self.validity.finish()
        return 1 + (self.validity is not None)


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
