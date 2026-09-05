"""The :class:`TimeSeries` reference type: one logical stream with a lazy Arrow loader."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import assert_never

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.errors import TimeFValidationError
from timenet.types import Span, StepInterval, TimeInterval, TimeSeriesSpec, new_id


def _validate_enum_values(
    spec: TimeSeriesSpec,
    values: Iterable[object],
) -> None:
    """Reject enum values outside the spec's codebook.

    Args:
        spec: The series' spec with its ``categories``.
        values: The label values to check.

    Raises:
        TimeFValidationError: If any value is not in ``spec.categories``.
    """
    allowed = set(spec.categories)
    unknown = {v for v in values if v not in allowed and not (v is None and spec.nullable)}
    if unknown:
        raise TimeFValidationError(
            f"enum series for {spec.spec_type!r} has values outside its categories "
            f"({list(spec.categories)!r}): {sorted(unknown, key=str)}"
        )


def _array_from_values(
    spec: TimeSeriesSpec, values: np.ndarray | Sequence[bool | int | float | str | None]
) -> pa.Array:
    """Build the declared scalar Arrow array without discarding Python nulls.

    Returns:
        Values cast to the spec's dtype with their validity bitmap preserved.

    Raises:
        TimeFValidationError: If a non-nullable spec receives a null value.
    """
    if any(value is None for value in values):
        if not spec.nullable:
            raise TimeFValidationError(f"series for {spec.spec_type!r} has null values but nullable=False")
        arrow_type = pa.string() if spec.dtype in {"str", "enum"} else pa.from_numpy_dtype(np.dtype(spec.dtype))
        array = pa.array(values, type=arrow_type)
    elif spec.dtype in {"str", "enum"}:
        array = pa.array(values, type=pa.string())
    else:
        # Keep the existing NumPy casting behavior for non-null inputs.
        array = pa.array(np.asarray(values, dtype=np.dtype(spec.dtype)))
    if spec.dtype == "enum":
        _validate_enum_values(spec, array.to_pylist())
        array = array.dictionary_encode()
    return array


@dataclass(frozen=True, eq=False, kw_only=True)
class TimeSeries:
    """Reference to one logical stream of time-series data, with optional windowing and a lazy loader.

    The writer dedupes by ``time_series_id``, not by value (``eq=False``). If you reuse one instance
    across records, or give two instances the same explicit id, they share one chunk on disk. Consumers
    read values through :meth:`to_arrow` or :meth:`to_numpy`. The connector supplies ``loader`` at
    build, or :class:`~timenet.reader.TimeFReader` supplies it on read-back.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, units, dtype, and per-timestep shape."""
    signal: str
    """Name of this signal within the modality. The signal must be non-empty."""
    time_axis: TimeAxis
    """Where this series' values sit in time. A :class:`~timenet.dataset.axis.RegularAxis` gives a
    cadence, an :class:`~timenet.dataset.axis.IrregularAxis` stores per-value time offsets, and an
    :class:`~timenet.dataset.axis.OrdinalAxis` marks a sequence with no time at all."""
    loader: Callable[[], pa.Array]
    """Lazy callable returning the series' values as an Arrow array."""
    time_offsets_loader: Callable[[], pa.Array] | None = None
    """Lazy callable returning one int64 microsecond time offset per value, for an irregular series only.

    An :class:`~timenet.dataset.axis.IrregularAxis` requires this callable, and any other axis
    rejects it. A regular axis computes its time offsets and an ordinal one has none. A stream attached to
    either gives a second, conflicting answer to the same question."""
    source_id: str | None = None
    """Optional identifier of the raw source recording."""
    time_series_id: str = field(default_factory=new_id)
    """Stable identity used to dedupe and share chunks. Defaults to a UUIDv7."""
    n_values: int
    """The number of values the series holds, one per timestep. If ``spec`` gives each timestep a shape,
    the series counts one value per timestep, not one per scalar. This matches the count a chunk's
    ``n_values`` reports.

    """

    def __post_init__(self) -> None:
        """Validate the intrinsic per-series invariants.

        Raises:
            TimeFValidationError: If ``n_values`` is not a positive integer, or if ``time_offsets_loader``
                and the axis shape disagree about whether this series stores per-value time offsets.
            ValueError: If ``signal`` is empty. The axis validates itself.
        """
        if not self.signal:
            raise ValueError("TimeSeries.signal must be non-empty")
        if isinstance(self.n_values, bool) or not isinstance(self.n_values, int) or self.n_values <= 0:
            raise TimeFValidationError(f"TimeSeries.n_values must be a positive integer, got {self.n_values!r}")
        irregular = isinstance(self.time_axis, IrregularAxis)
        if irregular and self.time_offsets_loader is None:
            raise TimeFValidationError(
                "an IrregularAxis series must carry time_offsets_loader: the axis states no cadence, so "
                "nothing else can say where its values sit. Build it with TimeSeries.from_irregular()"
            )
        if not irregular and self.time_offsets_loader is not None:
            raise TimeFValidationError(
                f"time_offsets_loader is only for an IrregularAxis series, but this one has "
                f"{type(self.time_axis).__name__}, which already determines every time offset"
            )

    @property
    def span_us(self) -> tuple[int, int] | None:
        """The half-open microsecond window this series covers, or ``None`` if it has no timeline.

        The axis derives this window, so the window and the axis cannot disagree. The dispatch ends in
        :func:`~typing.assert_never`, so a new axis shape breaks this method at type-check time.

        Returns:
            ``(first time_offset, one past the last)`` in microseconds, or ``None`` for an ordinal series.
        """
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            return (axis.time_offset_us(0), axis.time_offset_us(self.n_values))
        if isinstance(axis, IrregularAxis):
            # One microsecond past the last stored time offset. The axis states no cadence, so there is
            # no next time offset to end on. A microsecond is the finest unit the format addresses.
            return (axis.first_us, axis.last_us + 1)
        if isinstance(axis, OrdinalAxis):
            return None
        assert_never(axis)

    @classmethod
    def from_values(  # noqa: PLR0913
        cls,
        values: np.ndarray | Sequence[bool | int | float | str | None],
        *,
        spec: TimeSeriesSpec,
        signal: str,
        time_axis: TimeAxis,
        source_id: str | None = None,
        time_series_id: str | None = None,
    ) -> "TimeSeries":
        """Build a series from already-materialized values and wrap them in a loader for the spec's dtype.

        Use this path for connectors that hold an in-memory array. It caches ``values`` as an Arrow
        array cast to the spec's dtype behind the loader and takes ``n_values`` from the array length.
        For lazy sources like files or remote shards, use the ``loader=`` constructor directly and
        state the length, because nothing has read the values yet.

        Args:
            values: The signal's values (cast to the spec's dtype; for a ``"str"`` or ``"enum"``
                spec, the strings). Python ``None`` marks a missing timestep when ``spec.nullable``
                is true.
            spec: The series' measurement-modality spec.
            signal: The signal name.
            time_axis: Where the values sit in time.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.

        Returns:
            The constructed :class:`TimeSeries`.
        """
        array = _array_from_values(spec, values)
        return cls(
            spec=spec,
            signal=signal,
            time_axis=time_axis,
            loader=lambda: array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            n_values=len(array),
        )

    @classmethod
    def from_irregular(  # noqa: PLR0913
        cls,
        values: np.ndarray | Sequence[bool | int | float | str | None],
        *,
        time_offsets_us: np.ndarray | Sequence[int],
        spec: TimeSeriesSpec,
        signal: str,
        source_id: str | None = None,
        time_series_id: str | None = None,
    ) -> "TimeSeries":
        """Build an irregular series from materialized values and their time offsets.

        The axis endpoints come from the stream itself, so the two cannot disagree. You cannot state a
        first or last time offset that the time offsets do not have. To convert wall-clock moments, use
        :func:`~timenet.dataset.axis.time_offsets_from_datetimes` before you call.

        Args:
            values: The signal's values (cast to the spec's dtype; for a ``"str"`` or ``"enum"``
                spec, the strings). Python ``None`` marks a missing timestep when ``spec.nullable``
                is true.
            time_offsets_us: One time offset per value, in microseconds from the record's relative zero.
            spec: The series' measurement-modality spec.
            signal: The signal name.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.

        Returns:
            The constructed :class:`TimeSeries`.

        Raises:
            TimeFValidationError: If the time offsets are unusable, or there is not exactly one per
                value.
        """
        array = _array_from_values(spec, values)
        time_offsets = to_time_offsets_us(time_offsets_us)
        if len(time_offsets) != len(array):
            raise TimeFValidationError(
                f"an irregular series needs one time offset per value, got {len(time_offsets)} time offsets for {len(array)} values"
            )
        time_offset_array = pa.array(time_offsets)
        return cls(
            spec=spec,
            signal=signal,
            time_axis=IrregularAxis.spanning(time_offsets),
            loader=lambda: array,
            time_offsets_loader=lambda: time_offset_array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            n_values=len(array),
        )

    def time_offsets_us(self) -> np.ndarray:
        """Read this series' per-value time offsets.

        Only an irregular series has this. A regular axis computes its time offsets without a read, and an
        ordinal one has none. So neither has a stream to return.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFValidationError: If this series' axis is not an
                :class:`~timenet.dataset.axis.IrregularAxis`.
        """
        if self.time_offsets_loader is None:
            raise TimeFValidationError(
                f"only an IrregularAxis series stores time offsets, and this one has {type(self.time_axis).__name__}"
            )
        return self.time_offsets_loader().to_numpy(zero_copy_only=False)

    def to_arrow(self) -> pa.Array:
        """Read the series' values as an Arrow array.

        Returns:
            A primitive array for scalar values or a fixed-shape tensor array for N-D values.
        """
        return self.loader()

    def to_numpy(self) -> Shaped[np.ndarray, " time *value"]:
        """Read the series' values as a NumPy array.

        Returns:
            The series' values with shape ``(n_steps, *spec.value_shape)``.
        """
        values = self.to_arrow()
        if isinstance(values, pa.FixedShapeTensorArray):
            return values.to_numpy_ndarray()
        return values.to_numpy(zero_copy_only=False)

    def to_numpy_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Read values and an aligned boolean mask of observed timesteps.

        Missing positions contain zero or empty strings in the dense values buffer. Use the mask
        for missingness; these fill values are not observations. Numeric and boolean dtypes are
        preserved, including when every timestep is missing.

        Returns:
            Values shaped ``(n_steps, *spec.value_shape)`` and validity shaped ``(n_steps,)``.
        """
        values = self.to_arrow()
        valid = values.is_valid().to_numpy(zero_copy_only=False)
        if isinstance(values, pa.FixedShapeTensorArray):
            if values.null_count:
                scalar_fill = False if pa.types.is_boolean(values.type.value_type) else 0
                fill = pa.scalar([scalar_fill] * values.storage.type.list_size, type=values.storage.type)
                values = pa.ExtensionArray.from_storage(values.type, pc.fill_null(values.storage, fill))
            dense = values.storage.flatten().to_numpy(zero_copy_only=False).reshape((len(values), *values.type.shape))
            if values.type.permutation:
                dense = dense.transpose((0, *(axis + 1 for axis in values.type.permutation)))
            return dense, valid
        if values.null_count:
            if isinstance(values, pa.DictionaryArray):
                values = values.dictionary_decode()
            fill_value = "" if pa.types.is_string(values.type) else False if pa.types.is_boolean(values.type) else 0
            values = pc.fill_null(values, pa.scalar(fill_value, type=values.type))
        return values.to_numpy(zero_copy_only=False), valid

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open temporal step range as Arrow without forcing a NumPy conversion.

        Range-aware storage loaders read only the intersecting chunks. A connector loader that implements
        only the no-argument callable stays compatible through a full-read slice.

        Args:
            start: First temporal step, inclusive.
            stop: Last temporal step, exclusive.

        Returns:
            A primitive Arrow array for scalar series or a fixed-shape tensor array for N-D series.

        Raises:
            TimeFValidationError: If the range is negative or reversed.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError(f"expected 0 <= start <= stop, got start={start}, stop={stop}")
        read_steps = getattr(self.loader, "read_steps", None)
        if read_steps is not None:
            return read_steps(start, stop)
        return self.to_arrow().slice(start, stop - start)

    def step_range(self, span: Span) -> tuple[int, int]:
        """Return the half-open step range ``(start, stop)`` of this series that ``span`` covers.

        The bridge to step-based forecasting libraries: ``stop - start`` is the horizon ``h`` that
        GluonTS, Nixtla, and fev speak in. The pair feeds :meth:`read_steps` to read the ground
        truth. A step span already counts in this series' own steps, so it is the range, bounded by the
        series' length. The axis locates a time span instead: the steps whose time offsets fall in
        ``[start_us, end_us)``, each rounded up to the next step. An ordinal series has no timeline, so a
        time span has no answer on it.

        Args:
            span: The interval to locate. A step span must name this series.

        Returns:
            ``(start, stop)`` step indices, half-open, from this series' first step.

        Raises:
            TimeFValidationError: If ``span`` is a point. If a step span does not name this series or
                runs past its length. If an ordinal series receives a time span, or the span resolves
                past the series' steps. If the located range is empty.
        """
        if isinstance(span, StepInterval):
            if span.time_series_id != self.time_series_id:
                raise TimeFValidationError(
                    f"a step span counts on {span.time_series_id!r}, not this series {self.time_series_id!r}"
                )
            if span.stop > self.n_values:
                raise TimeFValidationError(
                    f"step span ({span.start}, {span.stop}) runs past this series' {self.n_values} steps"
                )
            return (span.start, span.stop)
        if not isinstance(span, TimeInterval):  # a point (TimePoint or StepPoint) names no range
            raise TimeFValidationError(f"step_range needs an interval span, not a point, got {span!r}")
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            start, stop = axis.index_at_or_after(span.start_us), axis.index_at_or_after(span.end_us)
        elif isinstance(axis, IrregularAxis):
            offsets = self.time_offsets_us()
            start = int(np.searchsorted(offsets, span.start_us, side="left"))
            stop = int(np.searchsorted(offsets, span.end_us, side="left"))
        elif isinstance(axis, OrdinalAxis):
            raise TimeFValidationError(
                f"a seconds span has no step range on ordinal series {self.time_series_id!r}, which has "
                f"no timeline; name the horizon in steps instead"
            )
        else:
            assert_never(axis)
        if start < 0 or stop > self.n_values:
            raise TimeFValidationError(
                f"span {span!r} runs past the {self.n_values} steps of series {self.time_series_id!r}: "
                f"it resolves to ({start}, {stop})"
            )
        if stop <= start:
            raise TimeFValidationError(
                f"span {span!r} covers no steps of series {self.time_series_id!r}: it resolves to the "
                f"empty range ({start}, {stop}). A forecast horizon needs at least one step"
            )
        return (start, stop)
