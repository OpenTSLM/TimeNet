"""The :class:`TimeSeries` reference type: one logical stream with a lazy Arrow loader."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import assert_never

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.errors import TimeFValidationError
from timenet.types import Span, SpanFrame, TimeSeriesSpec, new_id


@dataclass(frozen=True, eq=False, kw_only=True)
class TimeSeries:
    """Reference to one logical stream of time-series data, with optional windowing and a lazy loader.

    Identity-based equality (``eq=False``): the writer dedupes by ``time_series_id``, not by value, so
    reusing one instance across samples (or giving two instances the same explicit id) shares one chunk
    on disk. Consumers read values through :meth:`to_arrow` / :meth:`to_numpy`; ``loader`` is plumbing
    supplied by the connector (raw source) at curation or by :class:`~timenet.reader.TimeFReader`
    (the manifest-selected values backend) on read-back.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, units, dtype, and per-timestep shape."""
    channel: str
    """Name of this channel within the modality; must be non-empty."""
    time_axis: TimeAxis
    """Where this series' values sit in time: a :class:`~timenet.dataset.axis.RegularAxis` for a
    cadence, an :class:`~timenet.dataset.axis.IrregularAxis` for stored per-value time offsets, or an
    :class:`~timenet.dataset.axis.OrdinalAxis` for a sequence with no time at all."""
    loader: Callable[[], pa.Array]
    """Lazy callable returning the series' values as an Arrow array."""
    time_offsets_loader: Callable[[], pa.Array] | None = None
    """Lazy callable returning one int64 microsecond time offset per value, for an irregular series only.

    Required exactly when ``time_axis`` is an :class:`~timenet.dataset.axis.IrregularAxis`, and
    rejected otherwise: a regular axis computes its time offsets and an ordinal one has none, so a stream
    attached to either would be a second, unreconcilable answer to the same question."""
    source_id: str | None = None
    """Optional identifier of the raw source recording."""
    time_series_id: str = field(default_factory=new_id)
    """Stable identity used to dedupe and share chunks; defaults to a UUIDv7."""
    n_values: int
    """How many values the series holds, counting one per timestep. A series whose ``spec`` gives each
    timestep a shape contributes one value per timestep, not one per scalar, which is the same count
    a chunk's ``n_values`` reports.

    """

    def __post_init__(self) -> None:
        """Validate the intrinsic per-series invariants.

        Raises:
            TimeFValidationError: If ``n_values`` is not a positive integer, or if ``time_offsets_loader``
                and the axis shape disagree about whether this series stores per-value time offsets.
            ValueError: If ``channel`` is empty. The axis validates itself.
        """
        if not self.channel:
            raise ValueError("TimeSeries.channel must be non-empty")
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

        Derived from the axis rather than stored, so the window and the axis cannot disagree. The
        dispatch is positive and ends in :func:`~typing.assert_never`: a new axis shape breaks this
        method at type-check time instead of falling into whichever branch happens to be last.

        Returns:
            ``(first time_offset, one past the last)`` in microseconds, or ``None`` for an ordinal series.
        """
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            return (axis.time_offset_us(0), axis.time_offset_us(self.n_values))
        if isinstance(axis, IrregularAxis):
            # One microsecond past the last stored time_offset: the axis states no cadence, so there is no
            # next time offset to end on, and a microsecond is the finest the format addresses.
            return (axis.first_us, axis.last_us + 1)
        if isinstance(axis, OrdinalAxis):
            return None
        assert_never(axis)

    @classmethod
    def from_values(  # noqa: PLR0913
        cls,
        values: np.ndarray | Sequence[float],
        *,
        spec: TimeSeriesSpec,
        channel: str,
        time_axis: TimeAxis,
        source_id: str | None = None,
        time_series_id: str | None = None,
    ) -> "TimeSeries":
        """Build a series from already-materialized values, wrapping them in a float32 loader.

        The convenience path for connectors that hold an in-memory array: it caches ``values`` as a
        float32 Arrow array behind the loader and takes ``n_values`` from the array's own length. Use the
        ``loader=`` constructor directly for genuinely lazy sources (files, remote shards), where the
        length has to be stated because nothing has read the values yet.

        Args:
            values: The channel's values (cast to float32).
            spec: The series' measurement-modality spec.
            channel: The channel name.
            time_axis: Where the values sit in time.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.

        Returns:
            The constructed :class:`TimeSeries`.
        """
        array = pa.array(np.asarray(values, dtype=np.float32))
        return cls(
            spec=spec,
            channel=channel,
            time_axis=time_axis,
            loader=lambda: array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            n_values=len(array),
        )

    @classmethod
    def from_irregular(  # noqa: PLR0913
        cls,
        values: np.ndarray | Sequence[float],
        *,
        time_offsets_us: np.ndarray | Sequence[int],
        spec: TimeSeriesSpec,
        channel: str,
        source_id: str | None = None,
        time_series_id: str | None = None,
    ) -> "TimeSeries":
        """Build an irregular series from materialized values and their time offsets.

        The axis endpoints come from the stream itself, so the two cannot disagree: there is no way to
        state a first or last time offset that the time offsets do not have. Convert wall-clock moments with
        :func:`~timenet.dataset.axis.time_offsets_from_datetimes` before calling.

        Args:
            values: The channel's values (cast to float32).
            time_offsets_us: One time offset per value, in microseconds from the sample's relative zero.
            spec: The series' measurement-modality spec.
            channel: The channel name.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.

        Returns:
            The constructed :class:`TimeSeries`.

        Raises:
            TimeFValidationError: If the time offsets are unusable, or if there is not exactly one per
                value.
        """
        array = pa.array(np.asarray(values, dtype=np.float32))
        time_offsets = to_time_offsets_us(time_offsets_us)
        if len(time_offsets) != len(array):
            raise TimeFValidationError(
                f"an irregular series needs one time offset per value, got {len(time_offsets)} time offsets for {len(array)} values"
            )
        time_offset_array = pa.array(time_offsets)
        return cls(
            spec=spec,
            channel=channel,
            time_axis=IrregularAxis.spanning(time_offsets),
            loader=lambda: array,
            time_offsets_loader=lambda: time_offset_array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            n_values=len(array),
        )

    def time_offsets_us(self) -> np.ndarray:
        """Read this series' per-value time offsets.

        Available only on an irregular series. A regular axis computes its time offsets without a read and
        an ordinal one has none, so neither has a stream to return.

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

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open temporal step range as Arrow without forcing a NumPy conversion.

        Range-aware storage loaders read only the intersecting chunks. Connector loaders that only
        implement the original no-argument callable remain compatible through a full-read slice.

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
        """Return the half-open step range of this series that ``span`` covers.

        The bridge to step-based forecasting libraries: ``stop - start`` is the horizon ``h`` that
        GluonTS, Nixtla and fev speak in, and the pair feeds :meth:`read_steps` to read the ground truth.

        A steps span already counts in this series' own steps, so the range is the span, checked against
        its length. A seconds span is located on the axis instead: the steps whose instants fall in
        ``[start, end)``, each end rounded up to the next step. An ordinal series has no timeline, so a
        seconds span has no answer on it.

        Args:
            span: The interval to locate. A steps span must name this series.

        Returns:
            ``(start, stop)`` step indices, half-open, from this series' first step.

        Raises:
            TimeFValidationError: If ``span`` is a point; if a steps span does not name this series or
                runs past its length; or if a seconds span is used on an ordinal series.
        """
        if span.end is None:
            raise TimeFValidationError(f"step_range needs an interval span, not a point, got {span!r}")
        if span.frame is SpanFrame.STEPS:
            if span.time_series_ids is None or self.time_series_id not in span.time_series_ids:
                raise TimeFValidationError(
                    f"a steps span counts on {span.time_series_ids}, not this series {self.time_series_id!r}"
                )
            if span.end > self.n_values:
                raise TimeFValidationError(
                    f"steps span ({span.start}, {span.end}) runs past this series' {self.n_values} steps"
                )
            return (span.start, span.end)
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            return (axis.index_at_or_after(span.start), axis.index_at_or_after(span.end))
        if isinstance(axis, IrregularAxis):
            offsets = self.time_offsets_us()
            return (
                int(np.searchsorted(offsets, span.start, side="left")),
                int(np.searchsorted(offsets, span.end, side="left")),
            )
        if isinstance(axis, OrdinalAxis):
            raise TimeFValidationError(
                f"a seconds span has no step range on ordinal series {self.time_series_id!r}, which has "
                f"no timeline; name the horizon in steps instead"
            )
        assert_never(axis)
