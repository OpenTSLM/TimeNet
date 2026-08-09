"""The span geometry primitives: one time offset or one interval on a recording timeline.

A span is the shape shared by every time-localized field a task carries: the ``scope`` a task is asked
about and the target regions a :class:`~timenet.types.TemporalLocalizationTask` predicts are the same
type run in opposite directions. :class:`Span` holds the fields and the checks, and is what a field
annotates when either shape fits.

Bounds are whole microseconds on the source recording timeline. Storing an integer is what makes two
equal regions compare equal: a float's resolution changes with magnitude, so the same time offset derived
two ways need not agree. Callers rarely have microseconds, so they rarely write them. Each shape
builds itself from what a source actually has::

    IntervalSpan.seconds(5.0, 8.0)  # recording seconds, rounded once
    IntervalSpan.micros(5_000_000, 8_000_000)  # already integral
    IntervalSpan.from_datetime(t1, t2, start_time=sample.start_time)

    PointSpan.seconds(1.2)
    PointSpan.micros(1_200_000)
    PointSpan.from_datetime(t, start_time=sample.start_time)

The shape is named at the call site rather than inferred from how many bounds were passed, so
``IntervalSpan.seconds(5.0)`` is a type error rather than a point that quietly claims to be an
interval.

A span's numbers are read in one of two frames, named by :class:`SpanFrame` and stored beside them.
Seconds is the common case: microseconds on a recording timeline every series shares. Steps exists
because a series need not have a cadence at all: an ordinal sequence has positions but no timeline, so
``IntervalSpan.steps(0, 12, time_series_ids=(...))`` is the only way to name a region of one. The frame
is stored because a span is read back without the series it came from, so a bare pair of numbers would
otherwise compare equal whether it meant 8 microseconds of a recording or 8 steps of an ordinal series.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique

from timenet.errors import TimeFValidationError
from timenet.types.clock import offset_us, seconds_to_us


@unique
class SpanFrame(StrEnum):
    """Which frame a span's numbers are read in. Stored on disk beside them."""

    SECONDS = "seconds"
    """Microseconds on the source recording timeline, the frame a series' axis places its values in."""
    STEPS = "steps"
    """Step ordinals on the step axis of ``time_series_ids``, counting from their first stored step."""


@dataclass(frozen=True, kw_only=True)
class Span:
    """A point (``end is None``) or half-open interval ``[start, end)``, in one :class:`SpanFrame`.

    In :attr:`SpanFrame.SECONDS` the bounds are microseconds on the **source recording timeline**, the
    same frame a series' axis places its values in, so a span stays meaningful on a windowed sample that
    starts partway into the recording. In :attr:`SpanFrame.STEPS` they are step ordinals on the series
    named by ``time_series_ids``, which is why that field stops being optional there. That a span
    actually falls inside the sample it is attached to is checked by
    :meth:`~timenet.dataset.TimeFDataset.add_task`, which has the sample.

    Annotate with ``Span`` where either shape fits, and with :class:`PointSpan` or
    :class:`IntervalSpan` where only one does. Build with the builders on those two; the base is not
    constructible, so every span in circulation carries the shape it means.
    """

    start: int
    """Start of the interval, or the position itself: microseconds in :attr:`SpanFrame.SECONDS`, a step
    ordinal in :attr:`SpanFrame.STEPS`."""
    end: int | None = None
    """End of the interval, exclusive, in this span's frame; ``None`` makes the span a point."""
    time_series_ids: tuple[str, ...] | None = None
    """Series the span is scoped to; ``None`` covers every series in the sample, and is rejected in
    :attr:`SpanFrame.STEPS`, where a step ordinal means nothing without a series to count on."""
    frame: SpanFrame = SpanFrame.SECONDS
    """Which frame ``start`` and ``end`` are read in."""

    def __post_init__(self) -> None:
        """Coerce ``frame`` and reject a missing or fractional bound, a bad interval, or a bad scope.

        Raises:
            TimeFValidationError: If the base class is constructed directly; if ``frame`` is unknown;
                if ``start`` is missing; if a bound is not whole in this frame; if ``end`` is not
                strictly greater than ``start``; if ``time_series_ids`` is ``()`` rather than ``None``
                or non-empty; or, in :attr:`SpanFrame.STEPS`, if no series is named or the start is
                negative.
        """
        if type(self) is Span:
            raise TimeFValidationError(
                "Span is the shared base, not a shape; build a PointSpan or an IntervalSpan so the "
                "span says which one it is"
            )
        try:
            object.__setattr__(self, "frame", SpanFrame(self.frame))
        except ValueError as exc:
            raise TimeFValidationError(
                f"unknown span frame {self.frame!r}; expected one of {[f.value for f in SpanFrame]}"
            ) from exc
        unit = "whole microseconds" if self.frame is SpanFrame.SECONDS else "a whole step"
        # The field annotations are not enforced at runtime, so a None start would otherwise slip
        # past the loop below (which skips None) and construct a span with no position.
        if self.start is None:
            raise TimeFValidationError(f"Span start must be {unit}, got None; a span always has a start")
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                hint = " Use seconds() to convert from recording seconds" if self.frame is SpanFrame.SECONDS else ""
                raise TimeFValidationError(f"Span {name} must be {unit}, got {value!r}.{hint}")
        if self.end is not None and self.end <= self.start:
            raise TimeFValidationError(f"Span end ({self.end}) must be > start ({self.start})")
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError("Span time_series_ids must be None (whole sample) or non-empty, got ()")
        if self.frame is SpanFrame.STEPS:
            if self.time_series_ids is None:
                raise TimeFValidationError(
                    "a steps span must name time_series_ids: step 5 is a different region on every "
                    "series with a different rate, offset or length. Use a seconds span to cover the sample"
                )
            if self.start < 0:
                raise TimeFValidationError(f"steps span start must be >= 0, got {self.start}")

    @property
    def is_point(self) -> bool:
        """Whether the span marks a single position rather than a bounded interval."""
        return self.end is None

    @property
    def n_steps(self) -> int | None:
        """How many steps a :attr:`SpanFrame.STEPS` interval covers; ``None`` for any other span.

        This is the horizon ``h`` step-based forecasting libraries speak in, readable off the span
        alone because a steps span already counts in the series it names.
        """
        if self.frame is not SpanFrame.STEPS or self.end is None:
            return None
        return int(self.end - self.start)


@dataclass(frozen=True, kw_only=True)
class PointSpan(Span):
    """One time offset on the recording timeline."""

    end: None = None
    """Always ``None``: a point has no second bound."""

    def __post_init__(self) -> None:
        """Validate the base invariants, then reject a point that carries an end.

        Raises:
            TimeFValidationError: If ``end`` is set; a point marks one time offset and has no second bound.
        """
        super().__post_init__()
        if self.end is not None:
            raise TimeFValidationError(
                f"a PointSpan marks one time offset and has no end, got end={self.end!r}. Use "
                f"IntervalSpan for a bounded region"
            )

    @classmethod
    def seconds(cls, at: float, *, time_series_ids: tuple[str, ...] | None = None) -> "PointSpan":
        """Construct a point from recording seconds, rounded to the nearest microsecond.

        Args:
            at: The time offset, in recording seconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start=seconds_to_us(at), time_series_ids=time_series_ids)

    @classmethod
    def micros(cls, at: int, *, time_series_ids: tuple[str, ...] | None = None) -> "PointSpan":
        """Construct a point from whole microseconds, for a source that already has them.

        Args:
            at: The time offset, in microseconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start=at, time_series_ids=time_series_ids)

    @classmethod
    def from_datetime(
        cls,
        at: datetime,
        *,
        start_time: datetime | int | None,
        time_series_ids: tuple[str, ...] | None = None,
    ) -> "PointSpan":
        """Construct a point from a wall-clock moment, relative to the sample it is for.

        A span's bounds are offsets on the recording timeline, so a calendar moment only names an
        time offset once something says when that timeline began. ``start_time`` is that something, which
        is why it is required rather than inferred.

        Args:
            at: The wall-clock moment, timezone-aware.
            start_time: The target sample's ``start_time``.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start=offset_us(at, start_time), time_series_ids=time_series_ids)

    @classmethod
    def steps(cls, at: int, *, time_series_ids: tuple[str, ...]) -> "PointSpan":
        """Construct a point at one step ordinal on the named series.

        Args:
            at: The step ordinal, counting from the series' first stored step.
            time_series_ids: Series the ordinal counts on; required, since a step ordinal names no
                position without one.

        Returns:
            The point, in :attr:`SpanFrame.STEPS`.
        """
        return cls(start=at, frame=SpanFrame.STEPS, time_series_ids=time_series_ids)


@dataclass(frozen=True, kw_only=True)
class IntervalSpan(Span):
    """The half-open range ``[start, end)`` on the recording timeline."""

    # field() is what drops the base's None default; a bare redeclaration inherits it, and an
    # interval with no end would then construct and type-check clean.
    end: int = field()
    """End of the interval in microseconds, exclusive."""

    def __post_init__(self) -> None:
        """Validate the base invariants, then reject an interval with no end.

        Raises:
            TimeFValidationError: If ``end`` is ``None``; an interval needs a second bound. Use
                ``PointSpan`` for a time offset.
        """
        super().__post_init__()
        if self.end is None:
            raise TimeFValidationError("an IntervalSpan needs an end; use PointSpan for a time offset")

    @classmethod
    def seconds(cls, start: float, end: float, *, time_series_ids: tuple[str, ...] | None = None) -> "IntervalSpan":
        """Construct an interval from recording seconds, each bound rounded to the nearest microsecond.

        Args:
            start: Start of the interval, in recording seconds.
            end: End of the interval, exclusive, in recording seconds.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start=seconds_to_us(start), end=seconds_to_us(end), time_series_ids=time_series_ids)

    @classmethod
    def micros(cls, start: int, end: int, *, time_series_ids: tuple[str, ...] | None = None) -> "IntervalSpan":
        """Construct an interval from whole microseconds, for a source that already has them.

        Args:
            start: Start of the interval, in microseconds.
            end: End of the interval, exclusive, in microseconds.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start=start, end=end, time_series_ids=time_series_ids)

    @classmethod
    def from_datetime(
        cls,
        start: datetime,
        end: datetime,
        *,
        start_time: datetime | int | None,
        time_series_ids: tuple[str, ...] | None = None,
    ) -> "IntervalSpan":
        """Construct an interval from wall-clock moments, relative to the sample it is for.

        A span's bounds are offsets on the recording timeline, so calendar moments only name a region
        once something says when that timeline began. ``start_time`` is that something, which is why
        it is required rather than inferred.

        Args:
            start: Wall-clock start, timezone-aware.
            end: Wall-clock end, exclusive and timezone-aware.
            start_time: The target sample's ``start_time``.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(
            start=offset_us(start, start_time),
            end=offset_us(end, start_time),
            time_series_ids=time_series_ids,
        )

    @classmethod
    def steps(cls, start: int, end: int, *, time_series_ids: tuple[str, ...]) -> "IntervalSpan":
        """Construct a half-open interval of step ordinals on the named series.

        Args:
            start: First step ordinal, inclusive, counting from the series' first stored step.
            end: Last step ordinal, exclusive.
            time_series_ids: Series the ordinals count on; required, and they must share a step axis.

        Returns:
            The interval, in :attr:`SpanFrame.STEPS`.
        """
        return cls(start=start, end=end, frame=SpanFrame.STEPS, time_series_ids=time_series_ids)
