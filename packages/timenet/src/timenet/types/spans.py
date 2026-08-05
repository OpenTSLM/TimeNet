"""The span geometry primitives: one instant or one interval on a recording timeline.

A span is the shape shared by every time-localized field a task carries: the ``scope`` a task is asked
about and the target regions a :class:`~timenet.types.TemporalLocalizationTask` predicts are the same
type run in opposite directions. :class:`Span` holds the fields and the checks, and is what a field
annotates when either shape fits.

Bounds are whole microseconds on the source recording timeline. Storing an integer is what makes two
equal regions compare equal: a float's resolution changes with magnitude, so the same instant derived
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
"""

from dataclasses import dataclass, field
from datetime import datetime

from timenet.errors import TimeFValidationError
from timenet.types.clock import seconds_to_us, unix_us


def _offset_us(moment: datetime, start_time: datetime | int | None) -> int:
    """Convert a wall-clock moment to an offset on a sample's recording timeline.

    Args:
        moment: The wall-clock moment, timezone-aware.
        start_time: The target sample's ``start_time``.

    Returns:
        Microseconds from the sample's relative zero.

    Raises:
        TimeFValidationError: If ``start_time`` is ``None``, because a sample with no wall-clock
            anchor has no calendar time to measure a moment against.
    """
    if start_time is None:
        raise TimeFValidationError(
            "from_datetime needs the target sample's start_time, and that sample has none. A sample "
            "with no wall-clock anchor has no calendar time to measure against; use seconds() with "
            "an offset into the recording instead"
        )
    return unix_us(moment) - unix_us(start_time)


@dataclass(frozen=True, kw_only=True)
class Span:
    """A point (``end is None``) or half-open interval ``[start, end)``, optionally per-channel.

    Bounds are microseconds on the **source recording timeline**, the same frame as
    :attr:`~timenet.dataset.TimeSeries.t_start_s`, so a span stays meaningful on a windowed sample
    that starts partway into the recording. That a span actually falls inside the sample it is
    attached to is checked by :meth:`~timenet.dataset.TimeFDataset.add_task`, which has the sample.

    Annotate with ``Span`` where either shape fits, and with :class:`PointSpan` or
    :class:`IntervalSpan` where only one does. Build with the builders on those two; this constructor
    takes the stored fields and is what the reader rebuilds a span with.
    """

    start: int
    """Start of the interval, or the instant itself, in microseconds."""
    end: int | None = None
    """End of the interval in microseconds, exclusive; ``None`` makes the span a point."""
    time_series_ids: tuple[str, ...] | None = None
    """Series the span is scoped to; ``None`` covers every series in the sample."""

    def __post_init__(self) -> None:
        """Reject a fractional bound, a non-positive interval, or an empty ``time_series_ids``.

        Raises:
            TimeFValidationError: If a bound is not whole microseconds, if ``end`` is not strictly
                greater than ``start``, or if ``time_series_ids`` is ``()`` rather than ``None`` or
                non-empty.
        """
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TimeFValidationError(
                    f"Span {name} must be whole microseconds, got {value!r}. Use seconds() to convert "
                    f"from recording seconds"
                )
        if self.end is not None and self.end <= self.start:
            raise TimeFValidationError(f"Span end ({self.end}) must be > start ({self.start})")
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError("Span time_series_ids must be None (whole sample) or non-empty, got ()")

    @property
    def is_point(self) -> bool:
        """Whether the span marks an instant rather than a bounded interval."""
        return self.end is None


@dataclass(frozen=True, kw_only=True)
class PointSpan(Span):
    """One instant on the recording timeline."""

    end: None = None
    """Always ``None``: a point has no second bound."""

    @classmethod
    def seconds(cls, at: float, *, time_series_ids: tuple[str, ...] | None = None) -> "PointSpan":
        """Construct a point from recording seconds, rounded to the nearest microsecond.

        Args:
            at: The instant, in recording seconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start=seconds_to_us(at), time_series_ids=time_series_ids)

    @classmethod
    def micros(cls, at: int, *, time_series_ids: tuple[str, ...] | None = None) -> "PointSpan":
        """Construct a point from whole microseconds, for a source that already has them.

        Args:
            at: The instant, in microseconds.
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
        instant once something says when that timeline began. ``start_time`` is that something, which
        is why it is required rather than inferred.

        Args:
            at: The wall-clock moment, timezone-aware.
            start_time: The target sample's ``start_time``.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start=_offset_us(at, start_time), time_series_ids=time_series_ids)


@dataclass(frozen=True, kw_only=True)
class IntervalSpan(Span):
    """The half-open range ``[start, end)`` on the recording timeline."""

    # field() is what drops the base's None default; a bare redeclaration inherits it, and an
    # interval with no end would then construct and type-check clean.
    end: int = field()
    """End of the interval in microseconds, exclusive."""

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
            start=_offset_us(start, start_time),
            end=_offset_us(end, start_time),
            time_series_ids=time_series_ids,
        )
