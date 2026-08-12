"""Span types: one region a task or annotation localizes, in one of two frames.

A region has two independent traits: its shape (a point or a half-open interval) and its frame (time or
steps). The frame changes what the numbers mean, so the frame is the type. Shape is the same trait run
twice, so it is a subtype within each frame.

A time span reads its bounds as microseconds on the **source recording timeline**, the frame a series'
axis places its values in, so it stays meaningful on a windowed sample that starts partway into the
recording. It covers the whole sample, or a subset of series named by ``time_series_ids``.

A step span reads its bounds as ordinal indices into one series' own array. A step index means nothing
without a series to count on, so a step span names exactly one ``time_series_id``. Steps exist for a
series that has no timeline at all: an ordinal sequence has positions but no clock.

Which frame fits a series is decided by the series' axis, not by the caller. A timeline axis (regular or
irregular) takes a time span; an ordinal axis takes a step span.
:meth:`~timenet.dataset.TimeFDataset.add_task` checks a span against the axis of every series it names.

Build a concrete leaf. The bases (:class:`Span`, :class:`TimeSpan`, :class:`StepSpan`) are abstract, so
every span in circulation carries the shape and frame it means::

    TimePoint.seconds(1.2)              TimeInterval.seconds(5.0, 8.0)
    StepPoint(time_series_id="x", start=5)
    StepInterval(time_series_id="x", start=0, stop=12)
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.errors import TimeFValidationError
from timenet.types.clock import check_int64, seconds_to_us


def _check_whole(name: str, value: int, *, unit: str, hint: str = "") -> None:
    """Reject a bound that is not a whole int inside the int64 range TimeF stores it in.

    Args:
        name: The field being checked, named in the error message.
        value: The bound to check.
        unit: How the bound reads when it is not whole (e.g. ``"whole microseconds"``).
        hint: Extra guidance appended to the not-whole message.

    Raises:
        TimeFValidationError: If ``value`` is a bool, is not an ``int``, or falls outside int64.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{name} must be {unit}, got {value!r}"
        if hint:
            message = f"{message}. {hint}"
        raise TimeFValidationError(message)
    check_int64(name, value)


@dataclass(frozen=True, kw_only=True)
class Span:
    """Abstract base for every localized region. Not constructible; build a concrete leaf type.

    The concrete spans are :class:`TimePoint`, :class:`TimeInterval`, :class:`StepPoint`, and
    :class:`StepInterval`. Annotate with ``Span`` where any of them fits, and shared code reads them
    through this base.
    """

    is_point: ClassVar[bool] = False
    """Whether this span marks a single position rather than a bounded interval."""

    def __post_init__(self) -> None:
        """Reject construction of an abstract base.

        Raises:
            TimeFValidationError: If the constructed type is ``Span``, ``TimeSpan``, or ``StepSpan``,
                none of which says which shape and frame it means.
        """
        if type(self) in {Span, TimeSpan, StepSpan}:
            raise TimeFValidationError(
                f"{type(self).__name__} is an abstract span base; build a TimePoint, TimeInterval, "
                f"StepPoint, or StepInterval so the span says which shape and frame it is"
            )

    @property
    def exclusive_end(self) -> int:
        """The exclusive upper bound of the span, in its own frame.

        The region is half-open, so a point ends one unit past its position and an interval ends at
        its stored end. Each concrete leaf supplies this.
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class TimeSpan(Span):
    """Abstract base for a region on the recording timeline. Bounds are microseconds.

    Covers the whole sample when ``time_series_ids`` is ``None``, or a subset of series when it names
    them. Valid only on a series whose axis is a timeline (regular or irregular).
    """

    frame: ClassVar[str] = "seconds"
    """The frame its bounds read in, as stored on disk."""

    start_us: int
    """The position, or the start of the interval, in microseconds on the source recording timeline."""
    time_series_ids: tuple[str, ...] | None = None
    """Series the span is scoped to; ``None`` covers every series in the sample."""

    def __post_init__(self) -> None:
        """Validate the base, then the start bound and the series scope.

        Raises:
            TimeFValidationError: If an abstract base is constructed; if ``start_us`` is not whole
                microseconds; or if ``time_series_ids`` is ``()`` rather than ``None`` or non-empty.
        """
        super().__post_init__()
        _check_whole(
            "TimeSpan start_us",
            self.start_us,
            unit="whole microseconds",
            hint="Use seconds() to convert from recording seconds",
        )
        if self.time_series_ids is not None and not self.time_series_ids:
            raise TimeFValidationError("TimeSpan time_series_ids must be None (whole sample) or non-empty, got ()")


@dataclass(frozen=True, kw_only=True)
class TimePoint(TimeSpan):
    """One point on the recording timeline."""

    is_point: ClassVar[bool] = True

    @property
    def exclusive_end(self) -> int:
        """One microsecond past the point, so the half-open window covering it is ``[start_us, +1)``."""
        return self.start_us + 1

    @classmethod
    def seconds(cls, at: float, *, time_series_ids: tuple[str, ...] | None = None) -> "TimePoint":
        """Construct a point from recording seconds, rounded to the nearest microsecond.

        Args:
            at: The position, in recording seconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start_us=seconds_to_us(at), time_series_ids=time_series_ids)

    @classmethod
    def micros(cls, at: int, *, time_series_ids: tuple[str, ...] | None = None) -> "TimePoint":
        """Construct a point from whole microseconds, for a source that already has them.

        Args:
            at: The position, in microseconds.
            time_series_ids: Series the point is scoped to; ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start_us=at, time_series_ids=time_series_ids)


@dataclass(frozen=True, kw_only=True)
class TimeInterval(TimeSpan):
    """The half-open range ``[start_us, end_us)`` on the recording timeline."""

    end_us: int
    """End of the interval, exclusive, in microseconds."""

    def __post_init__(self) -> None:
        """Validate the time base, then the end bound.

        Raises:
            TimeFValidationError: If ``end_us`` is not whole microseconds, or is not greater than
                ``start_us``.
        """
        super().__post_init__()
        _check_whole(
            "TimeInterval end_us",
            self.end_us,
            unit="whole microseconds",
            hint="Use seconds() to convert from recording seconds",
        )
        if self.end_us <= self.start_us:
            raise TimeFValidationError(f"TimeInterval end_us ({self.end_us}) must be > start_us ({self.start_us})")

    @property
    def exclusive_end(self) -> int:
        """The interval's own exclusive end, ``end_us``."""
        return self.end_us

    @classmethod
    def seconds(cls, start: float, end: float, *, time_series_ids: tuple[str, ...] | None = None) -> "TimeInterval":
        """Construct an interval from recording seconds, each bound rounded to the nearest microsecond.

        Args:
            start: Start of the interval, in recording seconds.
            end: End of the interval, exclusive, in recording seconds.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start_us=seconds_to_us(start), end_us=seconds_to_us(end), time_series_ids=time_series_ids)

    @classmethod
    def micros(cls, start: int, end: int, *, time_series_ids: tuple[str, ...] | None = None) -> "TimeInterval":
        """Construct an interval from whole microseconds, for a source that already has them.

        Args:
            start: Start of the interval, in microseconds.
            end: End of the interval, exclusive, in microseconds.
            time_series_ids: Series the interval is scoped to; ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start_us=start, end_us=end, time_series_ids=time_series_ids)


@dataclass(frozen=True, kw_only=True)
class StepSpan(Span):
    """Abstract base for a region counted in the step ordinals of one series.

    A step index means nothing without a series to count on, so a step span names exactly one
    ``time_series_id``. Valid only on a series whose axis is ordinal (no timeline).
    """

    frame: ClassVar[str] = "steps"
    """The frame its bounds read in, as stored on disk."""

    time_series_id: str
    """The single series whose steps the span counts."""
    start: int
    """The position, or the start of the interval, as a step ordinal from the series' first stored step."""

    def __post_init__(self) -> None:
        """Validate the base, then the series id and the start ordinal.

        Raises:
            TimeFValidationError: If an abstract base is constructed; if ``time_series_id`` is empty;
                or if ``start`` is not a whole ordinal ``>= 0``.
        """
        super().__post_init__()
        if not self.time_series_id:
            raise TimeFValidationError("StepSpan time_series_id must name one series; got an empty id")
        _check_whole("StepSpan start", self.start, unit="a whole step ordinal")
        if self.start < 0:
            raise TimeFValidationError(f"StepSpan start must be >= 0, got {self.start}")


@dataclass(frozen=True, kw_only=True)
class StepPoint(StepSpan):
    """One step ordinal on the named series."""

    is_point: ClassVar[bool] = True

    @property
    def exclusive_end(self) -> int:
        """One step past the ordinal, so the half-open range covering it is ``[start, +1)``."""
        return self.start + 1


@dataclass(frozen=True, kw_only=True)
class StepInterval(StepSpan):
    """The half-open range ``[start, stop)`` of step ordinals on the named series."""

    stop: int
    """Last step ordinal, exclusive."""

    def __post_init__(self) -> None:
        """Validate the step base, then the stop ordinal.

        Raises:
            TimeFValidationError: If ``stop`` is not a whole ordinal, or is not greater than ``start``.
        """
        super().__post_init__()
        _check_whole("StepInterval stop", self.stop, unit="a whole step ordinal")
        if self.stop <= self.start:
            raise TimeFValidationError(f"StepInterval stop ({self.stop}) must be > start ({self.start})")

    @property
    def exclusive_end(self) -> int:
        """The range's own exclusive end, ``stop``."""
        return self.stop

    @property
    def n_steps(self) -> int:
        """How many steps the interval covers, the horizon ``h`` step-based forecasting libraries speak in."""
        return self.stop - self.start
