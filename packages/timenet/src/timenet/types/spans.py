"""Span types: one region a task or annotation localizes, in one of two frames.

A region has two independent traits: its shape (a point or a half-open interval) and its frame (time or
steps). The frame changes what the numbers mean, so the frame is the type. Shape is the same trait run
twice, so it is a subtype within each frame.

A time span reads its bounds as microseconds on the **source recording timeline**, the frame a series'
axis places its values in. This keeps it meaningful on a windowed record that starts partway into the
recording. It covers the whole record, or a subset of series given in ``time_series``.

A step span reads its bounds as ordinal indices into one series' own array. A step index means nothing
without a series to count on, so a step span names exactly one series in ``time_series``. Steps exist
for a series that has no timeline at all: an ordinal sequence has positions but no clock.

The series' axis decides which frame fits it, not the caller. A timeline axis (regular or
irregular) takes a time span. An ordinal axis takes a step span.
:meth:`~timenet.dataset.TimeFDataset.add_task` checks a span against the axis of every series it names.

Build a concrete leaf. The bases (:class:`Span`, :class:`TimeSpan`, :class:`StepSpan`) are abstract, so
every span in circulation carries the shape and frame it means::

    TimePoint.seconds(1.2)              TimeInterval.seconds(5.0, 8.0)
    StepPoint(time_series=signal, start=5)
    StepInterval(time_series=signal, start=0, stop=12)

A span holds the :class:`~timenet.dataset.Signal` objects it is scoped to, not their ids. The writer
stores their ids. The reader puts the read-back Signals in their place.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

from timenet.errors import TimeFValidationError
from timenet.types.clock import check_int64, seconds_to_us


if TYPE_CHECKING:
    # A runtime import creates a cycle, because dataset objects import these span types.
    from timenet.dataset.time_series import Signal


def _check_signal(name: str, value: object) -> None:
    """Reject a span scope entry that is not a Signal.

    Raises:
        TimeFValidationError: If ``value`` is not a :class:`~timenet.dataset.Signal`.
    """
    from timenet.dataset.time_series import Signal  # noqa: PLC0415

    if not isinstance(value, Signal):
        raise TimeFValidationError(f"{name} takes Signal objects, got {type(value).__name__}")


def _check_whole(name: str, value: int, *, unit: str, hint: str = "") -> None:
    """Reject a bound that is not a whole int inside the int64 range TimeF stores it in.

    Args:
        name: The field being checked, named in the error message.
        value: The bound to check.
        unit: How the bound reads when it is not whole, such as ``"whole microseconds"``.
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


@dataclass(frozen=True, kw_only=True, repr=False)
class Span:
    """Abstract base for every localized region. Not constructible. Build a concrete leaf type.

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

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "time_series" and value is not None:
                value = tuple(signal.name for signal in value) if isinstance(value, tuple) else value.name
            parts.append(f"{f.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @property
    def exclusive_end(self) -> int:
        """The exclusive upper bound of the span, in its own frame.

        The region is half-open, so a point ends one unit past its position and an interval ends at
        its stored end. Each concrete leaf supplies this.
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True, repr=False)
class TimeSpan(Span):
    """Abstract base for a region on the recording timeline. Bounds are microseconds.

    Covers the whole record when ``time_series`` is ``None``, or a subset of series when it holds
    them. Valid only on a series whose axis is a timeline (regular or irregular).
    """

    frame: ClassVar[str] = "seconds"
    """The frame its bounds read in, as stored on disk."""

    start_us: int
    """The position, or the start of the interval, in microseconds on the source recording timeline."""
    time_series: tuple["Signal", ...] | None = None
    """Series the span is scoped to. ``None`` covers every series in the record."""

    def __post_init__(self) -> None:
        """Validate the base, then the start bound and the series scope.

        Raises:
            TimeFValidationError: If an abstract base is constructed, if ``start_us`` is not whole
                microseconds, if ``time_series`` is ``()`` rather than ``None`` or non-empty, or if it
                holds something that is not a Signal.
        """
        super().__post_init__()
        _check_whole(
            "TimeSpan start_us",
            self.start_us,
            unit="whole microseconds",
            hint="Use seconds() to convert from recording seconds",
        )
        if self.time_series is not None:
            series = tuple(self.time_series)
            if not series:
                raise TimeFValidationError("TimeSpan time_series must be None (whole record) or non-empty, got ()")
            for signal in series:
                _check_signal("TimeSpan time_series", signal)
            object.__setattr__(self, "time_series", series)


@dataclass(frozen=True, kw_only=True, repr=False)
class TimePoint(TimeSpan):
    """One point on the recording timeline."""

    is_point: ClassVar[bool] = True

    @property
    def exclusive_end(self) -> int:
        """One microsecond past the point, so the half-open window covering it is ``[start_us, +1)``."""
        return self.start_us + 1

    @classmethod
    def seconds(cls, at: float, *, time_series: tuple["Signal", ...] | None = None) -> "TimePoint":
        """Construct a point from recording seconds, rounded to the nearest microsecond.

        Args:
            at: The position, in recording seconds.
            time_series: Series the point is scoped to. ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start_us=seconds_to_us(at), time_series=time_series)

    @classmethod
    def micros(cls, at: int, *, time_series: tuple["Signal", ...] | None = None) -> "TimePoint":
        """Construct a point from whole microseconds, for a source that already has them.

        Args:
            at: The position, in microseconds.
            time_series: Series the point is scoped to. ``None`` covers every series.

        Returns:
            The point.
        """
        return cls(start_us=at, time_series=time_series)


@dataclass(frozen=True, kw_only=True, repr=False)
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
    def seconds(cls, start: float, end: float, *, time_series: tuple["Signal", ...] | None = None) -> "TimeInterval":
        """Construct an interval from recording seconds, each bound rounded to the nearest microsecond.

        Args:
            start: Start of the interval, in recording seconds.
            end: End of the interval, exclusive, in recording seconds.
            time_series: Series the interval is scoped to. ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start_us=seconds_to_us(start), end_us=seconds_to_us(end), time_series=time_series)

    @classmethod
    def micros(cls, start: int, end: int, *, time_series: tuple["Signal", ...] | None = None) -> "TimeInterval":
        """Construct an interval from whole microseconds, for a source that already has them.

        Args:
            start: Start of the interval, in microseconds.
            end: End of the interval, exclusive, in microseconds.
            time_series: Series the interval is scoped to. ``None`` covers every series.

        Returns:
            The interval.
        """
        return cls(start_us=start, end_us=end, time_series=time_series)


@dataclass(frozen=True, kw_only=True, repr=False)
class StepSpan(Span):
    """Abstract base for a region counted in the step ordinals of one series.

    A step index means nothing without a series to count on, so a step span names exactly one
    series in ``time_series``. Valid only on a series whose axis is ordinal (no timeline).
    """

    frame: ClassVar[str] = "steps"
    """The frame its bounds read in, as stored on disk."""

    time_series: "Signal"
    """The single series whose steps the span counts."""
    start: int
    """The position, or the start of the interval, as a step ordinal from the series' first stored step."""

    def __post_init__(self) -> None:
        """Validate the base, then the series and the start ordinal.

        Raises:
            TimeFValidationError: If an abstract base is constructed, if ``time_series`` is not a
                Signal, or if ``start`` is not a whole ordinal ``>= 0``.
        """
        super().__post_init__()
        _check_signal("StepSpan time_series", self.time_series)
        _check_whole("StepSpan start", self.start, unit="a whole step ordinal")
        if self.start < 0:
            raise TimeFValidationError(f"StepSpan start must be >= 0, got {self.start}")


@dataclass(frozen=True, kw_only=True, repr=False)
class StepPoint(StepSpan):
    """One step ordinal on the named series."""

    is_point: ClassVar[bool] = True

    @property
    def exclusive_end(self) -> int:
        """One step past the ordinal, so the half-open range covering it is ``[start, +1)``."""
        return self.start + 1


@dataclass(frozen=True, kw_only=True, repr=False)
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
