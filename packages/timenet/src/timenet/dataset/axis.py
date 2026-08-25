"""Define how a series places its values in time, or state that it does not place them.

TimeF has three axis shapes, closed under :data:`TimeAxis`. :class:`RegularAxis` computes every time
offset from a period and an origin. It stores nothing per value. :class:`IrregularAxis` covers the
placements that no formula produces. It writes down every time offset beside the values.
:class:`OrdinalAxis` records order only. It gives no route to a time offset, so a time question about
one does not type-check.

This module uses two words that are not interchangeable, because the format names two different things:

**time offset**
    A position on a series' own axis. It is an integer microsecond offset from the sample's relative
    zero. Every axis quantity here is a time offset, and so are a span's bounds. A time offset says
    where a value sits within its recording. It says nothing about the calendar day. A series with no
    anchor has time offsets and no timestamps.

**timestamp**
    An absolute point on the wall clock, in Unix microseconds. Exactly one field carries one:
    :attr:`~timenet.dataset.Sample.start_time`. It is what a sample's relative zero refers to.

The wall clock enters once and composes by addition. A value's timestamp is the sample's
``start_time`` plus the value's time offset. The recording's own zero anchors the time offset.
The Unix epoch anchors the timestamp.

The period is a :class:`~fractions.Fraction` of microseconds, not a float rate. Not every real
rate is a whole number of them: 360 Hz is 25000/9 us and 256 Hz is 15625/4. A Fraction keeps the
arithmetic exact, so placing a value and locating a time offset are inverse without a tolerance band.
A Fraction also reduces and validates itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum, unique
from fractions import Fraction
import math
from typing import ClassVar, Self

import numpy as np

from timenet.errors import TimeFValidationError
from timenet.types.clock import INT64_MAX, US_PER_S, check_int64, offset_us


@unique
class AxisType(StrEnum):
    """Name which shape a series' time axis has.

    TimeF stores this value and dispatches on it before it reads any shape-specific column. TimeF
    never infers the case from which columns came back null.
    """

    REGULAR = "regular"
    """Time offsets computed from a period and an origin."""
    IRREGULAR = "irregular"
    """One stored time offset per value."""
    ORDINAL = "ordinal"
    """Order only: no cadence, no time offsets, no place on any timeline."""


@dataclass(frozen=True, kw_only=True)
class RegularAxis:
    """A constant cadence: value ``k`` sits at ``(start_index + k) * period_us`` microseconds.

    The origin is an index into the cadence, not a time, because a window rarely starts on a whole
    microsecond. At 44.1 kHz only 3 of 1000 window starts do, so a microsecond origin is inexact for
    almost every window. An index is exact for all of them.
    """

    axis_type: ClassVar[AxisType] = AxisType.REGULAR
    """The stored discriminator. Each shape declares its own tag, so a new shape must declare one
    too."""
    period_us: Fraction
    """Microseconds between values. It is a :class:`~fractions.Fraction` because not every real rate is
    a whole number of microseconds: 360 Hz is 25000/9 and 256 Hz is 15625/4. TimeF stores it as its
    numerator and denominator."""
    start_index: int = 0
    """Index of this series' first value on the cadence. It is non-zero for a window cut from a longer
    recording, which keeps a span written against the recording meaningful on the window."""

    def __post_init__(self) -> None:
        """Reject a period of the wrong type or too fine, or a non-integer or negative origin.

        Raises:
            TimeFValidationError: If ``period_us`` is not a :class:`~fractions.Fraction`, is under one
                microsecond, or has a term past int64. Also if ``start_index`` is not a non-negative
                integer that fits int64.
        """
        if not isinstance(self.period_us, Fraction):
            raise TimeFValidationError(
                f"RegularAxis.period_us must be a Fraction, got {type(self.period_us).__name__}. A "
                f"float loses the exact rate; build the axis with from_rate_hz() or pass a Fraction"
            )
        if self.period_us < 1:
            raise TimeFValidationError(
                f"RegularAxis.period_us is {self.period_us} us, finer than the one microsecond the "
                f"format can address. The highest rate it can carry is 1 MHz"
            )
        # TimeF stores the period as a numerator/denominator pair of int64 columns. It is >= 1, so
        # the reduced numerator is the larger term. A range check on the numerator covers the denominator.
        check_int64("RegularAxis.period_us", self.period_us.numerator)
        if isinstance(self.start_index, bool) or not isinstance(self.start_index, int):
            raise TimeFValidationError(f"RegularAxis.start_index must be an integer, got {self.start_index!r}")
        if self.start_index < 0:
            raise TimeFValidationError(f"RegularAxis.start_index must be >= 0, got {self.start_index}")
        check_int64("RegularAxis.start_index", self.start_index)

    @classmethod
    def from_rate_hz(cls, rate_hz: int | Fraction) -> Self:
        """Build the axis of a regularly sampled series from its exact rate.

        This method does not accept a float. Every real sampling rate is a whole number of values per
        second, so write ``500.0`` as ``500``. A rate that is not whole has no single reading: 29.97
        fps is 2997/100 by its spelling and 30000/1001 by its intent. Those two drift 3.6 ms apart
        over an hour. State which one with a :class:`~fractions.Fraction`, and build the Fraction from
        a string, not a float. ``Fraction(29.97)`` is the binary expansion
        (1054475631502295/35184372088832), and ``Fraction("29.97")`` is 2997/100.

        Args:
            rate_hz: Values per second.

        Returns:
            The axis whose period is one sampling period of ``rate_hz``.

        Raises:
            TimeFValidationError: If ``rate_hz`` is not a positive integer or ``Fraction``.
        """
        # bool is an int subclass, so without this guard it slips through as 1 Hz. A float carries no
        # single exact reading (29.97 fps is 2997/100 or 30000/1001), so this method refuses it too.
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, int | Fraction):
            raise TimeFValidationError(
                f"RegularAxis.from_rate_hz needs an integer or Fraction rate, got {rate_hz!r}. Write "
                f"a whole rate as an int and a non-whole one as a Fraction built from a string"
            )
        if rate_hz <= 0:
            raise TimeFValidationError(f"RegularAxis.from_rate_hz needs a positive rate, got {rate_hz!r}")
        return cls(period_us=Fraction(US_PER_S) / Fraction(rate_hz))

    def at_index(self, index: int) -> Self:
        """Return the axis of a window that starts ``index`` values into this one.

        This is exact at every rate, because the origin it moves is an index, not a derived time.

        Args:
            index: How many values into this axis the window starts.

        Returns:
            The window's axis, which shares this period.
        """
        return replace(self, start_index=self.start_index + index)

    def time_offset_us(self, index: int) -> int:
        """Return the time offset of one value, floored to whole microseconds.

        This floor pairs with the ceiling in :meth:`index_at_or_after`. The two are exactly inverse at
        every period of one microsecond or coarser, including the non-integral ones.

        Args:
            index: The value's index within this series.

        Returns:
            Microseconds from the sample's relative zero.
        """
        return math.floor((self.start_index + index) * self.period_us)

    def index_at_or_after(self, time_offset_us: int) -> int:
        """Return the first value at or after a time offset.

        Args:
            time_offset_us: The time offset, in microseconds from the sample's relative zero.

        Returns:
            The index within this series, which is negative if the time offset precedes its first value.
        """
        return math.ceil(time_offset_us / self.period_us) - self.start_index


def to_time_offsets_us(time_offsets_us: np.ndarray | Sequence[int]) -> np.ndarray:
    """Normalize a stream of per-value time offsets to int64 microseconds.

    Every irregular stream passes through this strict gate. It refuses the two inputs that are wrong
    by a constant factor, with nothing downstream to notice.

    This function refuses a ``datetime64`` array, it does not convert it. ``pandas.DatetimeIndex.values``
    is ``datetime64[ns]``, and reading it as int64 gives nanoseconds, so every time offset lands a
    thousandfold out but still looks plausible. It refuses floats for the same reason: ``1.5`` reads as
    both seconds and microseconds.

    Args:
        time_offsets_us: The per-value time offsets, in microseconds from the sample's relative zero.

    Returns:
        A C-contiguous int64 array.

    Raises:
        TimeFValidationError: If the stream is empty, not integral, or not non-decreasing.
    """
    array = np.asarray(time_offsets_us)
    if array.dtype.kind == "M":
        raise TimeFValidationError(
            f"time offsets must be whole microseconds, got a datetime64 array ({array.dtype}). Reading it "
            f"as int64 would give its own unit, not microseconds; convert with "
            f"time_offsets_from_datetimes(moments, start_time=...) instead"
        )
    if array.dtype.kind == "f" and array.size:
        raise TimeFValidationError(
            f"time offsets must be whole microseconds, got a float array ({array.dtype}). A bare number is "
            f"ambiguous between seconds and microseconds; use seconds_to_us() to say which you mean"
        )
    if array.ndim != 1:
        raise TimeFValidationError(f"time offsets must be one time offset per value, got shape {array.shape}")
    # This runs before the dtype check. An empty list is float64 by numpy default, and a
    # float-vs-microseconds error names the wrong defect.
    if array.size == 0:
        raise TimeFValidationError("time offsets must hold at least one time offset")
    if array.dtype.kind not in {"i", "u"}:
        raise TimeFValidationError(f"time offsets must be whole microseconds, got dtype {array.dtype}")
    # A uint64 value at or past 2**63 wraps to a negative int64 on the cast below, so range-check the
    # unsigned case first. Signed numpy ints are all int64 or narrower, so they cannot overflow it.
    if array.dtype.kind == "u" and array.size and int(array.max()) > INT64_MAX:
        raise TimeFValidationError(
            f"time offsets must fit int64 microseconds, got a value past {INT64_MAX}; a uint64 at or "
            f"above 2**63 would wrap to a negative int64"
        )
    time_offsets = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(np.diff(time_offsets) < 0):
        first = int(np.argmax(np.diff(time_offsets) < 0))
        raise TimeFValidationError(
            f"time offsets must be non-decreasing; time offset {first + 1} ({time_offsets[first + 1]}) precedes "
            f"time offset {first} ({time_offsets[first]})"
        )
    return time_offsets


def time_offsets_from_datetimes(moments: Sequence[datetime], *, start_time: datetime | int | None) -> np.ndarray:
    """Convert wall-clock moments to time offsets on a sample's recording timeline.

    This is the safe path from calendar time, and the reason :func:`to_time_offsets_us` refuses a
    ``datetime64`` array outright. This function measures each moment against the sample's anchor.
    The result lands in the same frame as a span's bounds and a regular axis' computed time offsets.

    Args:
        moments: The wall-clock moments, each timezone-aware.
        start_time: The target sample's ``start_time``.

    Returns:
        A C-contiguous int64 array of microseconds from the sample's relative zero.

    :func:`~timenet.types.clock.offset_us` raises if ``start_time`` is ``None``, because a sample with
    no wall-clock anchor has no calendar time to measure against.
    """
    return to_time_offsets_us(np.fromiter((offset_us(m, start_time) for m in moments), dtype=np.int64))


@dataclass(frozen=True, kw_only=True)
class IrregularAxis:
    """A placement no formula produces, so this axis writes down every time offset beside the values.

    This axis holds only the pair a builder can state and the writer can verify without a read. That
    pair is the first and the last stored time offset. The time offsets themselves ride the values plane. Reach
    them through :attr:`~timenet.dataset.TimeSeries.time_offsets_us`.

    That split is the point. Two ints compare and hash, so the axis goes whole into the writer's series
    identity and round-trips as a value through the samples struct. An axis holding the array does
    neither: a tuple comparison against an ndarray field raises instead of answering.

    The endpoints are metadata about the stream, not an identity for it. Two series whose time offsets
    differ only in the middle carry equal axes. Nothing can use axis equality to conclude the time
    offsets agree. Compare the streams instead.

    This axis has no ``time_offset_us`` and no ``index_at_or_after`` on purpose. Both need a read, and a
    read is a series-level operation. :class:`OrdinalAxis` sets the precedent: an axis that cannot
    answer in constant time does not offer the method.
    """

    axis_type: ClassVar[AxisType] = AxisType.IRREGULAR
    """The stored discriminator."""
    first_us: int
    """Time offset of the first value, in microseconds from the sample's relative zero."""
    last_us: int
    """Time offset of the last value. The writer checks it against the stream at write time, so it is
    verified metadata, not an unbacked claim."""

    def __post_init__(self) -> None:
        """Reject non-integral or backwards endpoints.

        Raises:
            TimeFValidationError: If either endpoint is not whole microseconds or does not fit int64,
                if ``first_us`` is negative, or if ``last_us`` precedes ``first_us``.
        """
        for name in ("first_us", "last_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TimeFValidationError(f"IrregularAxis.{name} must be whole microseconds, got {value!r}")
            check_int64(f"IrregularAxis.{name}", value)
        if self.first_us < 0:
            raise TimeFValidationError(
                f"IrregularAxis.first_us must be >= 0, got {self.first_us}; a time offset is measured "
                f"from the sample's relative zero"
            )
        if self.last_us < self.first_us:
            raise TimeFValidationError(
                f"IrregularAxis runs backwards: first_us={self.first_us}, last_us={self.last_us}"
            )

    @classmethod
    def spanning(cls, time_offsets_us: np.ndarray | Sequence[int]) -> Self:
        """Build the axis describing a stream of time offsets.

        Args:
            time_offsets_us: The per-value time offsets, in microseconds from the sample's relative zero.

        Returns:
            The axis carrying that stream's endpoints, after :func:`to_time_offsets_us` has vetted it.
        """
        time_offsets = to_time_offsets_us(time_offsets_us)
        return cls(first_us=int(time_offsets[0]), last_us=int(time_offsets[-1]))


@dataclass(frozen=True, kw_only=True)
class OrdinalAxis:
    """An Axis to indicate an order without a cadence, time offsets, or place on any timeline."""

    axis_type: ClassVar[AxisType] = AxisType.ORDINAL
    """The stored discriminator."""


TimeAxis = RegularAxis | IrregularAxis | OrdinalAxis
"""Every shape a series' time axis can have."""
