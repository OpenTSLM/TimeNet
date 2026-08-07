"""How a series' values are placed in time, or that they are not placed at all.

Three shapes, closed under :data:`TimeAxis`. :class:`RegularAxis` computes every time offset from a
period and an origin, storing nothing per value. :class:`IrregularAxis` covers the placements no
formula produces, so every time offset is written down beside the values. :class:`OrdinalAxis` records
order and offers no route to a time offset at all, so a time-valued question about one does not
type-check.

Two words, deliberately not interchangeable, because the format has two different things to name:

**time offset**
    A position on a series' own axis: an integer microsecond offset from the sample's relative zero.
    Every axis quantity here is one, as are a span's bounds. A time offset says where a value sits
    within its recording and nothing about what day that was. A series with no anchor has time offsets
    and no timestamps, which is the whole of case 4.

**timestamp**
    An absolute point on the wall clock, in Unix microseconds. Exactly one thing carries one:
    :attr:`~timenet.dataset.Sample.start_time`. It is what a sample's relative zero refers to.

Wall clock therefore enters once and composes by addition: a value's timestamp is the sample's
``start_time`` plus the value's time offset. One is measured from the recording's own zero and the
other from the Unix epoch, so the split is this format's, stated here and applied consistently rather
than inferred from the word.

The period is a :class:`~fractions.Fraction` of microseconds rather than a float rate, because not
every real rate is a whole number of them: 360 Hz is 25000/9 us and 256 Hz is 15625/4. A Fraction
keeps the arithmetic exact, so placing a value and locating a time offset are inverse without a
tolerance band to tune, and it reduces and validates itself.
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
from timenet.types.clock import US_PER_S, offset_us


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@unique
class AxisType(StrEnum):
    """Which shape a series' time axis has.

    Stored, and dispatched on before any shape-specific column is read, so the case is never inferred
    from which columns came back null.
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

    The origin is an index into the cadence rather than a time, because a window rarely starts on a
    whole microsecond. At 44.1 kHz only 3 of 1000 possible window starts do, so a microsecond origin
    would be inexact for almost every window a curator cuts. An index is exact for all of them.
    """

    axis_type: ClassVar[AxisType] = AxisType.REGULAR
    """The stored discriminator. Declared on each shape rather than derived by a dispatch function,
    so a new shape cannot be added without giving itself a tag."""
    period_us: Fraction
    """Microseconds between values. A :class:`~fractions.Fraction` because not every real rate is a
    whole number of microseconds: 360 Hz is 25000/9 and 256 Hz is 15625/4, the ECG and EEG rates a
    health corpus runs into. It is stored as its numerator and denominator."""
    start_index: int = 0
    """Index of this series' first value on the cadence. Non-zero for a window cut into a longer
    recording, which is what keeps a span written against the recording meaningful on the window."""

    def __post_init__(self) -> None:
        """Reject a period of the wrong type or too fine, or a non-integer or negative origin.

        Raises:
            TimeFValidationError: If ``period_us`` is not a :class:`~fractions.Fraction` or is under
                one microsecond, or ``start_index`` is not a non-negative integer.
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
        if isinstance(self.start_index, bool) or not isinstance(self.start_index, int):
            raise TimeFValidationError(f"RegularAxis.start_index must be an integer, got {self.start_index!r}")
        if self.start_index < 0:
            raise TimeFValidationError(f"RegularAxis.start_index must be >= 0, got {self.start_index}")

    @classmethod
    def from_rate_hz(cls, rate_hz: int | Fraction) -> Self:
        """Build the axis of a regularly sampled series from its exact rate.

        A float is not accepted. Every real sampling rate is a whole number of values per second, so
        ``500.0`` should be written ``500``, and a rate that genuinely is not whole has no single
        reading: 29.97 fps is 2997/100 by its spelling and 30000/1001 by its intent, and those drift
        3.6 ms apart over an hour. Say which with a :class:`~fractions.Fraction`, and build it from a
        string rather than a float, because ``Fraction(29.97)`` is the binary expansion
        (1054475631502295/35184372088832) while ``Fraction("29.97")`` is 2997/100.

        Args:
            rate_hz: Values per second.

        Returns:
            The axis whose period is one sampling period of ``rate_hz``.

        Raises:
            TimeFValidationError: If ``rate_hz`` is not a positive integer or ``Fraction``.
        """
        # bool is an int subclass, so it would otherwise slip through as 1 Hz; a float carries no
        # single exact reading (29.97 fps is 2997/100 or 30000/1001), so it is refused too.
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, int | Fraction):
            raise TimeFValidationError(
                f"RegularAxis.from_rate_hz needs an integer or Fraction rate, got {rate_hz!r}. Write "
                f"a whole rate as an int and a non-whole one as a Fraction built from a string"
            )
        if rate_hz <= 0:
            raise TimeFValidationError(f"RegularAxis.from_rate_hz needs a positive rate, got {rate_hz!r}")
        return cls(period_us=Fraction(US_PER_S) / Fraction(rate_hz))

    def at_index(self, index: int) -> Self:
        """Return the axis of a window starting at ``index`` values into this one.

        Exact at every rate, because the origin it moves is an index rather than a derived time.

        Args:
            index: How many values into this axis the window starts.

        Returns:
            The window's axis, sharing this period.
        """
        return replace(self, start_index=self.start_index + index)

    def time_offset_us(self, index: int) -> int:
        """Return the time offset of one value, floored to whole microseconds.

        Floor here pairs with the ceiling in :meth:`index_at_or_after`, and the two are exactly
        inverse at every period of one microsecond or coarser, including the non-integral ones.

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

    The strict gate every irregular stream passes through. It refuses the two inputs that would
    otherwise be wrong by a constant factor with nothing downstream to notice.

    A ``datetime64`` array is refused rather than converted. ``pandas.DatetimeIndex.values`` is
    ``datetime64[ns]``, and reading it as int64 yields nanoseconds, so every time offset lands a
    thousandfold out while staying a plausible-looking number. Floats are refused for the same reason
    seconds and microseconds are both readings of ``1.5``.

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
    # Before the dtype check: an empty list is float64 by numpy default, and reporting that as a
    # float-vs-microseconds problem would name the wrong defect.
    if array.size == 0:
        raise TimeFValidationError("time offsets must hold at least one time offset")
    if array.dtype.kind not in {"i", "u"}:
        raise TimeFValidationError(f"time offsets must be whole microseconds, got dtype {array.dtype}")
    # A uint64 value at or past 2**63 wraps to a negative int64 on the cast below, so range-check the
    # unsigned case first. Signed numpy ints are all int64 or narrower, so they cannot overflow it.
    if array.dtype.kind == "u" and array.size and int(array.max()) > _INT64_MAX:
        raise TimeFValidationError(
            f"time offsets must fit int64 microseconds, got a value past {_INT64_MAX}; a uint64 at or "
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

    The safe path from calendar time, and the reason :func:`to_time_offsets_us` refuses a ``datetime64``
    array outright. Each moment is measured against the sample's anchor, so the result is in the same
    frame as a span's bounds and a regular axis' computed time offsets.

    Args:
        moments: The wall-clock moments, each timezone-aware.
        start_time: The target sample's ``start_time``.

    Returns:
        A C-contiguous int64 array of microseconds from the sample's relative zero.

    :func:`~timenet.types.clock.offset_us` raises if ``start_time`` is ``None``, since a sample with
    no wall-clock anchor has no calendar time to measure against.
    """
    return to_time_offsets_us(np.fromiter((offset_us(m, start_time) for m in moments), dtype=np.int64))


@dataclass(frozen=True, kw_only=True)
class IrregularAxis:
    """A placement no formula produces, so every time offset is written down beside the values.

    What lives here is only the pair a curator can state and the writer can verify without a read: the
    first and the last stored time offset. The time offsets themselves ride the values plane and are reached
    through :attr:`~timenet.dataset.TimeSeries.time_offsets_us`.

    That split is the point. Two ints compare and hash, so the axis still goes whole into the writer's
    series identity and still round-trips as a value through the samples struct. An axis holding the
    array could do neither: a tuple comparison against an ndarray field raises rather than answering.

    The endpoints are metadata about the stream, not an identity for it. Two series whose time offsets
    differ only in the middle carry equal axes, so nothing may use axis equality to conclude the
    time offsets agree; compare the streams.

    There is deliberately no ``time_offset_us`` and no ``index_at_or_after``. Both need a read, and a read
    is a series-level operation. :class:`OrdinalAxis` sets the precedent: an axis that cannot answer in
    constant time does not offer the method.
    """

    axis_type: ClassVar[AxisType] = AxisType.IRREGULAR
    """The stored discriminator."""
    first_us: int
    """Time offset of the first value, in microseconds from the sample's relative zero."""
    last_us: int
    """Time offset of the last value. Checked against the stream itself at write time, so it is verified
    metadata rather than an unbacked claim."""

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
            if not (_INT64_MIN <= value <= _INT64_MAX):
                raise TimeFValidationError(f"IrregularAxis.{name} must fit int64 microseconds, got {value}")
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
