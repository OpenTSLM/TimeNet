"""How a series' values are placed in time, or that they are not placed at all.

Two shapes today, closed under :data:`TimeAxis`. :class:`RegularAxis` computes every time offset from a
period and an origin, storing nothing per value. :class:`OrdinalAxis` records order and offers no
route to a time offset at all, so a time-valued question about one does not type-check.

Every time offset is an integer microsecond offset from the sample's relative zero. Wall clock enters
once, through the sample's ``start_time``, and composes by addition.

The period is a :class:`~fractions.Fraction` of microseconds rather than a float rate, because not
every real rate is a whole number of them: 360 Hz is 25000/9 us and 256 Hz is 15625/4. A Fraction
keeps the arithmetic exact, so placing a value and locating a time offset are inverse without a
tolerance band to tune, and it reduces and validates itself.
"""

from dataclasses import dataclass, replace
from enum import StrEnum, unique
from fractions import Fraction
import math
from typing import Self

from timenet.errors import TimeFValidationError
from timenet.types.clock import US_PER_S


@unique
class AxisType(StrEnum):
    """Which shape a series' time axis has.

    Stored, and dispatched on before any shape-specific column is read, so the case is never inferred
    from which columns came back null.
    """

    REGULAR = "regular"
    """Time offsets computed from a period and an origin."""
    ORDINAL = "ordinal"
    """Order only: no cadence, no time offsets, no place on any timeline."""


@dataclass(frozen=True, kw_only=True)
class RegularAxis:
    """A constant cadence: value ``k`` sits at ``(start_index + k) * period_us`` microseconds.

    The origin is an index into the cadence rather than a time, because a window rarely starts on a
    whole microsecond. At 44.1 kHz only 3 of 1000 possible window starts do, so a microsecond origin
    would be inexact for almost every window a curator cuts. An index is exact for all of them.
    """

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


@dataclass(frozen=True, kw_only=True)
class OrdinalAxis:
    """Order and nothing more: no cadence, no time offsets, no place on any timeline.

    It has no fields, so there is nowhere to record a rate nobody measured. It also has no
    ``time_offset_us`` and no ``index_at_or_after``, so a caller that has narrowed to this class cannot
    ask a time-valued question at all; the type checker rejects it before the code runs.
    """


TimeAxis = RegularAxis | OrdinalAxis
"""Every shape a series' time axis can have. Closed, so narrowing over it is exhaustive."""
