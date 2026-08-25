"""Convert values onto TimeF's microsecond timeline.

TimeF stores every time quantity as a whole number of microseconds. A float changes the
resolution with the magnitude, and two equal offsets can compare as different. These functions
convert a caller's value onto that timeline. They hold the only rounding rule.
"""

from datetime import UTC, datetime, timedelta

from timenet.errors import TimeFValidationError


#: Microseconds in one second.
US_PER_S = 1_000_000

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def check_int64(name: str, value: int) -> None:
    """Reject a value that does not fit the int64 microsecond column TimeF stores it in.

    TimeF stores every time quantity as int64 microseconds. A Python int is unbounded. A value past
    the range wraps silently in numpy, or raises a bare ``OverflowError`` from pyarrow on write. This
    check turns that into a clear failure at the boundary.

    Args:
        name: The field being checked, used in the error message.
        value: The value to range-check.

    Raises:
        TimeFValidationError: If ``value`` is outside the signed 64-bit range.
    """
    if not (INT64_MIN <= value <= INT64_MAX):
        raise TimeFValidationError(f"{name} must fit int64 microseconds, got {value}")


def seconds_to_us(seconds: float) -> int:
    """Round seconds onto the microsecond timeline.

    Args:
        seconds: A time offset or duration in seconds.

    Returns:
        The nearest whole microsecond.
    """
    return round(seconds * US_PER_S)


def us_to_seconds(microseconds: int) -> float:
    """Render microseconds back as seconds.

    This is exact for everything TimeF can store. A value written in seconds reads back equal to
    itself, unless it was finer than a microsecond.

    Args:
        microseconds: A time offset or duration in microseconds.

    Returns:
        The same quantity in seconds.
    """
    return microseconds / US_PER_S


def unix_us(moment: datetime | int) -> int:
    """Normalize a wall-clock timestamp to Unix microseconds.

    This accepts two forms. A timezone-aware :class:`~datetime.datetime` already resolves to
    microseconds, so this changes the origin without rounding. An ``int`` passes through, for a source
    that gives microseconds directly.

    This refuses a float. The value ``1700000000.5`` can mean seconds or microseconds, and the wrong
    reading is off by a factor of a million. If the source gives seconds, wrap it in
    :func:`seconds_to_us` so the unit is visible at the call site.

    Args:
        moment: A timezone-aware datetime, or whole Unix microseconds.

    Returns:
        Unix microseconds.

    Raises:
        TimeFValidationError: If ``moment`` is a float, any other type, or a naive datetime. A naive
            datetime reads in the local zone of the building machine. That anchors the same recording
            differently for each builder.
    """
    if isinstance(moment, (bool, float)):
        raise TimeFValidationError(
            f"anchor must be a timezone-aware datetime or whole Unix microseconds, got {moment!r}. "
            f"A bare number is ambiguous between seconds and microseconds; use seconds_to_us() to say "
            f"which you mean"
        )
    if isinstance(moment, int):
        return moment
    if not isinstance(moment, datetime):
        raise TimeFValidationError(
            f"anchor must be a timezone-aware datetime or whole Unix microseconds, got {moment!r}"
        )
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise TimeFValidationError(
            f"anchor datetime must carry a timezone, got naive {moment!r}. Attach one with "
            f"moment.replace(tzinfo=timezone.utc) if the source is UTC, or the recording site's zone "
            f"if it is local time"
        )
    return (moment - _EPOCH) // timedelta(microseconds=1)


def offset_us(moment: datetime, start_time: datetime | int | None) -> int:
    """Convert a wall-clock moment to an offset on a sample's recording timeline.

    Args:
        moment: The wall-clock moment, timezone-aware.
        start_time: The target sample's ``start_time``.

    Returns:
        Microseconds from the sample's relative zero.

    Raises:
        TimeFValidationError: If ``start_time`` is ``None``. A sample with no wall-clock anchor has
            no calendar time to measure a moment against.
    """
    if start_time is None:
        raise TimeFValidationError(
            "a wall-clock moment needs the target sample's start_time, and that sample has none. A "
            "sample with no wall-clock anchor has no calendar time to measure against; use an offset "
            "into the recording instead"
        )
    return unix_us(moment) - unix_us(start_time)
