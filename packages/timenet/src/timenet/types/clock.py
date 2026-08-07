"""Conversions onto TimeF's microsecond timeline.

Every time quantity TimeF stores is a whole number of microseconds. Storing a float would make the
resolution change with magnitude and two equal time offsets need not compare equal, which is not
something a format should leave to its callers. These are the conversions that get a caller's value
onto that timeline, and they are the only place the rounding rule lives.
"""

from datetime import UTC, datetime, timedelta

from timenet.errors import TimeFValidationError


#: Microseconds in one second.
US_PER_S = 1_000_000

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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

    Exact for everything TimeF can store, so a value written in seconds reads back equal to itself
    unless it was finer than a microsecond to begin with.

    Args:
        microseconds: A time offset or duration in microseconds.

    Returns:
        The same quantity in seconds.
    """
    return microseconds / US_PER_S


def unix_us(moment: datetime | int) -> int:
    """Normalize a wall-clock timestamp to Unix microseconds.

    Accepts the two forms a curator actually has. A timezone-aware :class:`~datetime.datetime` is the
    common one, and it already resolves to microseconds, so this is a change of origin rather than a
    rounding. An ``int`` passes through, for a source that hands over microseconds directly.

    A float is refused. Seconds and microseconds are both plausible readings of ``1700000000.5``, and
    the wrong one is off by a factor of a million with nothing downstream to catch it. When the source
    really does give seconds, wrap it in :func:`seconds_to_us` so the unit is visible at the call site.

    Args:
        moment: A timezone-aware datetime, or whole Unix microseconds.

    Returns:
        Unix microseconds.

    Raises:
        TimeFValidationError: If ``moment`` is a float or any other type, or is a naive datetime. A
            naive datetime would be read in the curating machine's local zone, anchoring the same
            recording differently depending on who curated it.
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
        TimeFValidationError: If ``start_time`` is ``None``, because a sample with no wall-clock
            anchor has no calendar time to measure a moment against.
    """
    if start_time is None:
        raise TimeFValidationError(
            "a wall-clock moment needs the target sample's start_time, and that sample has none. A "
            "sample with no wall-clock anchor has no calendar time to measure against; use an offset "
            "into the recording instead"
        )
    return unix_us(moment) - unix_us(start_time)
