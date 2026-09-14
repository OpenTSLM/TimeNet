"""Place the calendar dates of the release on the microsecond timeline of a record.

The release states civil dates and no time of day, and every value covers a whole day or a
whole week. TimeF anchors a record at one wall-clock moment and measures everything else in
whole microseconds from it. This module holds the one rule that turns a date into such an
offset, so the series, the annotations and the tasks of a record cannot disagree about where
a day sits.

A record's zero is the midnight, in UTC, that begins its first day. The zone is a convention of
this connector and not a fact of the release, which names none. At a cadence of a day or a
week, another zone moves every value by the same few hours and reorders nothing.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import math

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, TimeInterval


# Microseconds in one civil day. The release states dates, so a day is the finest step it needs.
US_PER_DAY = 24 * 60 * 60 * US_PER_S

ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class Timeline:
    """The calendar day a record's zero falls on, and the arithmetic that places a date on it."""

    start: date

    @classmethod
    def covering(cls, first_value: date, period_days: int, earliest: date) -> "Timeline":
        """Give the timeline whose zero sits on the cadence of the values, at or before ``earliest``.

        The text of a domain can begin years before its first value. A span cannot start before
        a record's zero, so the zero moves back by whole periods until it covers the earliest
        text. The first value then sits a whole number of periods after zero, and the regular
        axis of the series states that number as its ``start_index``.

        Args:
            first_value: The date of the first value of the domain.
            period_days: The cadence of the values, in days.
            earliest: The earliest date anything of the domain states.

        Returns:
            The timeline. Its ``start`` is ``first_value`` when nothing precedes the values.
        """
        behind = max(0, (first_value - earliest).days)
        lead = math.ceil(behind / period_days)
        return cls(start=first_value - timedelta(days=lead * period_days))

    @property
    def start_time(self) -> datetime:
        """The wall-clock anchor of the record: midnight UTC of the start day."""
        return datetime(self.start.year, self.start.month, self.start.day, tzinfo=UTC)

    def offset_us(self, day: date) -> int:
        """Give the time offset of the midnight that begins ``day``.

        Args:
            day: A calendar day at or after the start day.

        Returns:
            Microseconds from the record's zero.

        Raises:
            TimeFValidationError: If ``day`` precedes the start day. A record holds no moment
                before its zero, so such a day has no offset.
        """
        days = (day - self.start).days
        if days < 0:
            raise TimeFValidationError(f"{day.isoformat()} precedes the start of the record, {self.start.isoformat()}")

        return days * US_PER_DAY

    def interval(self, first_day: date, last_day: date) -> TimeInterval:
        """Give the half-open interval that covers the days from ``first_day`` through ``last_day``.

        The release states an inclusive ``end_date``. TimeF states an exclusive end, so the
        interval ends at the midnight that follows the last day.

        Args:
            first_day: The first day the interval covers.
            last_day: The last day it covers, inclusive.

        Returns:
            The interval, which names no series and so covers the whole record.
        """
        return TimeInterval.micros(self.offset_us(first_day), self.offset_us(last_day + ONE_DAY))
