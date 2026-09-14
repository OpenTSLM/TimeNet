from datetime import UTC, date, datetime

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd.timeline import US_PER_DAY, Timeline


def test_the_zero_stays_on_the_first_value_when_nothing_precedes_it():
    timeline = Timeline.covering(date(1993, 4, 5), 7, earliest=date(1993, 4, 5))
    assert timeline.start == date(1993, 4, 5)


def test_the_zero_moves_back_by_whole_periods_to_cover_the_earliest_text():
    # 1979-12-31 is 4844 days, 692 whole weeks, before 1993-04-05.
    timeline = Timeline.covering(date(1993, 4, 5), 7, earliest=date(1979, 12, 31))
    assert timeline.start == date(1979, 12, 31)
    assert timeline.offset_us(date(1993, 4, 5)) == 692 * 7 * US_PER_DAY


def test_a_lead_that_is_not_whole_periods_rounds_the_zero_further_back():
    timeline = Timeline.covering(date(2020, 1, 6), 7, earliest=date(2020, 1, 2))
    assert timeline.start == date(2019, 12, 30)


def test_a_daily_cadence_moves_back_by_days():
    timeline = Timeline.covering(date(1980, 1, 1), 1, earliest=date(1979, 12, 31))
    assert timeline.start == date(1979, 12, 31)


def test_a_text_after_the_first_value_moves_nothing():
    timeline = Timeline.covering(date(1980, 1, 1), 1, earliest=date(2001, 1, 1))
    assert timeline.start == date(1980, 1, 1)


def test_the_anchor_is_midnight_utc_of_the_start_day():
    assert Timeline(start=date(1993, 4, 5)).start_time == datetime(1993, 4, 5, tzinfo=UTC)


def test_an_offset_counts_whole_days_from_the_start():
    timeline = Timeline(start=date(2020, 1, 6))
    assert timeline.offset_us(date(2020, 1, 6)) == 0
    assert timeline.offset_us(date(2020, 1, 13)) == 7 * US_PER_DAY


def test_a_day_before_the_start_has_no_offset():
    with pytest.raises(TimeFValidationError, match="precedes the start"):
        Timeline(start=date(2020, 1, 6)).offset_us(date(2020, 1, 5))


def test_an_interval_includes_its_last_day():
    timeline = Timeline(start=date(2020, 1, 6))
    assert timeline.interval(date(2020, 1, 13), date(2020, 1, 17)) == TimeInterval.micros(
        7 * US_PER_DAY, 12 * US_PER_DAY
    )


def test_a_one_day_interval_covers_that_day():
    timeline = Timeline(start=date(2020, 1, 6))
    assert timeline.interval(date(2020, 1, 6), date(2020, 1, 6)) == TimeInterval.micros(0, US_PER_DAY)
