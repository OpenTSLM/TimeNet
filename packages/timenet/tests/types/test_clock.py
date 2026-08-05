from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, seconds_to_us, unix_us, us_to_seconds


def test_seconds_round_to_the_nearest_microsecond():
    assert seconds_to_us(1.5) == 1_500_000
    assert seconds_to_us(1e-7) == 0  # finer than a microsecond
    assert us_to_seconds(1_500_000) == pytest.approx(1.5)


def test_utc_datetime_matches_its_timestamp():
    moment = datetime(2026, 8, 5, 12, 0, 0, 123456, tzinfo=UTC)
    assert unix_us(moment) == round(moment.timestamp() * US_PER_S)


@pytest.mark.parametrize("zone", ["Europe/Vienna", "America/New_York", "Australia/Lord_Howe"])
@pytest.mark.parametrize("month", [1, 8])
def test_a_zone_observing_dst_anchors_on_the_offset_in_force(zone, month):
    # Subtracting an epoch built with the same tzinfo object makes datetime compare wall clocks and
    # drop both offsets, which lands a summer time offset a whole DST shift away from where it belongs.
    moment = datetime(2026, month, 5, 12, 0, 0, 123456, tzinfo=ZoneInfo(zone))
    assert unix_us(moment) == round(moment.timestamp() * US_PER_S)


def test_whole_microseconds_pass_through():
    assert unix_us(1785931200123456) == 1785931200123456


def test_rejects_a_naive_datetime():
    with pytest.raises(TimeFValidationError, match="must carry a timezone"):
        unix_us(datetime(2026, 8, 5, 12, 0))


@pytest.mark.parametrize("value", [1.5, 1700000000.0, True, "1700000000", None])
def test_rejects_anything_that_is_not_a_datetime_or_whole_microseconds(value):
    with pytest.raises(TimeFValidationError, match="anchor must be"):
        unix_us(value)
