import dataclasses
from datetime import UTC, datetime

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import IntervalSpan, PointSpan, Span


def test_interval_and_point():
    interval = IntervalSpan.seconds(5.0, 8.0)
    assert not interval.is_point
    assert PointSpan.seconds(5.0).is_point  # no end => a time offset at start


def test_the_bounds_pick_the_shape():
    assert isinstance(IntervalSpan.seconds(5.0, 8.0), IntervalSpan)
    assert isinstance(PointSpan.seconds(5.0), PointSpan)
    assert isinstance(IntervalSpan.micros(5_000_000, 8_000_000), IntervalSpan)
    assert isinstance(PointSpan.micros(5_000_000), PointSpan)


def test_seconds_rounds_onto_the_microsecond_timeline():
    span = IntervalSpan.seconds(5.0, 8.0)
    assert (span.start, span.end) == (5_000_000, 8_000_000)
    assert PointSpan.seconds(1.2).start == 1_200_000


def test_the_builders_agree():
    assert IntervalSpan.seconds(5.0, 8.0) == IntervalSpan.micros(5_000_000, 8_000_000)


def test_micros_rejects_a_fractional_bound():
    # Seconds are what a caller usually has, and passing them here would be off by a million.
    with pytest.raises(TimeFValidationError, match="whole microseconds"):
        PointSpan.micros(5.5)  # ty: ignore[invalid-argument-type]


def test_an_interval_cannot_be_built_without_its_end():
    # A bare `end: int` would inherit the base's None default and construct clean.
    with pytest.raises(TypeError, match="end"):
        IntervalSpan(start=0)  # ty: ignore[missing-argument]


def test_a_point_and_an_interval_never_compare_equal():
    assert PointSpan(start=5_000_000) != IntervalSpan(start=5_000_000, end=8_000_000)


def test_from_datetime_measures_against_the_sample_anchor():
    anchor = datetime(2026, 8, 5, 9, 0, 0, tzinfo=UTC)
    span = IntervalSpan.from_datetime(
        datetime(2026, 8, 5, 9, 0, 5, tzinfo=UTC),
        datetime(2026, 8, 5, 9, 0, 8, tzinfo=UTC),
        start_time=anchor,
    )
    assert span == IntervalSpan.seconds(5.0, 8.0)


def test_from_datetime_needs_an_anchored_sample():
    # A sample with no wall-clock anchor has no calendar time to measure a moment against.
    with pytest.raises(TimeFValidationError, match="start_time"):
        PointSpan.from_datetime(datetime(2026, 8, 5, tzinfo=UTC), start_time=None)


def test_from_datetime_rejects_a_naive_moment():
    with pytest.raises(TimeFValidationError, match="must carry a timezone"):
        PointSpan.from_datetime(datetime(2026, 8, 5), start_time=1_000_000)


def test_rejects_a_non_positive_interval():
    with pytest.raises(TimeFValidationError, match="must be >"):
        IntervalSpan.seconds(8.0, 5.0)
    with pytest.raises(TimeFValidationError, match="must be >"):
        IntervalSpan.seconds(5.0, 5.0)


def test_rejects_an_explicitly_empty_channel_scope():
    # () would silently mean "no series at all"; None is how you say "every series".
    with pytest.raises(TimeFValidationError, match="non-empty"):
        IntervalSpan.seconds(0.0, 1.0, time_series_ids=())


def test_is_frozen_and_hashable():
    span = IntervalSpan.seconds(0.0, 1.0, time_series_ids=("II",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 2  # ty: ignore[invalid-assignment]
    assert span in {span}


def test_the_base_class_is_not_constructible():
    # Every span in circulation carries the shape it means, so the reader and equality agree.
    with pytest.raises(TimeFValidationError, match="not a shape"):
        Span(start=5_000_000, end=8_000_000)


def test_a_point_with_an_end_is_rejected():
    # Runtime doesn't enforce the field annotations, so the shape has to be checked explicitly.
    with pytest.raises(TimeFValidationError, match="has no end"):
        PointSpan(start=0, end=1_000_000)  # ty: ignore[invalid-argument-type]


def test_an_interval_with_no_end_is_rejected():
    with pytest.raises(TimeFValidationError, match="needs an end"):
        IntervalSpan(start=0, end=None)  # ty: ignore[invalid-argument-type]


def test_a_span_with_no_start_is_rejected():
    with pytest.raises(TimeFValidationError, match="start must be whole microseconds"):
        PointSpan(start=None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bound", [2**63, -(2**63) - 1])
def test_rejects_a_bound_past_int64(bound):
    with pytest.raises(TimeFValidationError, match="fit int64"):
        PointSpan.micros(bound)
