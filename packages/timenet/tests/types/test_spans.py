import dataclasses

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    Span,
    StepInterval,
    StepPoint,
    StepSpan,
    TimeInterval,
    TimePoint,
    TimeSpan,
)


def test_interval_and_point():
    interval = TimeInterval.seconds(5.0, 8.0)
    assert not interval.is_point
    assert TimePoint.seconds(5.0).is_point  # no end => a point at start


def test_the_bounds_pick_the_shape():
    assert isinstance(TimeInterval.seconds(5.0, 8.0), TimeInterval)
    assert isinstance(TimePoint.seconds(5.0), TimePoint)
    assert isinstance(TimeInterval.micros(5_000_000, 8_000_000), TimeInterval)
    assert isinstance(TimePoint.micros(5_000_000), TimePoint)


def test_seconds_rounds_onto_the_microsecond_timeline():
    span = TimeInterval.seconds(5.0, 8.0)
    assert (span.start_us, span.end_us) == (5_000_000, 8_000_000)
    assert TimePoint.seconds(1.2).start_us == 1_200_000


def test_the_builders_agree():
    assert TimeInterval.seconds(5.0, 8.0) == TimeInterval.micros(5_000_000, 8_000_000)


def test_micros_rejects_a_fractional_bound():
    # Seconds are what a caller usually has, and passing them here would be off by a million.
    with pytest.raises(TimeFValidationError, match="whole microseconds"):
        TimePoint.micros(5.5)  # ty: ignore[invalid-argument-type]


def test_an_interval_cannot_be_built_without_its_end():
    # end_us has no default, so an interval without it never constructs.
    with pytest.raises(TypeError, match="end_us"):
        TimeInterval(start_us=0)  # ty: ignore[missing-argument]


def test_a_point_and_an_interval_never_compare_equal():
    assert TimePoint(start_us=5_000_000) != TimeInterval(start_us=5_000_000, end_us=8_000_000)


def test_rejects_a_non_positive_interval():
    with pytest.raises(TimeFValidationError, match="must be >"):
        TimeInterval.seconds(8.0, 5.0)
    with pytest.raises(TimeFValidationError, match="must be >"):
        TimeInterval.seconds(5.0, 5.0)


def test_rejects_an_explicitly_empty_channel_scope():
    # () would silently mean "no series at all"; None is how you say "every series".
    with pytest.raises(TimeFValidationError, match="non-empty"):
        TimeInterval.seconds(0.0, 1.0, time_series_ids=())


def test_a_multi_id_time_scope_is_kept():
    span = TimeInterval.seconds(0.0, 1.0, time_series_ids=("I", "II"))
    assert span.time_series_ids == ("I", "II")


def test_is_frozen_and_hashable():
    span = TimeInterval.seconds(0.0, 1.0, time_series_ids=("II",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start_us = 2  # ty: ignore[invalid-assignment]
    assert span in {span}


def test_the_abstract_bases_are_not_constructible():
    # Every span in circulation carries the shape and frame it means.
    with pytest.raises(TimeFValidationError, match="abstract span base"):
        Span()
    with pytest.raises(TimeFValidationError, match="abstract span base"):
        TimeSpan(start_us=5_000_000)
    with pytest.raises(TimeFValidationError, match="abstract span base"):
        StepSpan(time_series_id="s", start=0)


def test_a_span_with_no_start_is_rejected():
    with pytest.raises(TimeFValidationError, match="whole microseconds"):
        TimePoint(start_us=None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bound", [2**63, -(2**63) - 1])
def test_rejects_a_bound_past_int64(bound):
    with pytest.raises(TimeFValidationError, match="fit int64"):
        TimePoint.micros(bound)


@pytest.mark.parametrize("bound", [2**63, -(2**63) - 1])
def test_a_step_bound_past_int64_is_rejected(bound):
    # Step ordinals share the int64 column time bounds use, so the same range applies.
    with pytest.raises(TimeFValidationError, match="fit int64"):
        StepInterval(time_series_id="s", start=0, stop=bound)


def test_step_builders_construct_both_shapes():
    interval = StepInterval(time_series_id="s", start=0, stop=12)
    assert isinstance(interval, StepSpan)
    assert (interval.start, interval.stop) == (0, 12)
    assert not interval.is_point
    point = StepPoint(time_series_id="s", start=5)
    assert isinstance(point, StepSpan)
    assert point.is_point


def test_a_step_span_names_one_series():
    assert StepInterval(time_series_id="s", start=0, stop=12).time_series_id == "s"
    assert StepPoint(time_series_id="s", start=5).time_series_id == "s"


def test_n_steps_is_the_horizon_of_a_step_interval():
    assert StepInterval(time_series_id="s", start=132, stop=144).n_steps == 12


def test_n_steps_lives_only_on_a_step_interval():
    # A seconds interval counts no steps, and a point spans no steps.
    assert not hasattr(TimeInterval.seconds(0.0, 12.0), "n_steps")
    assert not hasattr(StepPoint(time_series_id="s", start=5), "n_steps")


def test_a_step_span_must_name_the_series_it_counts_on():
    # Step 5 is a different region on every series with a different rate, offset or length.
    with pytest.raises(TimeFValidationError, match="must name one series"):
        StepInterval(time_series_id="", start=0, stop=12)


def test_a_step_span_rejects_a_negative_start():
    with pytest.raises(TimeFValidationError, match="must be >= 0"):
        StepInterval(time_series_id="s", start=-1, stop=12)


def test_a_step_span_rejects_a_fractional_step():
    with pytest.raises(TimeFValidationError, match="whole step"):
        StepInterval(time_series_id="s", start=0, stop=5.5)  # ty: ignore[invalid-argument-type]


def test_a_step_span_rejects_a_non_positive_interval():
    with pytest.raises(TimeFValidationError, match="must be >"):
        StepInterval(time_series_id="s", start=12, stop=12)
    with pytest.raises(TimeFValidationError, match="must be >"):
        StepInterval(time_series_id="s", start=12, stop=0)


def test_a_step_span_and_a_time_span_never_compare_equal():
    # The frame is what keeps 8 steps from colliding with 8 microseconds read back off disk.
    steps = StepInterval(time_series_id="s", start=0, stop=8)
    micros = TimeInterval.micros(0, 8)
    assert steps != micros
