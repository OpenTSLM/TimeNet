"""A span must come back as the leaf type its frame and bounds describe, or it stops comparing equal."""

import pytest

from timenet.control_plane.spans import span_from_row, span_row
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Span, StepInterval, StepPoint, TimeInterval, TimePoint


def _round_trip(span):
    return span_from_row(*span_row(span))


def test_a_step_interval_round_trips():
    span = StepInterval(time_series_id="s", start=0, stop=12)
    assert _round_trip(span) == span


def test_a_step_point_round_trips():
    span = StepPoint(time_series_id="s", start=5)
    assert _round_trip(span) == span


def test_a_time_interval_keeps_its_scope():
    span = TimeInterval.seconds(1.0, 3.0, time_series_ids=("s",))
    assert _round_trip(span) == span


def test_an_unscoped_time_point_round_trips():
    span = TimePoint.seconds(0.5)
    restored = _round_trip(span)
    assert restored == span
    assert isinstance(restored, TimePoint)


def test_an_unscoped_span_stores_a_null_scope_not_an_empty_one():
    # None covers every series; () is rejected by TimeSpan, so the two must stay distinguishable.
    assert span_row(TimePoint.seconds(0.5))[3] is None
    assert span_row(TimePoint.seconds(0.5, time_series_ids=("s",)))[3] == ["s"]


def test_an_abstract_span_cannot_be_stored():
    with pytest.raises(TimeFValidationError, match="not a concrete span"):
        span_row(Span.__new__(Span))


def test_an_unknown_frame_is_rejected():
    with pytest.raises(TimeFFormatError, match="unknown span frame"):
        span_from_row("furlongs", 1, 3, None)


def test_a_step_span_row_with_no_id_is_rejected():
    # A step span stores the one series it counts on; an empty id list cannot be rebuilt.
    with pytest.raises(TimeFFormatError, match="id list is empty"):
        span_from_row("steps", 0, 12, [])
