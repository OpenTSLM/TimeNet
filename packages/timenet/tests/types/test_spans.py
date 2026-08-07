import dataclasses

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import Span


def test_interval_and_point():
    interval = Span(start_s=5.0, end_s=8.0)
    assert not interval.is_point
    assert Span(start_s=5.0).is_point  # no end => a time offset at start_s


def test_named_constructors_match_direct_construction():
    assert Span.point(5.0, time_series_ids=("ecg",)) == Span(start_s=5.0, time_series_ids=("ecg",))
    assert Span.interval(5.0, 8.0, time_series_ids=("ecg",)) == Span(
        start_s=5.0,
        end_s=8.0,
        time_series_ids=("ecg",),
    )


def test_named_interval_constructor_preserves_validation():
    with pytest.raises(TimeFValidationError, match="must be >"):
        Span.interval(8.0, 5.0)


def test_rejects_a_non_positive_interval():
    with pytest.raises(TimeFValidationError, match="must be >"):
        Span(start_s=8.0, end_s=5.0)
    with pytest.raises(TimeFValidationError, match="must be >"):
        Span(start_s=5.0, end_s=5.0)


def test_rejects_an_explicitly_empty_channel_scope():
    # () would silently mean "no series at all"; None is how you say "every series".
    with pytest.raises(TimeFValidationError, match="non-empty"):
        Span(start_s=0.0, end_s=1.0, time_series_ids=())


def test_is_frozen_and_hashable():
    span = Span(start_s=0.0, end_s=1.0, time_series_ids=("II",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start_s = 2.0  # ty: ignore[invalid-assignment]
    assert span in {span}
