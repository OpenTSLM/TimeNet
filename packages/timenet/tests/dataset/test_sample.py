from datetime import UTC, datetime
from fractions import Fraction

import pytest

from timenet.dataset import Sample, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.dataset.sample import check_span_within_window
from timenet.errors import SpanOutsideWindowWarning, TimeFValidationError
from timenet.types import Annotation, StepInterval, StepPoint, TimeInterval, TimePoint


def test_add_static_annotation(make_series):
    sample = Sample(time_series=(make_series(),))
    ann = sample.add_annotation(Annotation(key="age", value=64))
    assert ann.value == 64
    assert sample.annotations == (ann,)


def test_add_multiple_annotations_preserves_order(make_series):
    sample = Sample(time_series=(make_series(),))
    a = sample.add_annotation(Annotation(key="age", value=64))
    b = sample.add_annotation(Annotation(key="sex", value="M"))
    assert sample.annotations == (a, b)


def test_channel_level_point_resolves_series_id(make_series):
    ts = make_series()
    sample = Sample(time_series=(ts,))
    sample.add_annotation(
        Annotation(key="stimulus", span=TimePoint.seconds(0.002, time_series_ids=(ts.time_series_id,)))
    )


def test_channel_level_annotation_unknown_id_rejected(make_series):
    sample = Sample(time_series=(make_series(),))
    with pytest.raises(ValueError, match="unknown"):
        sample.add_annotation(Annotation(key="stimulus", span=TimePoint.seconds(1.0, time_series_ids=("nope",))))


def test_trial_level_interval_over_differing_windows_is_accepted(make_series):
    # No common-window requirement anymore: an unscoped interval only has to be covered by the union of
    # the series' windows, which here is the contiguous [0, 20) s, so differing lengths do not reject it.
    a = make_series(channel="I", values=(0.0,) * 5000)  # [0, 10) s
    b = make_series(channel="II", values=(0.0,) * 10000)  # [0, 20) s
    sample = Sample(time_series=(a, b))
    sample.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0)))
    assert sample.annotations[0].key == "artifact"


def test_trial_level_interval_common_span_ok(make_series):
    a = make_series(channel="I", values=(0.0,) * 5000)
    b = make_series(channel="II", values=(0.0,) * 5000)
    sample = Sample(time_series=(a, b))
    sample.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0)))


def test_empty_time_series_ids_rejected():
    # The annotation itself rejects `()`, so add_annotation never sees one.
    with pytest.raises(ValueError, match="must be None"):
        Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0, time_series_ids=()))


def test_trial_level_point_needs_no_common_span(make_series):
    a = make_series(channel="I", values=(0.0,) * 5000)
    b = make_series(channel="II", values=(0.0,) * 10000)
    sample = Sample(time_series=(a, b))
    # A point marker imposes no common-span requirement.
    sample.add_annotation(Annotation(key="stimulus", span=TimePoint.seconds(1.0)))


def test_to_numpy_single_channel(make_series):
    sample = Sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    assert sample.to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_to_numpy_rejects_multi_channel(make_series):
    sample = Sample(time_series=(make_series(channel="I"), make_series(channel="II")))
    with pytest.raises(ValueError, match="single-channel"):
        sample.to_numpy()


def test_start_time_defaults_to_none(make_series):
    sample = Sample(time_series=(make_series(),))
    assert sample.start_time is None


def test_start_time_takes_whole_microseconds(make_series):
    anchor = 1_700_000_000_000_001
    sample = Sample(time_series=(make_series(),), start_time=anchor)
    assert sample.start_time == anchor


def test_start_time_takes_an_aware_datetime_and_normalizes_it(make_series):
    moment = datetime(2026, 8, 5, 0, 0, 0, 123456, tzinfo=UTC)
    sample = Sample(time_series=(make_series(),), start_time=moment)
    assert sample.start_time == 1_785_888_000_123_456


@pytest.mark.parametrize("anchor", [2**63, -(2**63) - 1])
def test_start_time_rejects_an_anchor_that_overflows_int64(anchor, make_series):
    with pytest.raises(ValueError, match="int64"):
        Sample(time_series=(make_series(),), start_time=anchor)


@pytest.mark.parametrize("anchor", [True, "1700000000", 1_700_000_000.5])
def test_start_time_rejects_an_ambiguous_anchor(anchor, make_series):
    # A bare float reads as either seconds or microseconds, and the wrong reading is off by a
    # million with nothing downstream to catch it.
    with pytest.raises(ValueError, match="datetime or whole Unix microseconds"):
        Sample(time_series=(make_series(),), start_time=anchor)


def test_time_interval_measures_wall_clock_against_the_sample_anchor(make_series):
    anchor = datetime(2026, 8, 5, 9, 0, 0, tzinfo=UTC)
    sample = Sample(time_series=(make_series(),), start_time=anchor)
    span = sample.time_interval(
        datetime(2026, 8, 5, 9, 0, 5, tzinfo=UTC),
        datetime(2026, 8, 5, 9, 0, 8, tzinfo=UTC),
    )
    assert span == TimeInterval.seconds(5.0, 8.0)


def test_time_point_needs_an_anchored_sample(make_series):
    # A sample with no wall-clock anchor has no calendar time to measure a moment against.
    sample = Sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="start_time"):
        sample.time_point(datetime(2026, 8, 5, tzinfo=UTC))


def test_time_point_rejects_a_naive_moment(make_series):
    sample = Sample(time_series=(make_series(),), start_time=1_000_000)
    with pytest.raises(TimeFValidationError, match="must carry a timezone"):
        sample.time_point(datetime(2026, 8, 5))


def test_start_time_rejects_a_naive_datetime(make_series):
    with pytest.raises(ValueError, match="must carry a timezone"):
        Sample(time_series=(make_series(),), start_time=datetime(2026, 8, 5))


def test_has_absolute_time(make_series):
    assert not Sample(time_series=(make_series(),)).has_absolute_time
    assert Sample(time_series=(make_series(),), start_time=0).has_absolute_time


def test_a_trial_interval_is_refused_on_a_timeless_sample(make_series):
    # An unscoped span needs a timeline to be placed against. An all-ordinal sample has no timed series
    # and no time_span, so there is nothing to check it against and it is refused.
    ordinal = TimeSeries.from_values([1.0, 2.0, 3.0], spec=make_series().spec, channel="c", time_axis=OrdinalAxis())
    sample = Sample(time_series=(ordinal,))
    with pytest.raises(ValueError, match="no timeline to place it"):
        sample.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(1.0, 2.0)))


def test_annotation_span_outside_the_window_is_rejected(make_series):
    # 5000 values at 500 Hz is a 10 s window [0, 10); an interval past it means nothing on the data.
    sample = Sample(time_series=(make_series(values=(0.0,) * 5000),))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        sample.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(5.0, 20.0)), warn_when_outside=False)


def test_annotation_in_a_gap_is_rejected_without_a_time_span(make_series):
    # An event logged between two sensor windows (one sensor off, another not yet on) is rejected by
    # default: the union of the series' windows has a real gap, and nothing says the session spans it.
    early = make_series(channel="early", values=(0.0,) * 5000)  # [0, 10) s
    late = make_series(
        channel="late", values=(0.0,) * 5000, time_axis=RegularAxis(period_us=Fraction(2000), start_index=10_000)
    )  # [20, 30) s
    sample = Sample(time_series=(early, late))
    # 15 s falls in the [10, 20) s gap, inside neither series.
    with pytest.raises(TimeFValidationError, match="falls in a gap"):
        sample.add_annotation(Annotation(key="note", span=TimePoint.seconds(15.0)), warn_when_outside=False)


def test_an_outside_span_warns_and_is_kept(make_series):
    sample = Sample(time_series=(make_series(values=(0.0,) * 5000),))  # a 10 s window [0, 10)
    span = TimeInterval.seconds(5.0, 20.0)
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside sample"):
        sample.add_annotation(Annotation(key="artifact", span=span))
    assert sample.annotations[0].span == span  # kept, not trimmed


def test_annotation_in_a_gap_is_accepted_with_a_time_span(make_series):
    # Declaring a session span that covers the gap says the recording really spanned it (the lunch
    # case): an unscoped span is then checked against the time_span, not the union of series windows.
    early = make_series(channel="early", values=(0.0,) * 5000)  # [0, 10) s
    late = make_series(
        channel="late", values=(0.0,) * 5000, time_axis=RegularAxis(period_us=Fraction(2000), start_index=10_000)
    )  # [20, 30) s
    sample = Sample(time_series=(early, late), time_span=TimeInterval.seconds(0.0, 30.0))
    sample.add_annotation(Annotation(key="note", span=TimePoint.seconds(15.0)))  # inside the session span
    assert sample.annotations[0].key == "note"


def test_a_scoped_span_must_lie_within_the_intersection(make_series):
    # A span scoped to two series claims to apply to both, so it must fall where both recorded: their
    # intersection. [0, 10) s and [5, 15) s intersect on [5, 10) s.
    early = make_series(channel="a", values=(0.0,) * 5000)  # [0, 10) s
    late = make_series(
        channel="b", values=(0.0,) * 5000, time_axis=RegularAxis(period_us=Fraction(2000), start_index=2_500)
    )  # [5, 15) s
    sample = Sample(time_series=(early, late))
    ids = (early.time_series_id, late.time_series_id)
    sample.add_annotation(Annotation(key="ok", span=TimeInterval.seconds(6.0, 8.0, time_series_ids=ids)))  # inside
    # 3 s is inside `early` but not `late`, so it is outside the intersection [5, 10) s.
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        sample.add_annotation(
            Annotation(key="bad", span=TimePoint.seconds(3.0, time_series_ids=ids)), warn_when_outside=False
        )


def test_a_scoped_span_over_non_overlapping_series_is_rejected(make_series):
    early = make_series(channel="a", values=(0.0,) * 5000)  # [0, 10) s
    late = make_series(
        channel="b", values=(0.0,) * 5000, time_axis=RegularAxis(period_us=Fraction(2000), start_index=10_000)
    )  # [20, 30) s
    sample = Sample(time_series=(early, late))
    ids = (early.time_series_id, late.time_series_id)
    with pytest.raises(TimeFValidationError, match="do not overlap"):
        sample.add_annotation(
            Annotation(key="bad", span=TimePoint.seconds(5.0, time_series_ids=ids)), warn_when_outside=False
        )


def test_time_span_must_contain_every_series_window(make_series):
    series = make_series(values=(0.0,) * 5000)  # [0, 10) s
    with pytest.raises(TimeFValidationError, match="must contain every series' window"):
        Sample(time_series=(series,), time_span=TimeInterval.seconds(0.0, 5.0))  # too short


def test_time_span_time_series_ids_must_be_none(make_series):
    with pytest.raises(TimeFValidationError, match="time_series_ids must be None"):
        Sample(time_series=(make_series(),), time_span=TimeInterval.seconds(0.0, 10.0, time_series_ids=("x",)))


def test_time_span_must_be_an_interval(make_series):
    with pytest.raises(TimeFValidationError, match="must be a TimeInterval"):
        Sample(time_series=(make_series(),), time_span=TimePoint.seconds(5.0))  # ty: ignore[invalid-argument-type]


def _ordinal(spec, n, tsid):
    return TimeSeries.from_values(
        [float(i) for i in range(n)], spec=spec, channel="c", time_axis=OrdinalAxis(), time_series_id=tsid
    )


def test_a_steps_span_is_accepted_on_an_ordinal_series(make_series):
    # An ordinal series has no timeline, so a steps span is the only way to name a region of it.
    ts = _ordinal(make_series().spec, 3, "ord")
    check_span_within_window("scope", StepInterval(time_series_id="ord", start=0, stop=3), (ts,), "s")


def test_a_steps_interval_past_the_series_length_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="runs past"):
        check_span_within_window("scope", StepInterval(time_series_id="ord", start=0, stop=4), (ts,), "s")


def test_a_steps_point_on_the_last_step_is_accepted(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    check_span_within_window("scope", StepPoint(time_series_id="ord", start=2), (ts,), "s")


def test_a_steps_point_past_the_last_step_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="runs past"):
        check_span_within_window("scope", StepPoint(time_series_id="ord", start=3), (ts,), "s")


def test_a_steps_span_on_an_unknown_series_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="unknown"):
        check_span_within_window("scope", StepInterval(time_series_id="nope", start=0, stop=2), (ts,), "s")


def test_a_steps_span_on_a_timeline_series_is_rejected(make_series):
    # The axis owns the frame: a timeline series takes a time span, not a steps span, so counting in
    # its steps is refused rather than bounded by length.
    ts = make_series(values=(0.0,) * 5000)
    with pytest.raises(TimeFValidationError, match="has a timeline"):
        check_span_within_window(
            "scope", StepInterval(time_series_id=ts.time_series_id, start=0, stop=5001), (ts,), "s"
        )


def test_a_time_span_on_an_ordinal_series_is_rejected(make_series):
    # The other direction of the axis-fit rule: an ordinal series has no timeline, so a time-valued
    # span means nothing on it.
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="no timeline"):
        check_span_within_window("scope", TimeInterval.seconds(0.0, 1.0, time_series_ids=("ord",)), (ts,), "s")
