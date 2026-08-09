from datetime import UTC, datetime
from fractions import Fraction

import pytest

from timenet.dataset import Sample, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.dataset.sample import check_span_within_window
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, IntervalSpan, PointSpan


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
        Annotation(key="stimulus", span=PointSpan.seconds(0.002, time_series_ids=(ts.time_series_id,)))
    )


def test_channel_level_annotation_unknown_id_rejected(make_series):
    sample = Sample(time_series=(make_series(),))
    with pytest.raises(ValueError, match="unknown"):
        sample.add_annotation(Annotation(key="stimulus", span=PointSpan.seconds(1.0, time_series_ids=("nope",))))


def test_trial_level_interval_requires_common_span(make_series):
    a = make_series(channel="I", values=(0.0,) * 5000)
    b = make_series(channel="II", values=(0.0,) * 10000)
    sample = Sample(time_series=(a, b))
    with pytest.raises(ValueError, match="common"):
        sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0)))


def test_trial_level_interval_common_span_ok(make_series):
    a = make_series(channel="I", values=(0.0,) * 5000)
    b = make_series(channel="II", values=(0.0,) * 5000)
    sample = Sample(time_series=(a, b))
    sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0)))


def test_empty_time_series_ids_rejected():
    # The annotation itself rejects `()`, so add_annotation never sees one.
    with pytest.raises(ValueError, match="must be None"):
        Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0, time_series_ids=()))


def test_trial_level_point_needs_no_common_span(make_series):
    a = make_series(channel="I", values=(0.0,) * 5000)
    b = make_series(channel="II", values=(0.0,) * 10000)
    sample = Sample(time_series=(a, b))
    # A point marker imposes no common-span requirement.
    sample.add_annotation(Annotation(key="stimulus", span=PointSpan.seconds(1.0)))


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


def test_start_time_rejects_a_naive_datetime(make_series):
    with pytest.raises(ValueError, match="must carry a timezone"):
        Sample(time_series=(make_series(),), start_time=datetime(2026, 8, 5))


def test_has_absolute_time(make_series):
    assert not Sample(time_series=(make_series(),)).has_absolute_time
    assert Sample(time_series=(make_series(),), start_time=0).has_absolute_time


def test_a_trial_interval_is_refused_on_a_timeless_sample(make_series):
    # add_task refuses this span; add_annotation must not accept it. An ordinal series reports
    # span_us None, so an all-ordinal sample collapses to a single distinct "window" of None, which
    # the common-window rule would otherwise read as agreement.
    ordinal = TimeSeries.from_values([1.0, 2.0, 3.0], spec=make_series().spec, channel="c", time_axis=OrdinalAxis())
    sample = Sample(time_series=(ordinal,))
    with pytest.raises(ValueError, match="no timeline at all"):
        sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0)))


def test_annotation_span_outside_the_window_is_rejected(make_series):
    # 5000 values at 500 Hz is a 10 s window [0, 10); an interval past it means nothing on the data.
    sample = Sample(time_series=(make_series(values=(0.0,) * 5000),))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(5.0, 20.0)))


def test_annotation_in_a_gap_between_disjoint_windows_is_accepted(make_series):
    # The bounds check uses the sample's overall span, so an event logged between two sensor windows
    # (one sensor off, another not yet on) is a valid annotation rather than an error.
    early = make_series(channel="early", values=(0.0,) * 5000)  # [0, 10) s
    late = make_series(
        channel="late", values=(0.0,) * 5000, time_axis=RegularAxis(period_us=Fraction(2000), start_index=10_000)
    )  # [20, 30) s
    sample = Sample(time_series=(early, late))
    # 15 s falls in the [10, 20) s gap, inside neither series but within the sample's overall span.
    sample.add_annotation(Annotation(key="note", span=PointSpan.seconds(15.0)))


def _ordinal(spec, n, tsid):
    return TimeSeries.from_values(
        [float(i) for i in range(n)], spec=spec, channel="c", time_axis=OrdinalAxis(), time_series_id=tsid
    )


def test_a_steps_span_is_accepted_on_an_ordinal_series(make_series):
    # An ordinal series has no timeline, so a steps span is the only way to name a region of it.
    ts = _ordinal(make_series().spec, 3, "ord")
    check_span_within_window("scope", IntervalSpan.steps(0, 3, time_series_ids=("ord",)), (ts,), "s")


def test_a_steps_interval_past_the_series_length_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="runs past"):
        check_span_within_window("scope", IntervalSpan.steps(0, 4, time_series_ids=("ord",)), (ts,), "s")


def test_a_steps_point_on_the_last_step_is_accepted(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    check_span_within_window("scope", PointSpan.steps(2, time_series_ids=("ord",)), (ts,), "s")


def test_a_steps_point_past_the_last_step_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="runs past"):
        check_span_within_window("scope", PointSpan.steps(3, time_series_ids=("ord",)), (ts,), "s")


def test_a_steps_span_on_an_unknown_series_is_rejected(make_series):
    ts = _ordinal(make_series().spec, 3, "ord")
    with pytest.raises(TimeFValidationError, match="unknown"):
        check_span_within_window("scope", IntervalSpan.steps(0, 2, time_series_ids=("nope",)), (ts,), "s")


def test_a_steps_span_on_a_timeline_series_is_bounded_by_count_not_window(make_series):
    # 5000 steps sit inside a 5000-value series; a window check would read 5001 as 5001 us, far inside
    # the 10 s window, and wrongly accept it. Steps bound by length.
    ts = make_series(values=(0.0,) * 5000)
    with pytest.raises(TimeFValidationError, match="runs past"):
        check_span_within_window("scope", IntervalSpan.steps(0, 5001, time_series_ids=(ts.time_series_id,)), (ts,), "s")
