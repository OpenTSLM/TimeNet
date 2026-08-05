from datetime import UTC, datetime

import pytest

from timenet.dataset import Sample
from timenet.types import Annotation, IntervalSpan, PointSpan, View


def test_add_static_annotation(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    ann = sample.add_annotation(Annotation(key="age", value=64))
    assert ann.value == 64
    assert sample.annotations == (ann,)


def test_add_multiple_annotations_preserves_order(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    a = sample.add_annotation(Annotation(key="age", value=64))
    b = sample.add_annotation(Annotation(key="sex", value="M"))
    assert sample.annotations == (a, b)


def test_channel_level_point_resolves_series_id(make_series):
    ts = make_series()
    sample = Sample(time_series=(ts,), view=View.FULL)
    sample.add_annotation(Annotation(key="stimulus", span=PointSpan.seconds(1.0, time_series_ids=(ts.time_series_id,))))


def test_channel_level_annotation_unknown_id_rejected(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    with pytest.raises(ValueError, match="unknown"):
        sample.add_annotation(Annotation(key="stimulus", span=PointSpan.seconds(1.0, time_series_ids=("nope",))))


def test_trial_level_interval_requires_common_span(make_series):
    a = make_series(channel="I", t_start_s=0.0, values=(0.0,) * 5000)
    b = make_series(channel="II", t_start_s=0.0, values=(0.0,) * 10000)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    with pytest.raises(ValueError, match="common"):
        sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0)))


def test_trial_level_interval_common_span_ok(make_series):
    a = make_series(channel="I", t_start_s=0.0, values=(0.0,) * 5000)
    b = make_series(channel="II", t_start_s=0.0, values=(0.0,) * 5000)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    sample.add_annotation(Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0)))


def test_empty_time_series_ids_rejected():
    # The annotation itself rejects `()`, so add_annotation never sees one.
    with pytest.raises(ValueError, match="must be None"):
        Annotation(key="artifact", span=IntervalSpan.seconds(1.0, 2.0, time_series_ids=()))


def test_trial_level_point_needs_no_common_span(make_series):
    a = make_series(channel="I", t_start_s=0.0, values=(0.0,) * 5000)
    b = make_series(channel="II", t_start_s=0.0, values=(0.0,) * 10000)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
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
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    assert sample.start_time is None


def test_start_time_takes_whole_microseconds(make_series):
    anchor = 1_700_000_000_000_001
    sample = Sample(time_series=(make_series(),), view=View.FULL, start_time=anchor)
    assert sample.start_time == anchor


def test_start_time_takes_an_aware_datetime_and_normalizes_it(make_series):
    moment = datetime(2026, 8, 5, 0, 0, 0, 123456, tzinfo=UTC)
    sample = Sample(time_series=(make_series(),), view=View.FULL, start_time=moment)
    assert sample.start_time == 1_785_888_000_123_456


@pytest.mark.parametrize("anchor", [2**63, -(2**63) - 1])
def test_start_time_rejects_an_anchor_that_overflows_int64(anchor, make_series):
    with pytest.raises(ValueError, match="int64"):
        Sample(time_series=(make_series(),), view=View.FULL, start_time=anchor)


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
