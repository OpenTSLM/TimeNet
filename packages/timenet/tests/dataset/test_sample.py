import pytest

from timenet.dataset import Sample
from timenet.types import IntervalAnnotation, PointAnnotation, StaticAnnotation, View


def test_add_static_annotation(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    ann = sample.add_annotation(StaticAnnotation(key="age", value=64))
    assert ann.value == 64
    assert sample.annotations == (ann,)


def test_add_multiple_annotations_preserves_order(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    a = sample.add_annotation(StaticAnnotation(key="age", value=64))
    b = sample.add_annotation(StaticAnnotation(key="sex", value="M"))
    assert sample.annotations == (a, b)


def test_channel_level_point_resolves_series_id(make_series):
    ts = make_series()
    sample = Sample(time_series=(ts,), view=View.FULL)
    sample.add_annotation(PointAnnotation(key="stimulus", start_time_s=1.0, time_series_ids=(ts.time_series_id,)))


def test_channel_level_annotation_unknown_id_rejected(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    with pytest.raises(ValueError, match="unknown"):
        sample.add_annotation(PointAnnotation(key="stimulus", start_time_s=1.0, time_series_ids=("nope",)))


def test_trial_level_interval_requires_common_span(make_series):
    a = make_series(channel="I", t_start_s=0.0, t_end_s=10.0)
    b = make_series(channel="II", t_start_s=0.0, t_end_s=20.0)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    with pytest.raises(ValueError, match="common"):
        sample.add_annotation(IntervalAnnotation(key="artifact", start_time_s=1.0, end_time_s=2.0))


def test_trial_level_interval_common_span_ok(make_series):
    a = make_series(channel="I", t_start_s=0.0, t_end_s=10.0)
    b = make_series(channel="II", t_start_s=0.0, t_end_s=10.0)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    sample.add_annotation(IntervalAnnotation(key="artifact", start_time_s=1.0, end_time_s=2.0))


def test_empty_time_series_ids_rejected(make_series):
    a = make_series(channel="I", t_start_s=0.0, t_end_s=10.0)
    b = make_series(channel="II", t_start_s=0.0, t_end_s=20.0)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    with pytest.raises(ValueError, match="empty time_series_ids"):
        sample.add_annotation(IntervalAnnotation(key="artifact", start_time_s=1.0, end_time_s=2.0, time_series_ids=()))


def test_trial_level_point_needs_no_common_span(make_series):
    a = make_series(channel="I", t_start_s=0.0, t_end_s=10.0)
    b = make_series(channel="II", t_start_s=0.0, t_end_s=20.0)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    # A point marker imposes no common-span requirement.
    sample.add_annotation(PointAnnotation(key="stimulus", start_time_s=1.0))
