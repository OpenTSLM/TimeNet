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


def test_add_annotations_attaches_a_batch_in_order(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    batch = [StaticAnnotation(key="age", value=64), StaticAnnotation(key="sex", value="M")]
    assert sample.add_annotations(batch) == tuple(batch)
    assert sample.annotations == tuple(batch)


def test_add_annotations_accepts_a_generator(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    attached = sample.add_annotations(StaticAnnotation(key=key, value=1) for key in ("a", "b"))
    assert [ann.key for ann in attached] == ["a", "b"]


def test_add_annotations_with_nothing_is_a_no_op(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    assert sample.add_annotations([]) == ()
    assert sample.annotations == ()


def test_add_annotations_rejects_a_bad_one_in_a_batch(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    good = StaticAnnotation(key="age", value=64)
    bad = PointAnnotation(key="stimulus", start_time_s=1.0, time_series_ids=("nope",))
    with pytest.raises(ValueError, match="unknown"):
        sample.add_annotations([good, bad])
    assert sample.annotations == (good,)  # attached as it went, so the valid prefix stays


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


def test_empty_time_series_ids_rejected():
    # The annotation itself rejects `()`, so add_annotations never sees one.
    with pytest.raises(ValueError, match="must be None"):
        IntervalAnnotation(key="artifact", start_time_s=1.0, end_time_s=2.0, time_series_ids=())


def test_trial_level_point_needs_no_common_span(make_series):
    a = make_series(channel="I", t_start_s=0.0, t_end_s=10.0)
    b = make_series(channel="II", t_start_s=0.0, t_end_s=20.0)
    sample = Sample(time_series=(a, b), view=View.SUBSET)
    # A point marker imposes no common-span requirement.
    sample.add_annotation(PointAnnotation(key="stimulus", start_time_s=1.0))


def test_to_numpy_single_channel(make_series):
    sample = Sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    assert sample.to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_to_numpy_rejects_multi_channel(make_series):
    sample = Sample(time_series=(make_series(channel="I"), make_series(channel="II")))
    with pytest.raises(ValueError, match="single-channel"):
        sample.to_numpy()


def test_t0_unix_ns_defaults_to_none(make_series):
    sample = Sample(time_series=(make_series(),), view=View.FULL)
    assert sample.t0_unix_ns is None


def test_t0_unix_ns_accepts_int64(make_series):
    anchor = 1_700_000_000_000_000_001
    sample = Sample(time_series=(make_series(),), view=View.FULL, t0_unix_ns=anchor)
    assert sample.t0_unix_ns == anchor


def test_t0_unix_ns_rejects_out_of_int64_range(make_series):
    with pytest.raises(ValueError, match="int64"):
        Sample(time_series=(make_series(),), view=View.FULL, t0_unix_ns=2**63)
    with pytest.raises(ValueError, match="int64"):
        Sample(time_series=(make_series(),), view=View.FULL, t0_unix_ns=-(2**63) - 1)


@pytest.mark.parametrize("anchor", [True, 1.5, "1700000000000000000"])
def test_t0_unix_ns_rejects_non_integer(anchor, make_series):
    with pytest.raises(ValueError, match="integer or None"):
        Sample(time_series=(make_series(),), t0_unix_ns=anchor)


def test_has_absolute_time(make_series):
    assert not Sample(time_series=(make_series(),)).has_absolute_time
    assert Sample(time_series=(make_series(),), t0_unix_ns=0).has_absolute_time
