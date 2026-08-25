import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnnotationDescriptor,
    AnnotationType,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    License,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    Version,
    ureg,
)


def _metadata():
    return DatasetMetadata(
        dataset_id="timenet/hello-world",
        dataset_version=Version(1, 0, 0),
        name="Hello World",
        description="demo",
        license=License.CC_BY_4_0,
    )


def _dataset():
    return TimeFDataset(metadata=_metadata())


def test_add_sample_registers_and_returns(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),), subject_ids=("p1",))
    assert isinstance(sample, Sample)
    assert ds.samples == (sample,)
    assert sample.subject_ids == ("p1",)


def test_add_sample_rejects_empty_time_series():
    with pytest.raises(ValueError):
        _dataset().add_sample(time_series=())


def test_samples_property_is_read_only_copy(make_series):
    ds = _dataset()
    ds.add_sample(time_series=(make_series(),))
    assert isinstance(ds.samples, tuple)


def test_add_task_links_sample_and_task(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    task = ds.add_task(sample, ClassificationTask(target="afib"))
    assert task.sample_ids == (sample.sample_id,)
    assert sample.task_ids == (task.id,)
    assert ds.tasks == (task,)


def test_add_task_multiple_samples(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),))
    s2 = ds.add_sample(time_series=(make_series(),))
    task = ds.add_task((s1, s2), ClassificationTask(target="x"))
    assert set(task.sample_ids) == {s1.sample_id, s2.sample_id}


def test_add_task_rejects_empty_samples():
    with pytest.raises(ValueError):
        _dataset().add_task((), ClassificationTask(target="x"))


def test_scope_series_id_resolution(make_series):
    ds = _dataset()
    ts = make_series()
    sample = ds.add_sample(time_series=(ts,))
    ds.add_task(
        sample, ClassificationTask(target="beat", scope=TimePoint.seconds(0.0, time_series_ids=(ts.time_series_id,)))
    )
    with pytest.raises(ValueError, match="unknown time_series_id"):
        ds.add_task(sample, ClassificationTask(target="beat", scope=TimePoint.seconds(0.0, time_series_ids=("nope",))))


def test_from_tasks_on_constructor_registers(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    base = ds.add_task(sample, ClassificationTask(target="a"))
    derived = ds.add_task(sample, AnswerTask(prompt="q", target="a", from_tasks=(base,)))
    assert derived.from_tasks == (base,)
    assert derived.from_task_ids == (base.id,)


def test_add_task_rejects_a_from_tasks_parent_that_is_not_registered(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    orphan = ClassificationTask(target="a")  # never added to the dataset
    qa = AnswerTask(prompt="q", target="a", from_tasks=(orphan,))
    with pytest.raises(TimeFValidationError, match="derives from task"):
        ds.add_task(sample, qa)


def test_derive_schema(make_series):
    ds = _dataset()
    ts = make_series()
    sample = ds.add_sample(time_series=(ts,))
    sample.add_annotation(Annotation(key="age", value=64, unit="years"))
    sample.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.004)))
    ds.add_task(sample, ClassificationTask(target="afib"))

    schema = ds.derive_schema()
    assert schema.time_series_specs == (ts.spec,)
    assert (
        AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int", unit="years")
        in schema.annotations
    )
    assert ClassificationTask in schema.tasks
    assert ds.schema is schema


def test_derive_schema_dedupes_specs(make_series):
    ds = _dataset()
    # Two channels of the same modality => one distinct spec.
    a = make_series(channel="I")
    b = make_series(channel="II")
    ds.add_sample(time_series=(a, b))
    schema = ds.derive_schema()
    assert len(schema.time_series_specs) == 1


def test_derive_schema_rejects_conflicting_specs_with_same_type(make_series):
    ds = _dataset()
    scalar = make_series(channel="scalar")
    image_spec = TimeSeriesSpec(
        spec_type=scalar.spec.spec_type,
        name=scalar.spec.name,
        unit_value=scalar.spec.unit_value,
        dtype="uint8",
        value_shape=(8, 8, 3),
    )
    image = TimeSeries(
        spec=image_spec,
        channel="image",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=1,
        loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((1, 8, 8, 3), dtype="uint8")),
    )
    ds.add_sample(time_series=(scalar, image))
    with pytest.raises(ValueError, match="conflicting TimeSeriesSpec contracts"):
        ds.derive_schema()


def test_derive_schema_rejects_conflicting_annotation_descriptors(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),))
    s1.add_annotation(Annotation(key="age", value=64))
    s2 = ds.add_sample(time_series=(make_series(),))
    s2.add_annotation(Annotation(key="age", value="sixty-four"))
    with pytest.raises(ValueError, match="conflicting descriptors"):
        ds.derive_schema()


def test_schema_none_until_derived(make_series):
    ds = _dataset()
    ds.add_sample(time_series=(make_series(),))
    assert ds.schema is None


def test_no_loader_calls_during_build():
    ds = _dataset()
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return pa.array([1.0], type=pa.float32())

    spec = TimeSeriesSpec(
        spec_type="s",
        name="S",
        unit_value=ureg.dimensionless,
    )
    ts = TimeSeries(spec=spec, channel="c", time_axis=RegularAxis.from_rate_hz(1), n_values=1, loader=loader)
    sample = ds.add_sample(time_series=(ts,))
    ds.add_task(sample, ClassificationTask(target="a"))
    ds.derive_schema()
    assert calls["n"] == 0  # building/deriving never reads values


def test_add_sample_rejects_duplicate_time_series_ids(make_series):
    # TimeSeries uses identity equality with an auto-uuid id, so the same instance twice would
    # silently collapse to one series on write.
    ts = make_series()
    with pytest.raises(TimeFValidationError, match="distinct time_series_ids"):
        _dataset().add_sample(time_series=(ts, ts))


def test_add_task_rejects_scope_outside_sample_span(make_series):
    # Span times are in the source recording timeline, so a window past the series' end is invalid.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        dataset.add_task(sample, ClassificationTask(target="walking", scope=TimeInterval.seconds(5.0, 20.0)))


def test_add_task_rejects_a_point_at_the_exclusive_window_end(make_series):
    # 5000 values at 500 Hz is a 10 s window [0, 10); a point at exactly 10.0 s is outside it.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        dataset.add_task(sample, ClassificationTask(target="walking", scope=TimePoint.seconds(10.0)))


def test_add_task_accepts_an_interval_up_to_the_exclusive_window_end(make_series):
    # An interval's own end is exclusive too, so [2, 10) fits inside the window [0, 10).
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    task = dataset.add_task(sample, ClassificationTask(target="walking", scope=TimeInterval.seconds(2.0, 10.0)))
    assert task.scope == TimeInterval.seconds(2.0, 10.0)


def test_add_task_accepts_scope_inside_sample_span(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    scope = TimeInterval.seconds(2.0, 8.0)
    task = dataset.add_task(sample, ClassificationTask(target="walking", scope=scope))
    assert task.scope == scope  # stamped onto the task, so a read-back task is self-describing


def test_add_task_checks_every_span_a_task_carries(make_series):
    # Localization target spans are bounds-checked the same way a scope is.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        dataset.add_task(
            sample,
            TemporalLocalizationTask(
                prompt="Locate the onsets.", target=(TimePoint.seconds(2.0), TimePoint.seconds(42.0))
            ),
        )


def test_add_task_requires_an_answer(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="needs an answer"):
        dataset.add_task(sample, ClassificationTask())


def test_add_task_rejects_an_answer_given_twice(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    annotation = Annotation(key="stage", value="N2")
    sample.add_annotation(annotation)
    with pytest.raises(TimeFValidationError, match="not both"):
        dataset.add_task(sample, ClassificationTask(target="N2", target_annotation_ids=(annotation.id,)))


def test_add_task_accepts_an_answer_stored_by_reference(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    annotation = Annotation(key="stage", value="N2", span=TimeInterval.seconds(0.0, 0.004))
    sample.add_annotation(annotation)
    task = dataset.add_task(
        sample, TemporalLocalizationTask(prompt="Segment it.", target_annotation_ids=(annotation.id,))
    )
    assert task.target is None and task.target_annotation_ids == (annotation.id,)


def test_add_task_rejects_an_annotation_ref_the_samples_do_not_carry(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="not registered with register_annotations"):
        dataset.add_task(sample, ClassificationTask(target="a", input_annotation_ids=("nope",)))


def test_add_task_accepts_a_registered_annotation_ref(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    options = Annotation(key="answer_options", value=["yes", "no"], id="opts-0")
    dataset.register_annotations([options])
    task = dataset.add_task(sample, AnswerTask(prompt="Q?", target="yes", input_annotation_ids=(options.id,)))
    assert task.input_annotation_ids == ("opts-0",)
    assert dataset.registered_annotations == (options,)  # carried by no sample


def test_register_annotations_rejects_an_inconsistent_duplicate_id():
    dataset = _dataset()
    dataset.register_annotations([Annotation(key="answer_options", value=["yes"], id="opts-0")])
    with pytest.raises(TimeFValidationError, match="registered twice with different values"):
        dataset.register_annotations([Annotation(key="answer_options", value=["no"], id="opts-0")])


def test_set_task_stream_rejects_after_add_task(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    dataset.add_task(sample, ClassificationTask(target="a"))
    with pytest.raises(TimeFValidationError, match="either streams its tasks or"):
        dataset.set_task_stream([ClassificationTask], lambda: iter(()))


def test_add_task_rejects_on_a_streamed_dataset(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(),))
    dataset.set_task_stream([AnswerTask], lambda: iter(()))
    with pytest.raises(TimeFValidationError, match="streamed dataset"):
        dataset.add_task(sample, AnswerTask(prompt="q", target="a"))


def test_streamed_task_validation_rejects_an_unknown_sample(make_series):
    dataset = _dataset()
    dataset.add_sample(time_series=(make_series(),), sample_id="s-0")
    task = AnswerTask(prompt="q", target="a")
    task.sample_ids = ("missing",)
    dataset.set_task_stream([AnswerTask], lambda: iter((task,)))
    with pytest.raises(TimeFValidationError, match="unknown sample"):
        list(dataset.iter_streamed_tasks_validated())


def test_streamed_task_validation_rejects_an_undeclared_type(make_series):
    dataset = _dataset()
    dataset.add_sample(time_series=(make_series(),), sample_id="s-0")
    task = ClassificationTask(target="a")
    task.sample_ids = ("s-0",)
    dataset.set_task_stream([AnswerTask], lambda: iter((task,)))  # declared AnswerTask, streamed a different type
    with pytest.raises(TimeFValidationError, match="not one of the declared"):
        list(dataset.iter_streamed_tasks_validated())


def test_add_task_accepts_a_series_answer_without_a_target(make_series):
    # A forecast's answer is the produced sample, so the target/target_annotation_ids rule does not apply.
    dataset = _dataset()
    context = dataset.add_sample(time_series=(make_series(),))
    target = dataset.add_sample(time_series=(make_series(),))
    task = dataset.add_task(
        context,
        ForecastingTask(context_sample_ids=(context.sample_id,), target_sample_id=target.sample_id),
    )
    assert task.target is None


def test_add_task_accepts_a_target_span_forecast_with_a_context_scope(make_series):
    # A target_span forecast needs a scope to bound its context; the leak check runs against it.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    task = dataset.add_task(
        sample,
        ForecastingTask(target_span=TimeInterval.micros(4000, 6000), scope=TimeInterval.micros(0, 4000)),
    )
    assert task.scope == TimeInterval.micros(0, 4000)


def test_add_task_rejects_a_target_span_forecast_with_no_scope_anywhere(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    with pytest.raises(TimeFValidationError, match="needs an explicit scope"):
        dataset.add_task(sample, ForecastingTask(target_span=TimeInterval.micros(4000, 6000)))


def test_add_task_rejects_a_target_span_forecast_whose_scope_leaks_the_target(make_series):
    # scope [0, 6000) covers the whole target [4000, 6000) once stamped at add_task.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    with pytest.raises(TimeFValidationError, match="end at or before the target"):
        dataset.add_task(
            sample,
            ForecastingTask(target_span=TimeInterval.micros(4000, 6000), scope=TimeInterval.micros(0, 6000)),
        )


def test_tasks_of_filters_by_type(make_series):
    ds = _dataset()
    s = ds.add_sample(time_series=(make_series(),))
    classification = ds.add_task(s, ClassificationTask(target="a"))
    qa = ds.add_task(s, AnswerTask(prompt="q", target="b"))
    assert ds.tasks_of(ClassificationTask) == (classification,)
    assert ds.tasks_of(AnswerTask) == (qa,)


def test_tasks_for_resolves_and_filters_sample_tasks(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),))
    s2 = ds.add_sample(time_series=(make_series(),))
    classification = ds.add_task(s1, ClassificationTask(target="a"))
    qa = ds.add_task(s1, AnswerTask(prompt="q", target="b"))
    ds.add_task(s2, ClassificationTask(target="c"))
    assert ds.tasks_for(s1) == (classification, qa)
    assert ds.tasks_for(s1, ClassificationTask) == (classification,)
    assert ds.tasks_for(s2, AnswerTask) == ()


def test_tasks_for_unregistered_sample_raises(make_series):
    ds = _dataset()
    stranger = _dataset().add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="not registered"):
        ds.tasks_for(stranger)


def test_tasks_for_non_reciprocal_link_raises(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),))
    s2 = ds.add_sample(time_series=(make_series(),))
    task = ds.add_task(s1, ClassificationTask(target="a"))
    s2.task_ids = (*s2.task_ids, task.id)  # s2 claims the task, but the task does not link back
    with pytest.raises(TimeFValidationError, match="does not link back"):
        ds.tasks_for(s2)


def _matrix(x):
    return x.flatten().to_numpy(zero_copy_only=False).reshape(len(x), -1)


def test_to_features_and_targets_returns_arrow_matrix_and_targets(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
        ds.add_task(s, ClassificationTask(target=label))
    x, y = ds.to_features_and_targets(task=ClassificationTask)
    assert isinstance(x, pa.Array) and pa.types.is_fixed_size_list(x.type)
    assert isinstance(y, pa.Array)
    assert _matrix(x).tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert y.to_pylist() == ["a", "b"]


def test_to_features_and_targets_numpy_output(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
        ds.add_task(s, ClassificationTask(target=label))
    x, y = ds.to_features_and_targets(task=ClassificationTask, output="numpy")
    assert isinstance(x, np.ndarray) and x.shape == (2, 3) and x.dtype == np.float32
    assert isinstance(y, np.ndarray) and y.tolist() == ["a", "b"]


def _ragged_dataset(make_series):
    ds = _dataset()
    ds.add_task(ds.add_sample(time_series=(make_series(values=(1.0, 2.0)),)), ClassificationTask(target="a"))
    ds.add_task(ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),)), ClassificationTask(target="b"))
    return ds


def test_to_features_and_targets_timestep_rejects_unequal_lengths(make_series):
    with pytest.raises(ValueError, match="equal-length samples"):
        _ragged_dataset(make_series).to_features_and_targets(task=ClassificationTask)  # features="timestep"


def test_to_features_and_targets_series_arrow_supports_ragged(make_series):
    x, y = _ragged_dataset(make_series).to_features_and_targets(task=ClassificationTask, features="series")
    assert pa.types.is_list(x.type)
    assert x.to_pylist() == [[1.0, 2.0], [1.0, 2.0, 3.0]]
    assert y.to_pylist() == ["a", "b"]


def test_to_features_and_targets_series_numpy_object_array(make_series):
    x, y = _ragged_dataset(make_series).to_features_and_targets(
        task=ClassificationTask, output="numpy", features="series"
    )
    assert x.shape == (2,) and x.dtype == object
    assert x[0].tolist() == [1.0, 2.0]
    assert x[1].tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == ["a", "b"]


def test_to_features_and_targets_no_matching_task_raises(make_series):
    ds = _dataset()
    ds.add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="needs exactly one per sample"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_unlabeled_sample_raises(make_series):
    ds = _dataset()
    labeled = ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
    ds.add_task(labeled, ClassificationTask(target="a"))
    ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))  # no task on this one
    with pytest.raises(TimeFValidationError, match="needs exactly one per sample"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_multiple_matching_tasks_raises(make_series):
    ds = _dataset()
    s = ds.add_sample(time_series=(make_series(),))
    ds.add_task(s, ClassificationTask(target="a"))
    ds.add_task(s, ClassificationTask(target="b"))
    with pytest.raises(TimeFValidationError, match="needs exactly one per sample"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_infers_sole_task_type(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
        ds.add_task(s, ClassificationTask(target=label))
    x, y = ds.to_features_and_targets()  # task inferred: only ClassificationTask present
    assert len(x) == 2
    assert y.to_pylist() == ["a", "b"]


def test_to_features_and_targets_ambiguous_task_type_raises(make_series):
    ds = _dataset()
    s = ds.add_sample(time_series=(make_series(),))
    ds.add_task(s, ClassificationTask(target="a"))
    ds.add_task(s, AnswerTask(prompt="q", target="b"))
    with pytest.raises(ValueError, match="pass task="):
        ds.to_features_and_targets()


def test_to_features_and_targets_keeps_a_scalar_target_numeric(make_series):
    ds = _dataset()
    for value in [1.5, 2.5]:
        s = ds.add_sample(time_series=(make_series(values=(1.0, 2.0, 3.0)),))
        ds.add_task(s, ScalarPredictionTask(target=value, unit="bpm", target_name="rate"))
    _, y = ds.to_features_and_targets(task=ScalarPredictionTask)
    assert pa.types.is_floating(y.type)  # a regression target keeps its type instead of stringifying
    assert y.to_pylist() == pytest.approx([1.5, 2.5])


def test_to_features_and_targets_rejects_a_task_with_no_inline_target(make_series):
    ds = _dataset()
    s = ds.add_sample(time_series=(make_series(),))
    annotation = Annotation(key="stage", value="N2")
    s.add_annotation(annotation)
    ds.add_task(s, ClassificationTask(target_annotation_ids=(annotation.id,)))
    with pytest.raises(ValueError, match="no inline target"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_add_task_bounds_checks_a_point_span(make_series):
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    dataset.add_task(sample, ClassificationTask(target="beat", scope=TimePoint.seconds(9.5)))  # inside
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        dataset.add_task(sample, ClassificationTask(target="beat", scope=TimePoint.seconds(10.5)))


def test_annotation_and_task_agree_on_an_out_of_window_span(make_series):
    # The shared window check must reject the same span whether it arrives as an annotation or a scope.
    dataset = _dataset()
    sample = dataset.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    outside = TimePoint.seconds(50.0)
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        sample.add_annotation(Annotation(key="mark", span=outside))
    with pytest.raises(TimeFValidationError, match="falls outside sample"):
        dataset.add_task(sample, ClassificationTask(target="x", scope=outside))


def test_add_annotations_attaches_all_and_returns_them(make_series):
    sample = _dataset().add_sample(time_series=(make_series(),))
    anns = sample.add_annotations([Annotation(key="a", value=1), Annotation(key="b", value=2)])
    assert tuple(a.key for a in anns) == ("a", "b")
    assert sample.annotations == anns


def test_add_annotations_rejects_the_whole_batch_when_one_is_invalid(make_series):
    sample = _dataset().add_sample(time_series=(make_series(),))
    good = Annotation(key="a", value=1)
    bad = Annotation(key="b", span=TimePoint.seconds(0.0, time_series_ids=("nope",)))
    with pytest.raises(TimeFValidationError):
        sample.add_annotations([good, bad])
    assert sample.annotations == ()  # all-or-nothing: the valid one is not left attached


def test_add_tasks_registers_all_in_order_and_links(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    tasks = ds.add_tasks(sample, [ClassificationTask(target="a"), ClassificationTask(target="b")])
    assert tuple(t.target for t in tasks) == ("a", "b")
    assert ds.tasks == tasks
    assert sample.task_ids == tuple(t.id for t in tasks)


def test_add_tasks_rejects_the_whole_batch_when_one_is_invalid(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(values=(0.0,) * 5000),))
    good = ClassificationTask(target="a")
    bad = ClassificationTask(target="b", scope=TimeInterval.seconds(5.0, 20.0))  # outside the sample span
    with pytest.raises(TimeFValidationError):
        ds.add_tasks(sample, [good, bad])
    assert ds.tasks == ()  # all-or-nothing: the valid one is not registered either
    assert sample.task_ids == ()


def test_add_tasks_allows_deriving_from_another_task_in_the_same_batch(make_series):
    # The deriving task is listed before its parent, so this only passes because the batch is validated
    # as a unit rather than task by task.
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    base = ClassificationTask(target="a")
    derived = AnswerTask(prompt="q", target="a", from_tasks=(base,))
    registered = ds.add_tasks(sample, [derived, base])
    assert registered == (derived, base)
    assert derived.from_tasks == (base,)


def test_add_tasks_rejects_duplicate_ids_within_the_batch(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="share an id"):
        ds.add_tasks(sample, [ClassificationTask(target="a", id="dup"), ClassificationTask(target="b", id="dup")])
    assert ds.tasks == ()


def test_add_tasks_rejects_an_id_already_registered(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    ds.add_task(sample, ClassificationTask(target="a", id="task-0"))
    with pytest.raises(TimeFValidationError, match="already registered"):
        ds.add_tasks(sample, [ClassificationTask(target="b", id="task-0")])


def test_add_tasks_rejects_a_self_dependency(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    task = AnswerTask(prompt="q", target="a")
    task.from_tasks = (task,)  # derives from itself
    with pytest.raises(TimeFValidationError, match="lists itself"):
        ds.add_tasks(sample, [task])


def test_add_tasks_rejects_a_derivation_cycle(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    first = AnswerTask(prompt="q", target="a")
    second = AnswerTask(prompt="q", target="b", from_tasks=(first,))
    first.from_tasks = (second,)  # first <- second <- first
    with pytest.raises(TimeFValidationError, match="cyclic"):
        ds.add_tasks(sample, [first, second])


def test_add_tasks_drains_the_batch_before_checking_refs(make_series):
    # A connector generator may attach an annotation and then yield a task referencing it; draining the
    # batch before the refs are checked means the annotation is already on the sample by then.
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))

    def gen():
        ann = sample.add_annotation(Annotation(key="peak", span=TimePoint.seconds(0.0)))
        yield ClassificationTask(target="x", input_annotation_ids=(ann.id,))

    (task,) = ds.add_tasks(sample, gen())
    assert task.input_annotation_ids == (sample.annotations[0].id,)


def test_add_task_rejects_a_time_series_ref_not_on_the_sample(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    with pytest.raises(TimeFValidationError, match="not on its samples"):
        ds.add_task(sample, TSCorrespondenceTask(target_time_series_ids=("no-such-series",)))


def test_add_task_accepts_a_time_series_ref_on_the_sample(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),))
    series_id = sample.time_series[0].time_series_id
    task = ds.add_task(sample, TSCorrespondenceTask(target_time_series_ids=(series_id,)))
    assert task.target_time_series_ids == (series_id,)
