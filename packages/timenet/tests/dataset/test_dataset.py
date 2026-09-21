import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Record, Source, TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import SpanOutsideWindowWarning, TimeFValidationError
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
from timenet.values_backends import ValuesBackend


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


def test_add_record_registers_and_returns(make_series):
    ds = _dataset()
    record = ds.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(),)),), subject_ids=("p1",))
    )
    assert isinstance(record, Record)
    assert ds.records == (record,)
    assert record.subject_ids == ("p1",)


def test_add_record_accepts_a_complete_hierarchy(make_series):
    signal = make_series(time_series_id="signal")
    source = Source(id="source", name="Device", signals=(signal,))
    record = Record(record_id="record", sources=(source,))
    dataset = _dataset()

    assert dataset.add_record(record=record) is record
    assert dataset.records == (record,)


def test_add_record_rejects_duplicate_source_ids_inside_one_hierarchy(make_series):
    left = Source(id="sensor", name="Left", signals=(make_series(time_series_id="left"),))
    right = Source(id="sensor", name="Right", signals=(make_series(time_series_id="right"),))
    record = Record(record_id="record", sources=(left, right))

    with pytest.raises(TimeFValidationError, match="duplicate source IDs"):
        _dataset().add_record(record=record)


@pytest.mark.parametrize("values_backend", [ValuesBackend.PARQUET, ValuesBackend.ZARR])
def test_declarative_write_and_open_round_trip(tmp_path, make_series, values_backend):
    dataset = _dataset()
    signal = make_series(time_series_id="signal")
    record = Record(
        record_id="record",
        sources=(Source(id="source", name="Device", signals=(signal,)),),
    )
    dataset.add_record(record=record)

    version_path = dataset.write(path=tmp_path, values_backend=values_backend)
    restored = TimeFDataset.open(path=version_path)

    assert restored.records[0].sources[0].signals[0].to_arrow().equals(signal.to_arrow())
    with pytest.raises(TimeFValidationError, match="each Signal has one owner"):
        restored.add_record(
            record=Record(
                record_id="other-record",
                sources=(Source(id="other-source", name="Other", signals=(signal,)),),
            )
        )


def test_records_property_is_read_only_copy(make_series):
    ds = _dataset()
    ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    assert isinstance(ds.records, tuple)


def test_add_task_links_record_and_task(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = ds.add_task(task=ClassificationTask(inputs=(record,), targets=("afib",)))
    assert task.inputs == (record,)
    assert record.task_ids == (task.id,)
    assert ds.tasks == (task,)


def test_add_task_uses_object_inputs(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = AnswerTask(inputs=(record,), prompt="Alive?", targets=("Yes",))

    assert dataset.add_task(task=task) is task
    assert task.inputs == (record,)


def test_add_task_multiple_records(make_series):
    ds = _dataset()
    s1 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    s2 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = ds.add_task(task=ClassificationTask(inputs=(s1, s2), targets=("x",)))
    assert task.inputs == (s1, s2)


def test_add_task_accepts_no_input_records():
    task = _dataset().add_task(task=ClassificationTask(targets=("x",)))
    assert task.inputs == ()


def test_scope_series_id_resolution(make_series):
    ds = _dataset()
    ts = make_series()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(ts,)),)))
    ds.add_task(
        task=ClassificationTask(
            inputs=(record,),
            targets=("beat",),
            scope=TimePoint.seconds(0.0, time_series_ids=(ts.time_series_id,)),
        )
    )
    with pytest.raises(ValueError, match="unknown time_series_id"):
        ds.add_task(
            task=ClassificationTask(
                inputs=(record,),
                targets=("beat",),
                scope=TimePoint.seconds(0.0, time_series_ids=("nope",)),
            )
        )


def test_from_tasks_on_constructor_registers(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    base = ds.add_task(task=ClassificationTask(inputs=(record,), targets=("a",)))
    derived = ds.add_task(task=AnswerTask(inputs=(record,), prompt="q", targets=("a",), from_tasks=(base,)))
    assert derived.from_tasks == (base,)


def test_add_task_rejects_a_from_tasks_parent_that_is_not_registered(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    orphan = ClassificationTask(targets=("a",))  # never added to the dataset
    qa = AnswerTask(inputs=(record,), prompt="q", targets=("a",), from_tasks=(orphan,))
    with pytest.raises(TimeFValidationError, match="derives from task"):
        ds.add_task(task=qa)


def test_derive_schema(make_series):
    ds = _dataset()
    ts = make_series()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(ts,)),)))
    record.add_annotation(Annotation(key="age", value=64, unit="years"))
    record.add_annotation(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.004)))
    ds.add_task(task=ClassificationTask(inputs=(record,), targets=("afib",)))

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
    # Two signals of the same modality => one distinct spec.
    a = make_series(signal="I")
    b = make_series(signal="II")
    ds.add_record(record=Record(sources=(Source(name="Source", signals=(a, b)),)))
    schema = ds.derive_schema()
    assert len(schema.time_series_specs) == 1


def test_derive_schema_rejects_conflicting_specs_with_same_type(make_series):
    ds = _dataset()
    scalar = make_series(signal="scalar")
    image_spec = TimeSeriesSpec(
        spec_type=scalar.spec.spec_type,
        name=scalar.spec.name,
        unit_value=scalar.spec.unit_value,
        dtype="uint8",
        value_shape=(8, 8, 3),
    )
    image = TimeSeries.from_loader(
        spec=image_spec,
        name="image",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=1,
        loader=lambda: pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((1, 8, 8, 3), dtype="uint8")),
    )
    ds.add_record(record=Record(sources=(Source(name="Source", signals=(scalar, image)),)))
    with pytest.raises(ValueError, match="conflicting TimeSeriesSpec contracts"):
        ds.derive_schema()


def test_derive_schema_rejects_conflicting_annotation_descriptors(make_series):
    ds = _dataset()
    s1 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    s1.add_annotation(Annotation(key="age", value=64))
    s2 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    s2.add_annotation(Annotation(key="age", value="sixty-four"))
    with pytest.raises(ValueError, match="conflicting descriptors"):
        ds.derive_schema()


def test_schema_none_until_derived(make_series):
    ds = _dataset()
    ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
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
    ts = TimeSeries.from_loader(spec=spec, name="c", time_axis=RegularAxis.from_rate_hz(1), n_values=1, loader=loader)
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(ts,)),)))
    ds.add_task(task=ClassificationTask(inputs=(record,), targets=("a",)))
    ds.derive_schema()
    assert calls["n"] == 0  # building/deriving never reads values


def test_add_record_rejects_duplicate_time_series_ids(make_series):
    # TimeSeries uses identity equality with an auto-uuid id, so the same instance twice would
    # silently collapse to one series on write.
    ts = make_series()
    with pytest.raises(TimeFValidationError, match="duplicate signal IDs"):
        _dataset().add_record(record=Record(sources=(Source(name="Source", signals=(ts, ts)),)))


def test_add_record_rejects_a_signal_owned_by_two_records(make_series):
    dataset = _dataset()
    signal = make_series(time_series_id="shared-signal")
    dataset.add_record(
        record=Record(
            record_id="record-a",
            sources=(Source(id="source-a", name="A", signals=(signal,)),),
        )
    )

    with pytest.raises(TimeFValidationError, match="each Signal has one owner"):
        dataset.add_record(
            record=Record(
                record_id="record-b",
                sources=(Source(id="unused-source", name="B", signals=(signal,)),),
            )
        )

    dataset.add_record(
        record=Record(
            record_id="record-c",
            sources=(
                Source(
                    id="unused-source",
                    name="C",
                    signals=(make_series(time_series_id="new-signal"),),
                ),
            ),
        )
    )


def test_add_record_rejects_a_source_owned_by_two_records(make_series):
    dataset = _dataset()
    source = Source(id="shared-source", name="Device", signals=(make_series(),))
    dataset.add_record(record=Record(record_id="record-a", sources=(source,)))

    with pytest.raises(TimeFValidationError, match="each Source has one owner"):
        dataset.add_record(record=Record(record_id="record-b", sources=(source,)))


def test_add_task_warns_for_a_scope_outside_record_span(make_series):
    # Span times are in the source recording timeline, so a window past the series' end warns.
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        dataset.add_task(
            task=ClassificationTask(inputs=(record,), targets=("walking",), scope=TimeInterval.seconds(5.0, 20.0))
        )


def test_add_task_warns_for_a_point_at_the_exclusive_window_end(make_series):
    # 5000 values at 500 Hz is a 10 s window [0, 10); a point at exactly 10.0 s is outside it.
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("walking",), scope=TimePoint.seconds(10.0)))


def test_add_task_accepts_an_interval_up_to_the_exclusive_window_end(make_series):
    # An interval's own end is exclusive too, so [2, 10) fits inside the window [0, 10).
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    task = dataset.add_task(
        task=ClassificationTask(inputs=(record,), targets=("walking",), scope=TimeInterval.seconds(2.0, 10.0))
    )
    assert task.scope == TimeInterval.seconds(2.0, 10.0)


def test_add_task_accepts_scope_inside_record_span(make_series):
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    scope = TimeInterval.seconds(2.0, 8.0)
    task = dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("walking",), scope=scope))
    assert task.scope == scope  # stamped onto the task, so a read-back task is self-describing


def test_add_task_checks_every_span_a_task_carries(make_series):
    # Localization target spans are bounds-checked the same way a scope is.
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        dataset.add_task(
            task=TemporalLocalizationTask(
                inputs=(record,),
                prompt="Locate the onsets.",
                targets=(TimePoint.seconds(2.0), TimePoint.seconds(42.0)),
            )
        )


def test_add_task_requires_an_answer(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    with pytest.raises(TimeFValidationError, match="needs an answer"):
        dataset.add_task(task=ClassificationTask(inputs=(record,)))


def test_add_task_rejects_an_answer_given_twice(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    annotation = record.add_annotation(Annotation(key="stage", value="N2"))
    with pytest.raises(TimeFValidationError, match="one answer representation"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("N2",), target_annotations=(annotation,)))


def test_add_task_accepts_an_answer_stored_by_reference(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    annotation = record.add_annotation(Annotation(key="stage", value="N2", span=TimeInterval.seconds(0.0, 0.004)))
    task = dataset.add_task(
        task=TemporalLocalizationTask(inputs=(record,), prompt="Segment it.", target_annotations=(annotation,))
    )
    assert task.targets is None and task.target_annotations == (annotation,)


def test_add_task_rejects_an_annotation_ref_the_records_do_not_carry(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    detached = Annotation(key="context", value="missing")
    with pytest.raises(TimeFValidationError, match="not attached"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("a",), input_annotations=(detached,)))


def test_register_annotations_rejects_an_inconsistent_duplicate_id():
    dataset = _dataset()
    dataset.register_annotations([Annotation(key="answer_options", value=["yes"], id="opts-0")])
    with pytest.raises(TimeFValidationError, match="registered twice with different values"):
        dataset.register_annotations([Annotation(key="answer_options", value=["no"], id="opts-0")])


def test_set_task_stream_rejects_after_add_task(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("a",)))
    with pytest.raises(TimeFValidationError, match="either streams its tasks or"):
        dataset.set_task_stream([ClassificationTask], lambda: iter(()))


def test_add_task_rejects_on_a_streamed_dataset(make_series):
    dataset = _dataset()
    record = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    dataset.set_task_stream([AnswerTask], lambda: iter(()))
    with pytest.raises(TimeFValidationError, match="streamed dataset"):
        dataset.add_task(task=AnswerTask(inputs=(record,), prompt="q", targets=("a",)))


def test_streamed_task_validation_rejects_an_unknown_record(make_series):
    dataset = _dataset()
    dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),), record_id="s-0"))
    unknown = Record(
        record_id="missing",
        sources=(Source(id="missing-source", name="Missing", signals=(make_series(time_series_id="missing"),)),),
    )
    task = AnswerTask(prompt="q", targets=("a",), inputs=(unknown,))
    dataset.set_task_stream([AnswerTask], lambda: iter((task,)))
    with pytest.raises(TimeFValidationError, match="not registered"):
        list(dataset.iter_streamed_tasks_validated())


def test_streamed_task_validation_rejects_an_undeclared_type(make_series):
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(),)),), record_id="s-0")
    )
    task = ClassificationTask(targets=("a",), inputs=(record,))
    dataset.set_task_stream([AnswerTask], lambda: iter((task,)))  # declared AnswerTask, streamed a different type
    with pytest.raises(TimeFValidationError, match="not one of the declared"):
        list(dataset.iter_streamed_tasks_validated())


def test_streamed_task_validation_does_not_rewalk_registered_signals(make_series, monkeypatch):
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(),)),), record_id="record-0")
    )
    dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(),)),), record_id="unrelated-record")
    )
    task = AnswerTask(prompt="q", targets=("a",), inputs=(record,))
    calls = 0
    original = Record.signals

    def tracked_signals(self):
        nonlocal calls
        calls += 1
        return original.__get__(self, Record)

    monkeypatch.setattr(Record, "signals", property(tracked_signals))
    dataset.set_task_stream([AnswerTask], lambda: iter((task, task)))

    assert list(dataset.iter_streamed_tasks_validated()) == [task, task]
    assert calls == 0


def test_add_task_accepts_a_series_answer_without_a_target(make_series):
    dataset = _dataset()
    context = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    target = dataset.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = dataset.add_task(
        task=ForecastingTask(inputs=(context,), targets=(target,)),
    )
    assert task.targets == (target,)


def test_add_task_accepts_a_span_as_a_forecast_target(make_series):
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
    )
    task = dataset.add_task(
        task=ForecastingTask(
            inputs=(record,),
            targets=(TimeInterval.micros(4000, 6000),),
            scope=TimeInterval.micros(0, 4000),
        )
    )
    assert task.scope == TimeInterval.micros(0, 4000)


def test_tasks_of_filters_by_type(make_series):
    ds = _dataset()
    s = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    classification = ds.add_task(task=ClassificationTask(inputs=(s,), targets=("a",)))
    qa = ds.add_task(task=AnswerTask(inputs=(s,), prompt="q", targets=("b",)))
    assert ds.tasks_of(ClassificationTask) == (classification,)
    assert ds.tasks_of(AnswerTask) == (qa,)


def test_tasks_for_resolves_and_filters_record_tasks(make_series):
    ds = _dataset()
    s1 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    s2 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    classification = ds.add_task(task=ClassificationTask(inputs=(s1,), targets=("a",)))
    qa = ds.add_task(task=AnswerTask(inputs=(s1,), prompt="q", targets=("b",)))
    ds.add_task(task=ClassificationTask(inputs=(s2,), targets=("c",)))
    assert ds.tasks_for(s1) == (classification, qa)
    assert ds.tasks_for(s1, ClassificationTask) == (classification,)
    assert ds.tasks_for(s2, AnswerTask) == ()


def test_tasks_for_unregistered_record_raises(make_series):
    ds = _dataset()
    stranger = _dataset().add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    with pytest.raises(TimeFValidationError, match="not registered"):
        ds.tasks_for(stranger)


def test_tasks_for_non_reciprocal_link_raises(make_series):
    ds = _dataset()
    s1 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    s2 = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = ds.add_task(task=ClassificationTask(inputs=(s1,), targets=("a",)))
    s2.task_ids = (*s2.task_ids, task.id)  # s2 claims the task, but the task does not link back
    with pytest.raises(TimeFValidationError, match="does not link back"):
        ds.tasks_for(s2)


def _matrix(x):
    return x.flatten().to_numpy(zero_copy_only=False).reshape(len(x), -1)


def test_to_features_and_targets_returns_arrow_matrix_and_targets(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_record(
            record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
        )
        ds.add_task(task=ClassificationTask(inputs=(s,), targets=(label,)))
    x, y = ds.to_features_and_targets(task=ClassificationTask)
    assert isinstance(x, pa.Array) and pa.types.is_fixed_size_list(x.type)
    assert isinstance(y, pa.Array)
    assert _matrix(x).tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert y.to_pylist() == ["a", "b"]


def test_to_features_and_targets_numpy_output(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_record(
            record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
        )
        ds.add_task(task=ClassificationTask(inputs=(s,), targets=(label,)))
    x, y = ds.to_features_and_targets(task=ClassificationTask, output="numpy")
    assert isinstance(x, np.ndarray) and x.shape == (2, 3) and x.dtype == np.float32
    assert isinstance(y, np.ndarray) and y.tolist() == ["a", "b"]


def _ragged_dataset(make_series):
    ds = _dataset()
    first = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0)),)),)))
    second = ds.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
    )
    ds.add_task(task=ClassificationTask(inputs=(first,), targets=("a",)))
    ds.add_task(task=ClassificationTask(inputs=(second,), targets=("b",)))
    return ds


def test_to_features_and_targets_timestep_rejects_unequal_lengths(make_series):
    with pytest.raises(ValueError, match="equal-length records"):
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
    ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    with pytest.raises(TimeFValidationError, match="needs exactly one per record"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_unlabeled_record_raises(make_series):
    ds = _dataset()
    labeled = ds.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
    )
    ds.add_task(task=ClassificationTask(inputs=(labeled,), targets=("a",)))
    ds.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
    )  # no task on this one
    with pytest.raises(TimeFValidationError, match="needs exactly one per record"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_multiple_matching_tasks_raises(make_series):
    ds = _dataset()
    s = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    ds.add_task(task=ClassificationTask(inputs=(s,), targets=("a",)))
    ds.add_task(task=ClassificationTask(inputs=(s,), targets=("b",)))
    with pytest.raises(TimeFValidationError, match="needs exactly one per record"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_to_features_and_targets_infers_sole_task_type(make_series):
    ds = _dataset()
    for label in ["a", "b"]:
        s = ds.add_record(
            record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
        )
        ds.add_task(task=ClassificationTask(inputs=(s,), targets=(label,)))
    x, y = ds.to_features_and_targets()  # task inferred: only ClassificationTask present
    assert len(x) == 2
    assert y.to_pylist() == ["a", "b"]


def test_to_features_and_targets_ambiguous_task_type_raises(make_series):
    ds = _dataset()
    s = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    ds.add_task(task=ClassificationTask(inputs=(s,), targets=("a",)))
    ds.add_task(task=AnswerTask(inputs=(s,), prompt="q", targets=("b",)))
    with pytest.raises(ValueError, match="pass task="):
        ds.to_features_and_targets()


def test_to_features_and_targets_keeps_a_scalar_target_numeric(make_series):
    ds = _dataset()
    for value in [1.5, 2.5]:
        s = ds.add_record(
            record=Record(sources=(Source(name="Source", signals=(make_series(values=(1.0, 2.0, 3.0)),)),))
        )
        ds.add_task(task=ScalarPredictionTask(inputs=(s,), targets=(value,), unit="bpm", target_name="rate"))
    _, y = ds.to_features_and_targets(task=ScalarPredictionTask)
    assert pa.types.is_floating(y.type)  # a regression target keeps its type instead of stringifying
    assert y.to_pylist() == pytest.approx([1.5, 2.5])


def test_to_features_and_targets_rejects_a_task_with_no_inline_target(make_series):
    ds = _dataset()
    s = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    annotation = s.add_annotation(Annotation(key="stage", value="N2"))
    ds.add_task(task=ClassificationTask(inputs=(s,), target_annotations=(annotation,)))
    with pytest.raises(ValueError, match="exactly one scalar target"):
        ds.to_features_and_targets(task=ClassificationTask)


def test_add_task_bounds_checks_a_point_span(make_series):
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("beat",), scope=TimePoint.seconds(9.5)))
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("beat",), scope=TimePoint.seconds(10.5)))


def test_annotation_and_task_agree_on_an_out_of_window_span(make_series):
    # The shared window check must treat the same span alike, whether it arrives as an annotation
    # or as a scope.
    dataset = _dataset()
    record = dataset.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    outside = TimePoint.seconds(50.0)
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        record.add_annotation(Annotation(key="mark", span=outside))
    with pytest.warns(SpanOutsideWindowWarning, match="falls outside record"):
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=("x",), scope=outside))


def test_add_annotations_attaches_all_and_returns_them(make_series):
    record = _dataset().add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    anns = record.add_annotations([Annotation(key="a", value=1), Annotation(key="b", value=2)])
    assert tuple(a.key for a in anns) == ("a", "b")
    assert record.annotations == anns


def test_add_annotations_rejects_the_whole_batch_when_one_is_invalid(make_series):
    record = _dataset().add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    good = Annotation(key="a", value=1)
    bad = Annotation(key="b", span=TimePoint.seconds(0.0, time_series_ids=("nope",)))
    with pytest.raises(TimeFValidationError):
        record.add_annotations([good, bad])
    assert record.annotations == ()  # all-or-nothing: the valid one is not left attached


def test_add_tasks_registers_all_in_order_and_links(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    tasks = ds.add_tasks(
        tasks=[
            ClassificationTask(inputs=(record,), targets=("a",)),
            ClassificationTask(inputs=(record,), targets=("b",)),
        ]
    )
    assert tuple(t.targets for t in tasks) == (("a",), ("b",))
    assert ds.tasks == tasks
    assert record.task_ids == tuple(t.id for t in tasks)


def test_add_tasks_rejects_the_whole_batch_when_one_is_invalid(make_series):
    ds = _dataset()
    record = ds.add_record(
        record=Record(sources=(Source(name="Source", signals=(make_series(values=(0.0,) * 5000),)),))
    )
    good = ClassificationTask(inputs=(record,), targets=("a",))
    bad = ClassificationTask(
        inputs=(record,),
        targets=("b",),
        scope=TimePoint.seconds(1.0, time_series_ids=("nope",)),
    )
    with pytest.raises(TimeFValidationError):
        ds.add_tasks(tasks=[good, bad])
    assert ds.tasks == ()  # all-or-nothing: the valid one is not registered either
    assert record.task_ids == ()


def test_add_tasks_allows_deriving_from_another_task_in_the_same_batch(make_series):
    # The deriving task is listed before its parent, so this only passes because the batch is validated
    # as a unit rather than task by task.
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    base = ClassificationTask(inputs=(record,), targets=("a",))
    derived = AnswerTask(inputs=(record,), prompt="q", targets=("a",), from_tasks=(base,))
    registered = ds.add_tasks(tasks=[derived, base])
    assert registered == (derived, base)
    assert derived.from_tasks == (base,)


def test_add_tasks_rejects_duplicate_ids_within_the_batch(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    with pytest.raises(TimeFValidationError, match="share an id"):
        ds.add_tasks(
            tasks=[
                ClassificationTask(inputs=(record,), targets=("a",), id="dup"),
                ClassificationTask(inputs=(record,), targets=("b",), id="dup"),
            ]
        )
    assert ds.tasks == ()


def test_add_tasks_rejects_an_id_already_registered(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    ds.add_task(task=ClassificationTask(inputs=(record,), targets=("a",), id="task-0"))
    with pytest.raises(TimeFValidationError, match="already registered"):
        ds.add_tasks(tasks=[ClassificationTask(inputs=(record,), targets=("b",), id="task-0")])


def test_add_tasks_rejects_a_self_dependency(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    task = AnswerTask(inputs=(record,), prompt="q", targets=("a",))
    task.from_tasks = (task,)  # derives from itself
    with pytest.raises(TimeFValidationError, match="lists itself"):
        ds.add_tasks(tasks=[task])


def test_add_tasks_rejects_a_derivation_cycle(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    first = AnswerTask(inputs=(record,), prompt="q", targets=("a",))
    second = AnswerTask(inputs=(record,), prompt="q", targets=("b",), from_tasks=(first,))
    first.from_tasks = (second,)  # first <- second <- first
    with pytest.raises(TimeFValidationError, match="cyclic"):
        ds.add_tasks(tasks=[first, second])


def test_add_tasks_drains_the_batch_before_checking_refs(make_series):
    # A connector generator may attach an annotation and then yield a task referencing it; draining the
    # batch before the refs are checked means the annotation is already on the record by then.
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))

    def gen():
        ann = record.add_annotation(Annotation(key="peak", span=TimePoint.seconds(0.0)))
        yield ClassificationTask(inputs=(record,), targets=("x",), input_annotations=(ann,))

    (task,) = ds.add_tasks(tasks=gen())
    assert task.input_annotations == (record.annotations[0],)


def test_add_task_rejects_a_time_series_ref_not_on_the_record(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    unknown = make_series(time_series_id="no-such-series")
    with pytest.raises(TimeFValidationError, match="not registered"):
        ds.add_task(task=TSCorrespondenceTask(inputs=(record,), targets=(unknown,)))


def test_add_task_accepts_a_time_series_ref_on_the_record(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    signal = record.signals[0]
    task = ds.add_task(task=TSCorrespondenceTask(inputs=(record,), targets=(signal,)))
    assert task.targets == (signal,)


def test_add_task_accepts_mixed_targets(make_series):
    ds = _dataset()
    record = ds.add_record(record=Record(sources=(Source(name="Source", signals=(make_series(),)),)))
    signal = record.signals[0]
    task = ds.add_task(task=TSCorrespondenceTask(inputs=(record,), targets=(record, signal, "match")))
    assert task.targets == (record, signal, "match")
