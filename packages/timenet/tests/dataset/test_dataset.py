import numpy as np
import pyarrow as pa
import pytest

from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    ClassificationTask,
    DatasetMetadata,
    IntervalAnnotation,
    LabelingTask,
    License,
    QATask,
    ReasoningTask,
    StaticAnnotation,
    TimeSeriesSpec,
    Version,
    View,
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
    sample = ds.add_sample(time_series=(make_series(),), view=View.FULL, subject_ids=("p1",))
    assert isinstance(sample, Sample)
    assert ds.samples == (sample,)
    assert sample.subject_ids == ("p1",)


def test_add_sample_rejects_empty_time_series():
    with pytest.raises(ValueError):
        _dataset().add_sample(time_series=(), view=View.FULL)


def test_samples_property_is_read_only_copy(make_series):
    ds = _dataset()
    ds.add_sample(time_series=(make_series(),), view=View.FULL)
    assert isinstance(ds.samples, tuple)


def test_add_task_links_sample_and_task(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    task = ds.add_task(sample, ClassificationTask(target="afib"))
    assert task.sample_ids == (sample.sample_id,)
    assert sample.task_ids == (task.id,)
    assert ds.tasks == (task,)


def test_add_task_multiple_samples(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    s2 = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    task = ds.add_task((s1, s2), ClassificationTask(target="x"))
    assert set(task.sample_ids) == {s1.sample_id, s2.sample_id}


def test_add_task_rejects_empty_samples():
    with pytest.raises(ValueError):
        _dataset().add_task((), ClassificationTask(target="x"))


def test_labeling_task_id_resolution(make_series):
    ds = _dataset()
    ts = make_series()
    sample = ds.add_sample(time_series=(ts,), view=View.FULL)
    ds.add_task(sample, LabelingTask(target="beat", time_series_ids=(ts.time_series_id,)))
    with pytest.raises(ValueError, match="unknown"):
        ds.add_task(sample, LabelingTask(target="beat", time_series_ids=("nope",)))


def test_from_tasks_via_kwarg(make_series):
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    base = ds.add_task(sample, ClassificationTask(target="a"))
    derived = ds.add_task(sample, ReasoningTask(question="q", target="a"), from_tasks=(base,))
    assert derived.from_tasks == (base,)
    assert derived.from_task_ids == (base.id,)


def test_from_tasks_on_constructor_not_clobbered(make_series):
    # A task built with from_tasks= must not lose it when add_task is called without the kwarg.
    ds = _dataset()
    sample = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    base = ds.add_task(sample, ClassificationTask(target="a"))
    qa = QATask(question="q", target="a", from_tasks=(base,))
    ds.add_task(sample, qa)
    assert qa.from_tasks == (base,)


def test_derive_schema(make_series):
    ds = _dataset()
    ts = make_series()
    sample = ds.add_sample(time_series=(ts,), view=View.FULL)
    sample.add_annotation(StaticAnnotation(key="age", value=64, unit="years"))
    sample.add_annotation(IntervalAnnotation(key="artifact", start_time_s=0.0, end_time_s=1.0))
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
    ds.add_sample(time_series=(a, b), view=View.SUBSET)
    schema = ds.derive_schema()
    assert len(schema.time_series_specs) == 1


def test_derive_schema_rejects_conflicting_annotation_descriptors(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    s1.add_annotation(StaticAnnotation(key="age", value=64))
    s2 = ds.add_sample(time_series=(make_series(),), view=View.FULL)
    s2.add_annotation(StaticAnnotation(key="age", value="sixty-four"))
    with pytest.raises(ValueError, match="conflicting descriptors"):
        ds.derive_schema()


def test_schema_none_until_derived(make_series):
    ds = _dataset()
    ds.add_sample(time_series=(make_series(),), view=View.FULL)
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
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.dimensionless,
    )
    ts = TimeSeries(spec=spec, channel="c", sampling_rate_hz=1.0, loader=loader)
    sample = ds.add_sample(time_series=(ts,), view=View.FULL)
    ds.add_task(sample, ClassificationTask(target="a"))
    ds.derive_schema()
    assert calls["n"] == 0  # building/deriving never reads values


def test_add_sample_defaults_to_full_view(make_series):
    sample = _dataset().add_sample(time_series=(make_series(),))
    assert sample.view is View.FULL


def test_tasks_of_filters_by_type(make_series):
    ds = _dataset()
    s = ds.add_sample(time_series=(make_series(),))
    classification = ds.add_task(s, ClassificationTask(target="a"))
    qa = ds.add_task(s, QATask(question="q", target="b"))
    assert ds.tasks_of(ClassificationTask) == (classification,)
    assert ds.tasks_of(QATask) == (qa,)


def test_tasks_for_resolves_and_filters_sample_tasks(make_series):
    ds = _dataset()
    s1 = ds.add_sample(time_series=(make_series(),))
    s2 = ds.add_sample(time_series=(make_series(),))
    classification = ds.add_task(s1, ClassificationTask(target="a"))
    qa = ds.add_task(s1, QATask(question="q", target="b"))
    ds.add_task(s2, ClassificationTask(target="c"))
    assert ds.tasks_for(s1) == (classification, qa)
    assert ds.tasks_for(s1, ClassificationTask) == (classification,)
    assert ds.tasks_for(s2, QATask) == ()


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
    with pytest.raises(ValueError, match="no sample carries"):
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
    ds.add_task(s, QATask(question="q", target="b"))
    with pytest.raises(ValueError, match="pass task="):
        ds.to_features_and_targets()
