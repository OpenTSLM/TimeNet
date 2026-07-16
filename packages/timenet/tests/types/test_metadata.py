from dataclasses import replace
import pickle

import pytest

from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    ClassificationTask,
    DatasetMetadata,
    DatasetSchema,
    DataSource,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)


def _metadata(**overrides):
    base = DatasetMetadata(
        dataset_id="timenet/hello-world",
        dataset_version=Version(1, 0, 0),
        name="Hello World",
        description="A synthetic demo dataset.",
        license=License.CC_BY_4_0,
    )
    return replace(base, **overrides) if overrides else base


def test_metadata_required_and_defaults():
    m = _metadata()
    assert m.dataset_id == "timenet/hello-world"
    assert m.dataset_version == Version(1, 0, 0)
    assert m.domains == ()
    assert m.tags == ()
    assert m.source_url is None
    assert m.yaml_schema_version == 1


def test_metadata_with_optionals():
    m = _metadata(domains=(Domain.GENERAL,), tags=("demo",), source_url="https://example.org")
    assert m.domains == (Domain.GENERAL,)
    assert m.tags == ("demo",)


def test_metadata_frozen():
    with pytest.raises(AttributeError):
        _metadata().name = "x"


def test_schema_defaults_empty():
    s = DatasetSchema()
    assert s.time_series_specs == ()
    assert s.data_sources == ()
    assert s.annotations == ()
    assert s.tasks == ()


def test_schema_holds_descriptors_and_task_types():
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
    )
    schema = DatasetSchema(
        time_series_specs=(spec,),
        data_sources=(DataSource(data_source_type="holter", name="Holter"),),
        annotations=(AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int"),),
        tasks=(ClassificationTask,),
    )
    assert schema.time_series_specs[0] is spec
    assert schema.tasks == (ClassificationTask,)


def test_schema_rejects_spec_referencing_unknown_data_source():
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
        data_source=DataSource(data_source_type="holter", name="Holter"),
    )
    with pytest.raises(ValueError, match="not in data_sources"):
        DatasetSchema(time_series_specs=(spec,), data_sources=())


def test_schema_rejects_spec_referencing_mismatched_data_source():
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
        data_source=DataSource(data_source_type="holter", name="B"),
    )
    with pytest.raises(ValueError, match="not in data_sources"):
        DatasetSchema(
            time_series_specs=(spec,),
            data_sources=(DataSource(data_source_type="holter", name="A"),),
        )


def test_schema_accepts_spec_with_matching_data_source():
    source = DataSource(data_source_type="holter", name="Holter")
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
        data_source=source,
    )
    schema = DatasetSchema(time_series_specs=(spec,), data_sources=(source,))
    assert schema.time_series_specs[0].data_source == source


def test_schema_rejects_conflicting_data_sources():
    with pytest.raises(ValueError, match="conflicting entries"):
        DatasetSchema(
            data_sources=(
                DataSource(data_source_type="holter", name="A"),
                DataSource(data_source_type="holter", name="B"),
            ),
        )


def test_metadata_picklable():
    assert pickle.loads(pickle.dumps(_metadata())) == _metadata()
