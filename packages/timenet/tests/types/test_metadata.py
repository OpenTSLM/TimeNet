import pickle

from pydantic import ValidationError
import pytest

from timenet.manifest.manifest import _metadata_to_dict
from timenet.types import (
    Access,
    AnnotationDescriptor,
    AnnotationType,
    ClassificationTask,
    DatasetMetadata,
    DatasetSchema,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)


def _metadata(**overrides):
    values = {
        "dataset_id": "timenet/hello-world",
        "dataset_version": Version(1, 0, 0),
        "name": "Hello World",
        "description": "A synthetic demo dataset.",
        "license": License.CC_BY_4_0,
    }
    values.update(overrides)
    return DatasetMetadata.model_validate(values)


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
    with pytest.raises(ValidationError):
        _metadata().name = "x"


def test_schema_defaults_empty():
    s = DatasetSchema()
    assert s.time_series_specs == ()
    assert s.annotations == ()
    assert s.tasks == ()


def test_schema_holds_descriptors_and_task_types():
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )
    schema = DatasetSchema(
        time_series_specs=(spec,),
        annotations=(AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int"),),
        tasks=(ClassificationTask,),
    )
    assert schema.time_series_specs[0] == spec
    assert schema.tasks == (ClassificationTask,)


def test_metadata_picklable():
    assert pickle.loads(pickle.dumps(_metadata())) == _metadata()


def test_license_other_requires_a_license_url():
    with pytest.raises(ValidationError, match="license_url is required"):
        _metadata(license=License.OTHER)
    assert _metadata(license=License.OTHER, license_url="https://l.example").license is License.OTHER


def test_non_open_access_requires_an_access_url():
    with pytest.raises(ValidationError, match="access_url is required"):
        _metadata(access=Access.CREDENTIALED)
    ok = _metadata(access=Access.CREDENTIALED, access_url="https://physionet.example/dua")
    assert ok.access is Access.CREDENTIALED


def test_access_defaults_to_open():
    m = _metadata()
    assert m.access is Access.OPEN
    assert m.access_url is None and m.license_url is None and m.citation is None


def test_metadata_round_trips_access_and_license_fields():
    m = _metadata(
        license=License.OTHER,
        license_url="https://l.example",
        citation="Author et al., 2024",
        access=Access.CREDENTIALED,
        access_url="https://physionet.example/dua",
    )
    assert DatasetMetadata.model_validate(_metadata_to_dict(m)) == m


def test_metadata_coerces_string_license():
    m = _metadata(license="MIT")
    assert m.license is License.MIT


def test_metadata_coerces_string_domains():
    m = _metadata(domains=("health", "cardiology"))
    assert m.domains == (Domain.HEALTH, Domain.CARDIOLOGY)


def test_metadata_coerces_string_dataset_version():
    m = _metadata(dataset_version="2.1.0")
    assert m.dataset_version == Version(2, 1, 0)


def test_metadata_coerces_string_access():
    m = _metadata(access="credentialed", access_url="https://physionet.example/dua")
    assert m.access is Access.CREDENTIALED


def test_metadata_rejects_invalid_string_license():
    with pytest.raises(ValidationError, match="license"):
        _metadata(license="not-a-license")


def test_metadata_rejects_invalid_string_domain():
    with pytest.raises(ValidationError, match="domains"):
        _metadata(domains=("not_a_domain",))


def test_metadata_model_validate_defaults_when_access_fields_absent():
    m = DatasetMetadata.model_validate(
        {"dataset_id": "org/name", "dataset_version": "1.0.0", "name": "N", "description": "D", "license": "MIT"}
    )
    assert m.access is Access.OPEN and m.access_url is None and m.citation is None


def test_metadata_model_validate_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="future_field"):
        DatasetMetadata.model_validate(
            {
                "dataset_id": "org/name",
                "dataset_version": "1.0.0",
                "name": "N",
                "description": "D",
                "license": "MIT",
                "future_field": {"value": True},
            }
        )


def test_metadata_model_validate_exposes_pydantic_validation_errors():
    with pytest.raises(ValidationError) as exc_info:
        DatasetMetadata.model_validate(
            {
                "dataset_id": "org/name",
                "dataset_version": "1.0.0",
                "description": "D",
                "license": "MIT",
            }
        )
    assert exc_info.value.errors()[0]["loc"] == ("name",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "   "),
        ("description", ""),
        ("tags", "not-a-list"),
        ("tags", ("valid", 3)),
        ("yaml_schema_version", True),
        ("yaml_schema_version", 1.0),
        ("yaml_schema_version", 2),
    ],
)
def test_metadata_rejects_invalid_field_types_and_values(field, value):
    with pytest.raises(ValidationError, match=field):
        _metadata(**{field: value})
