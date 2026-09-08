import jsonschema
import pytest

from timenet.manifest import FilePart, Manifest, ManifestCounts, ManifestFiles
from timenet.schemas import MANIFEST_SCHEMA
from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    AnswerTask,
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


def _manifest() -> Manifest:
    source = DataSource(data_source_type="holter", name="Holter Monitor", provider="Acme")
    spec = TimeSeriesSpec(
        spec_type="ecg",
        name="ECG",
        unit_value=ureg.millivolt,
        data_source=source,
    )
    rhythm = TimeSeriesSpec(
        spec_type="rhythm",
        name="Rhythm",
        unit_value=ureg.dimensionless,
        dtype="str",
    )
    schema = DatasetSchema(
        time_series_specs=(spec, rhythm),
        annotations=(
            AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int", unit="years"),
            AnnotationDescriptor(key="artifact", annotation_type=AnnotationType.INTERVAL),
        ),
        tasks=(ClassificationTask, AnswerTask),
    )
    metadata = DatasetMetadata(
        dataset_id="demo/ecg",
        dataset_version=Version(1, 2, 0),
        name="ECG",
        description="demo",
        license=License.CC_BY_4_0,
        domains=(Domain.CARDIOLOGY,),
        tags=("demo",),
    )
    return Manifest(
        dataset_id="demo/ecg",
        metadata=metadata,
        schema=schema,
        counts=ManifestCounts(
            records=2,
            annotations=4,
            tasks={"classification": 2},
            time_series_chunks=3,
            time_series_index_rows=3,
            time_series_specs={"ecg": 2},
        ),
        files=ManifestFiles(
            records=(FilePart("records.parquet", "sha256:" + "a" * 64, 10),),
            annotations=(FilePart("annotations.parquet", "sha256:" + "b" * 64, 20),),
            time_series_index=(FilePart("time_series_index.parquet", "sha256:" + "c" * 64, 30),),
            tasks=(FilePart("tasks/task=classification/part-0.parquet", "sha256:" + "d" * 64, 40),),
            time_series=(FilePart("time_series/part-00000.parquet", "sha256:" + "e" * 64, 50),),
        ),
    )


def test_schema_is_well_formed():
    jsonschema.Draft202012Validator.check_schema(MANIFEST_SCHEMA)


def test_to_dict_validates_against_schema():
    jsonschema.validate(_manifest().to_dict(), MANIFEST_SCHEMA)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("metadata"),  # missing required top-level block
        lambda d: d.update(timef_format_version=99),  # not the pinned const
        lambda d: d["schema"].update(tasks=[{"task_type": "nope"}]),  # unknown task_type
        lambda d: d["schema"]["annotations"][0].update(annotation_type="sideways"),  # bad annotation_type
        lambda d: d["files"].update(records="single.parquet"),  # a bare string, not a list of parts
        lambda d: d["metadata"].update(license="Nope"),  # unknown license
    ],
)
def test_invalid_manifests_are_rejected(mutate):
    data = _manifest().to_dict()
    mutate(data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, MANIFEST_SCHEMA)


def test_every_serialized_key_is_documented():
    """Every key to_dict() emits must be a documented schema property, so the contract never lags the codec."""
    emitted = set(_manifest().to_dict())
    documented = set(MANIFEST_SCHEMA["properties"])
    assert emitted <= documented, f"manifest keys missing from the schema: {sorted(emitted - documented)}"
