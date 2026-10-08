import jsonschema
import pytest

from timenet.manifest import ControlFiles, FilePart, Manifest, ManifestCounts, ManifestFiles, TimeSeriesFiles
from timenet.schemas import MANIFEST_SCHEMA
from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DatasetSchema,
    Domain,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.values_backends import ValuesBackend


def _manifest() -> Manifest:
    spec = TimeSeriesSpec(
        spec_type="ecg",
        name="ECG",
        unit_value=ureg.millivolt,
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
        dataset_schema=schema,
        counts=ManifestCounts(
            records=2,
            sources=2,
            signals=2,
            axes=1,
            annotation_contents=3,
            annotation_occurrences=4,
            tasks={"classification": 2},
            signal_chunks=3,
            signals_by_spec={"ecg": 2},
        ),
        files=ManifestFiles(
            control=ControlFiles(
                backend="duckdb",
                parts=(FilePart(path="control.duckdb", checksum="sha256:" + "0" * 64, size=5),),
            ),
            time_series=TimeSeriesFiles(
                backend=ValuesBackend.PARQUET,
                encoding={"ecg": "byte_stream_split", "rhythm": "dictionary"},
                parts=(
                    FilePart(
                        path="time_series/part-00000.parquet",
                        checksum="sha256:" + "e" * 64,
                        size=50,
                    ),
                ),
            ),
        ),
        timef_format_version=1,
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
        lambda d: d["files"].update(control="control.duckdb"),  # a bare string, not a group object
        lambda d: d["files"].update(time_series=d["files"]["time_series"]["parts"]),  # old list form
        lambda d: d["files"].pop("time_series"),  # missing values group
        lambda d: d["files"]["control"].update(backend="parquet"),  # control is always duckdb
        lambda d: d["files"]["control"].update(encoding={}),  # control has no encoding
        lambda d: d["files"]["control"].update(parts=[]),  # control needs exactly one part
        lambda d: d["files"]["time_series"].update(backend="hdf5"),  # unknown values backend
        lambda d: d["files"]["time_series"].pop("backend"),  # backend is required
        lambda d: d["files"]["time_series"].pop("encoding"),  # encoding is always present
        lambda d: d["files"]["time_series"].update(encoding={"ecg": "zip"}),  # unknown encoding
        lambda d: d.update(values_backend="parquet"),  # old top-level backend key
        lambda d: d.update(value_encoding={}),  # old top-level encoding key
        lambda d: d["metadata"].update(license="Nope"),  # unknown license
        lambda d: d["metadata"].update(concepts=["snomed:80891009"]),  # unknown metadata key
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
