import jsonschema
import pytest

from timenet.manifest import FileGroup, FileKind, FilePart, Manifest, ManifestCounts, ManifestFiles
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
        schema=schema,
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
            groups=(
                FileGroup(FileKind.CONTROL, "duckdb", (FilePart("control.duckdb", "sha256:" + "0" * 64, 5),)),
                FileGroup(
                    FileKind.TIME_SERIES,
                    "parquet",
                    (FilePart("time_series/part-00000.parquet", "sha256:" + "e" * 64, 50),),
                    encoding={"ecg": "dictionary"},
                ),
            )
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
        lambda d: d["files"][0].update(parts="control.duckdb"),  # a bare string, not a list of parts
        lambda d: d["files"].pop(0),  # no control group
        lambda d: d["files"].append(d["files"][0]),  # two control groups
        lambda d: d["files"].append(d["files"][1]),  # two time_series groups
        lambda d: d["files"][0]["parts"].append(d["files"][0]["parts"][0]),  # control with two files
        lambda d: d["files"][0].update(backend="parquet"),  # control on a values backend
        lambda d: d["files"][1].update(backend="duckdb"),  # time series on the control backend
        lambda d: d["files"][1].update(kind="images"),  # unknown kind
        lambda d: d["files"][1].pop("parts"),  # group without parts
        lambda d: d["files"][1].update(encoding={"ecg": "gzip"}),  # unknown values encoding
        lambda d: d.update(timef_format_version=2),  # the previous format version
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
    data = _manifest().to_dict()
    emitted = set(data)
    documented = set(MANIFEST_SCHEMA["properties"])
    assert emitted <= documented, f"manifest keys missing from the schema: {sorted(emitted - documented)}"
    group_keys = {key for group in data["files"] for key in group}
    group_documented = set(MANIFEST_SCHEMA["$defs"]["fileGroup"]["properties"])
    assert group_keys <= group_documented, f"file group keys missing from the schema: {group_keys - group_documented}"
