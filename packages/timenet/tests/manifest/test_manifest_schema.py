from functools import reduce
import operator

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


# The two shapes a version's files block can take. Both go through the same codec, so the schema
# has to accept both.
CONTROL_DB_FILES = ManifestFiles(
    control_db=FilePart("control.duckdb", "sha256:" + "a" * 64, 4096),
    time_series=(FilePart("time_series/part-00000.parquet", "sha256:" + "e" * 64, 50),),
)
PARQUET_FILES = ManifestFiles(
    records=(FilePart("records.parquet", "sha256:" + "a" * 64, 10),),
    annotations=(FilePart("annotations.parquet", "sha256:" + "b" * 64, 20),),
    time_series_index=(FilePart("time_series_index.parquet", "sha256:" + "c" * 64, 30),),
    tasks=(FilePart("tasks/task=classification/part-0.parquet", "sha256:" + "d" * 64, 40),),
    time_series=(FilePart("time_series/part-00000.parquet", "sha256:" + "e" * 64, 50),),
)
BOTH_CONTROL_PLANES = pytest.mark.parametrize("files", [CONTROL_DB_FILES, PARQUET_FILES], ids=["duckdb", "parquet"])


def _manifest(files: ManifestFiles = CONTROL_DB_FILES) -> Manifest:
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
        files=files,
    )


def test_schema_is_well_formed():
    jsonschema.Draft202012Validator.check_schema(MANIFEST_SCHEMA)


@BOTH_CONTROL_PLANES
def test_to_dict_validates_against_schema(files):
    jsonschema.validate(_manifest(files).to_dict(), MANIFEST_SCHEMA)


@BOTH_CONTROL_PLANES
def test_the_required_file_keys_are_the_ones_the_codec_emits(files):
    """The schema describes to_dict()'s output, so its required list is that output's key set.

    Requiring less lets a truncated files block, one that names no artifact at all, validate and
    fail much later in the reader. Requiring more rejects a manifest the writer really wrote.
    """
    assert set(MANIFEST_SCHEMA["$defs"]["files"]["required"]) == set(_manifest(files).to_dict()["files"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("metadata"),  # missing required top-level block
        lambda d: d.update(timef_format_version=99),  # not the pinned const
        lambda d: d["schema"].update(tasks=[{"task_type": "nope"}]),  # unknown task_type
        lambda d: d["schema"]["annotations"][0].update(annotation_type="sideways"),  # bad annotation_type
        lambda d: d["files"].update(records="single.parquet"),  # a bare string, not a list of parts
        lambda d: d.update(files={}),  # a truncated files block naming no artifact at all
        lambda d: d["files"].pop("control_db"),  # a files block that skips the control plane
        lambda d: d["metadata"].update(license="Nope"),  # unknown license
    ],
)
def test_invalid_manifests_are_rejected(mutate):
    data = _manifest().to_dict()
    mutate(data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, MANIFEST_SCHEMA)


def _resolve(node):
    while "$ref" in node:
        node = reduce(operator.getitem, node["$ref"].removeprefix("#/").split("/"), MANIFEST_SCHEMA)
    return node


def _branches(node):
    node = _resolve(node)
    found = [node]
    for keyword in ("oneOf", "anyOf", "allOf"):
        for branch in node.get(keyword, ()):
            found.extend(_branches(branch))
    return found


def _undocumented(value, node, path):
    branches = _branches(node)
    if isinstance(value, dict):
        named = {key: schema for branch in branches for key, schema in branch.get("properties", {}).items()}
        free = [
            branch["additionalProperties"]
            for branch in branches
            if isinstance(branch.get("additionalProperties"), dict)
        ]
        for key, item in value.items():
            if key in named:
                yield from _undocumented(item, named[key], f"{path}.{key}")
            elif free:
                yield from _undocumented(item, free[0], f"{path}.{key}")
            else:
                yield f"{path}.{key}"
    elif isinstance(value, list):
        items = next((branch["items"] for branch in branches if "items" in branch), None)
        if items is not None:
            for index, item in enumerate(value):
                yield from _undocumented(item, items, f"{path}[{index}]")


@BOTH_CONTROL_PLANES
def test_every_serialized_key_is_documented(files):
    """Every key to_dict() emits, at every depth, must be a documented schema property.

    Walking only the top level let a nested key such as files.control_db go undocumented for as long
    as its parent block was named.
    """
    missing = sorted(_undocumented(_manifest(files).to_dict(), MANIFEST_SCHEMA, "manifest"))
    assert not missing, f"manifest keys missing from the schema: {missing}"


def test_the_walk_finds_a_nested_key_the_schema_does_not_document():
    data = _manifest().to_dict()
    data["files"]["control_db"]["compression"] = "zstd"
    assert sorted(_undocumented(data, MANIFEST_SCHEMA, "manifest")) == ["manifest.files.control_db.compression"]
