from typing import Any, TypeVar, cast

from hypothesis import given, strategies as st
import jsonschema
import pint
from pydantic import BaseModel, ValidationError
import pytest

from timenet.manifest import FilePart, LockedDependency, Manifest, ManifestCounts, ManifestFiles
from timenet.schemas import MANIFEST_SCHEMA
from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DatasetRef,
    DatasetSchema,
    Domain,
    InputModality,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.values_backends import ValuesBackend


_CONTROL = (FilePart(path="control.duckdb", checksum="sha256:" + "0" * 64, size=5),)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _replace(model: _ModelT, **changes: Any) -> _ModelT:
    """Return a validated model with selected fields replaced."""
    return type(model).model_validate({**model.__dict__, **changes})


def _manifest(*, values_backend: str = ValuesBackend.PARQUET) -> Manifest:
    ecg = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
    )
    schema = DatasetSchema(
        time_series_specs=(ecg,),
        annotations=(
            AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int", unit="years"),
            AnnotationDescriptor(key="artifact", annotation_type=AnnotationType.INTERVAL),
        ),
        tasks=(ClassificationTask, AnswerTask),
    )
    metadata = DatasetMetadata(
        dataset_id="demo/ecg",
        dataset_version=Version(1, 2, 0),
        name="ECG Dataset",
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
            signals_by_spec={"ecg_lead": 2},
        ),
        files=ManifestFiles(
            control=_CONTROL,
            time_series=(
                FilePart(
                    path="time_series/part-00000.parquet",
                    checksum="sha256:" + "e" * 64,
                    size=50,
                ),
            ),
        ),
        values_backend=cast("ValuesBackend", values_backend),
        timef_format_version=1,
    )


def test_files_all_parts_concatenates_in_order():
    files = _manifest().files
    assert files.all_parts() == (
        *(p.path for p in files.control),
        *(p.path for p in files.time_series),
    )


def test_format_version():
    assert _manifest().timef_format_version == 1


def test_unknown_spec_unit_is_null_in_manifest():
    base = _manifest()
    unknown = _replace(base.dataset_schema.time_series_specs[0], unit_value=None)
    manifest = _replace(
        base,
        dataset_schema=_replace(base.dataset_schema, time_series_specs=(unknown,)),
        files=ManifestFiles(control=_CONTROL),
    )
    data = manifest.to_dict()

    assert data["schema"]["time_series_specs"][0]["unit_value"] is None
    jsonschema.validate(data, MANIFEST_SCHEMA)
    assert Manifest.model_validate(data).dataset_schema.time_series_specs[0].unit_value is None


@pytest.mark.parametrize(
    "domain",
    [
        "respiratory",
        "motion",
        "environment",
        "energy",
        "transport",
        "observability",
        "audio",
    ],
)
def test_manifest_round_trips_benchmark_domain(domain):
    payload = _manifest().to_dict()
    payload["metadata"]["domains"] = [domain]
    restored = Manifest.model_validate(payload)
    assert restored.to_dict()["metadata"]["domains"] == [domain]


def test_manifest_rejects_unknown_domain():
    payload = _manifest().to_dict()
    payload["metadata"]["domains"] = ["not-a-domain"]
    with pytest.raises(ValidationError, match=r"metadata\.domains"):
        Manifest.model_validate(payload)


def test_nullable_schema_roundtrips_at_current_format_version():
    # Reading nullable artifacts still requires an SDK
    # that supports nullability, including the parallel validity arrays in Zarr.
    base = _replace(
        _manifest(),
        files=ManifestFiles(control=_CONTROL),
    )
    spec = _replace(base.dataset_schema.time_series_specs[0], nullable=True)
    manifest = Manifest(
        dataset_id=base.dataset_id,
        metadata=base.metadata,
        files=base.files,
        dataset_schema=_replace(base.dataset_schema, time_series_specs=(spec,)),
        timef_format_version=1,
    )
    assert manifest.timef_format_version == 1
    assert manifest.to_dict()["schema"]["time_series_specs"][0]["nullable"] is True
    assert Manifest.model_validate_json(manifest.to_json()) == manifest
    jsonschema.validate(manifest.to_dict(), MANIFEST_SCHEMA)


def test_missing_nullable_defaults_to_false():
    data = _replace(
        _manifest(),
        files=ManifestFiles(control=_CONTROL),
    ).to_dict()
    data["schema"]["time_series_specs"][0].pop("nullable", None)
    restored = Manifest.model_validate(data)
    assert restored.timef_format_version == 1
    assert restored.dataset_schema.time_series_specs[0].nullable is False
    jsonschema.validate(data, MANIFEST_SCHEMA)


@pytest.mark.parametrize("nullable", [1, None, "true"])
def test_manifest_rejects_nonboolean_nullable(nullable):
    data = _replace(
        _manifest(),
        files=ManifestFiles(control=_CONTROL),
    ).to_dict()
    data["schema"]["time_series_specs"][0]["nullable"] = nullable
    with pytest.raises(ValidationError, match="nullable"):
        Manifest.model_validate(data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, MANIFEST_SCHEMA)


@pytest.mark.parametrize("version", [None, True, 1.0])
def test_parsed_manifest_requires_integer_version(version):
    data = _manifest().to_dict()
    data["timef_format_version"] = version
    with pytest.raises(ValidationError, match="timef_format_version"):
        Manifest.model_validate(data)


@pytest.mark.parametrize("modality", [InputModality.TIME_SERIES, InputModality.IMAGE, InputModality.AUDIO])
def test_dict_roundtrip(modality):
    m = _manifest()
    m = _replace(
        m,
        dataset_schema=_replace(
            m.dataset_schema,
            time_series_specs=(_replace(m.dataset_schema.time_series_specs[0], modality=modality),),
        ),
    )
    jsonschema.validate(m.to_dict(), MANIFEST_SCHEMA)
    assert Manifest.model_validate(m.to_dict()) == m


def test_json_roundtrip():
    m = _manifest()
    assert Manifest.model_validate_json(m.to_json()) == m


def test_model_validate_json_exposes_pydantic_json_error():
    with pytest.raises(ValidationError) as exc_info:
        Manifest.model_validate_json("{")
    assert exc_info.value.errors()[0]["type"] == "json_invalid"


def test_to_dict_shape():
    d = _manifest().to_dict()
    assert d["timef_format_version"] == 1
    assert d["dataset_id"] == "demo/ecg"
    assert d["metadata"]["dataset_version"] == "1.2.0"
    assert d["metadata"]["license"] == "CC-BY-4.0"
    assert d["schema"]["time_series_specs"][0]["unit_value"] == "millivolt"
    assert "data_source" not in d["schema"]["time_series_specs"][0]
    assert d["schema"]["tasks"] == [{"task_type": "classification"}, {"task_type": "answer"}]
    assert d["counts"]["tasks"] == {"classification": 2}


def test_model_validate_resolves_tasks_to_real_classes():
    schema = Manifest.model_validate(_manifest().to_dict()).dataset_schema
    assert schema.tasks == (ClassificationTask, AnswerTask)


def test_units_roundtrip_as_pint():
    spec = Manifest.model_validate(_manifest().to_dict()).dataset_schema.time_series_specs[0]
    assert spec.unit_value == ureg.millivolt
    assert isinstance(spec.unit_value, pint.Unit)


def test_str_dtype_round_trips():
    m = _manifest()
    spec = m.dataset_schema.time_series_specs[0]
    str_spec = _replace(spec, dtype="str")
    m = _replace(m, dataset_schema=_replace(m.dataset_schema, time_series_specs=(str_spec,)))
    restored = Manifest.model_validate(m.to_dict())
    assert restored.dataset_schema.time_series_specs[0].dtype == "str"


def test_unsupported_format_version_rejected():
    d = _manifest().to_dict()
    d["timef_format_version"] = 99
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


def test_direct_construction_validates_format_version():
    with pytest.raises(ValidationError):
        Manifest(
            dataset_id="x",
            metadata=_manifest().metadata,
            files=_manifest().files,
            timef_format_version=99,
        )


def test_dataset_id_must_match_metadata():
    with pytest.raises(ValidationError, match="does not match"):
        Manifest(
            dataset_id="other/dataset",
            metadata=_manifest().metadata,
            files=_manifest().files,
            timef_format_version=1,
        )


@pytest.mark.parametrize("missing", ["timef_format_version", "dataset_id", "metadata", "files"])
def test_model_validate_requires_core_blocks(missing):
    d = _manifest().to_dict()
    del d[missing]
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


@pytest.mark.parametrize(
    "entry",
    [
        "oops",  # a bare string where a file descriptor object is required
        {"path": "time_series/part-00000000.parquet"},  # missing checksum and size
        {"path": "x", "checksum": "sha256:" + "a" * 64},  # missing size
        {"path": 123, "checksum": "sha256:" + "a" * 64, "size": 10},  # path not a string
        {"path": "", "checksum": "sha256:" + "a" * 64, "size": 10},  # empty path
        {"path": "x", "checksum": "md5:whatever", "size": 10},  # checksum not sha256-prefixed
        {"path": "x", "checksum": "sha256:" + "a" * 64, "size": -1},  # negative size
        {"path": "x", "checksum": "sha256:" + "a" * 64, "size": True},  # bool masquerading as an int
        {"path": "/etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # absolute path
        {"path": "../../etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # traversal above root
        {"path": "time_series/../../etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # traversal mid-path
    ],
)
def test_model_validate_rejects_a_malformed_file_entry(entry):
    # A file group is a list of {path, checksum, size} descriptors. Pydantic reports a malformed
    # entry with its exact nested field location.
    d = _manifest().to_dict()
    d["files"]["time_series"] = [entry]
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


@pytest.mark.parametrize(
    "part",
    [
        ("x", "sha256:ee", 1),
        ("x", "sha256:" + "A" * 64, 1),
        ("x", "sha256:" + "a" * 64, -1),
        ("x", "sha256:" + "a" * 64, True),
        ("../x", "sha256:" + "a" * 64, 1),
    ],
)
def test_direct_file_part_construction_enforces_wire_invariants(part):
    with pytest.raises(ValidationError):
        FilePart(path=part[0], checksum=part[1], size=part[2])


@pytest.mark.parametrize("value", [-1, True, "1", 1.0])
def test_direct_manifest_counts_require_non_negative_integers(value):
    with pytest.raises(ValidationError, match="records"):
        ManifestCounts(records=value)


def test_optional_schema_and_counts_default_empty():
    d = _manifest().to_dict()
    del d["schema"]
    del d["counts"]
    m = Manifest.model_validate(d)
    assert m.dataset_schema == DatasetSchema()
    assert m.counts == ManifestCounts()


def test_values_backend_defaults_to_parquet():
    assert _manifest().values_backend == "parquet"
    assert _manifest().to_dict()["values_backend"] == "parquet"


def test_values_backend_absent_reads_as_parquet():
    d = _manifest().to_dict()
    del d["values_backend"]  # a pre-backend manifest
    assert Manifest.model_validate(d).values_backend == "parquet"


def test_values_backend_round_trips():
    m = _manifest(values_backend="parquet")
    assert Manifest.model_validate_json(m.to_json()).values_backend == "parquet"


def test_unknown_values_backend_rejected_when_parsing():
    with pytest.raises(ValidationError, match="values_backend"):
        _manifest(values_backend="feather")


def test_unknown_values_backend_rejected():
    data = _manifest().to_dict()
    data["values_backend"] = "hdf5"
    with pytest.raises(ValidationError, match="values_backend"):
        Manifest.model_validate(data)


def test_unmodeled_metadata_keys_rejected():
    d = _manifest().to_dict()
    d["metadata"]["concepts"] = ["snomed:80891009"]  # not a modeled field
    with pytest.raises(ValidationError, match="concepts"):
        Manifest.model_validate(d)


def test_unknown_task_type_rejected():
    d = _manifest().to_dict()
    d["schema"]["tasks"] = [{"task_type": "not_a_task"}]
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


def test_bad_unit_string_rejected():
    d = _manifest().to_dict()
    d["schema"]["time_series_specs"][0]["unit_value"] = "not_a_unit"
    with pytest.raises(ValidationError, match="schema"):
        Manifest.model_validate(d)


def test_non_string_dataset_version_rejected():
    d = _manifest().to_dict()
    d["metadata"]["dataset_version"] = 3
    with pytest.raises(ValidationError, match="metadata"):
        Manifest.model_validate(d)


@pytest.mark.parametrize("block", ["schema", "counts", "metadata", "files"])
def test_null_block_rejected(block):
    d = _manifest().to_dict()
    d[block] = None
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


@given(
    version=st.tuples(st.integers(0, 50), st.integers(0, 50), st.integers(0, 50)),
    records=st.integers(0, 10_000),
    task_counts=st.dictionaries(st.sampled_from(["classification", "labeling"]), st.integers(0, 999)),
)
def test_codec_roundtrip_property(version, records, task_counts):
    manifest = Manifest(
        dataset_id="demo/ds",
        metadata=DatasetMetadata(
            dataset_id="demo/ds",
            dataset_version=Version(*version),
            name="n",
            description="d",
            license=License.MIT,
        ),
        counts=ManifestCounts(records=records, tasks=task_counts),
        files=ManifestFiles(control=_CONTROL),
        timef_format_version=1,
    )
    assert Manifest.model_validate_json(manifest.to_json()) == manifest


@pytest.mark.parametrize(
    ("block", "key"),
    [("metadata", "tags"), ("metadata", "domains"), ("files", "control"), ("files", "time_series")],
)
def test_string_for_list_field_rejected(block, key):
    # a bare string where a list is expected must not be silently split into characters
    d = _manifest().to_dict()
    d[block][key] = "oops"
    with pytest.raises(ValidationError):
        Manifest.model_validate(d)


@pytest.mark.parametrize("block", ["value_encoding", "build_env"])
def test_bad_dict_block_names_itself(block):
    # Pydantic includes the block name in its native error location.
    d = _manifest().to_dict()
    d[block] = "oops"
    with pytest.raises(ValidationError, match=block):
        Manifest.model_validate(d)


def test_build_env_defaults_to_empty():
    assert _manifest().build_env == {}


def test_build_env_round_trips():
    m = _replace(_manifest(), build_env={"python": "3.11.9", "packages": {"timenet": "0.1.0"}})
    assert Manifest.model_validate(m.to_dict()).build_env == m.build_env


@pytest.mark.parametrize(
    "change",
    [
        {"build_env": {"python": 313}},
        {"value_encoding": {"ecg_lead": "zip"}},
    ],
)
def test_direct_manifest_rejects_invalid_provenance_blocks(change):
    with pytest.raises(ValidationError):
        _replace(_manifest(), **change)


def test_to_dict_copies_build_env():
    build_env = {"python": "3.11.9", "packages": {"timenet": "0.1.0"}}
    d = _replace(_manifest(), build_env=build_env).to_dict()
    d["build_env"]["python"] = "2.7.0"
    assert build_env["python"] == "3.11.9"


@pytest.mark.parametrize("method", ["to_dict", "to_json"])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.build_env.update(python=313),
        lambda manifest: manifest.counts.tasks.update(classification=-1),
    ],
)
def test_serialization_revalidates_mutable_nested_values(mutate, method):
    manifest = _manifest()
    mutate(manifest)
    with pytest.raises(ValidationError):
        getattr(manifest, method)()


def _lock(parent: DatasetRef) -> LockedDependency:
    return LockedDependency(
        dataset_id=parent.dataset_id, version=parent.version, manifest_checksum="sha256:" + "a" * 64
    )


def _with_parent(manifest: Manifest, parent: DatasetRef) -> Manifest:
    return _replace(
        manifest,
        metadata=_replace(
            manifest.metadata,
            parents=(DatasetRef(dataset_id=parent.dataset_id, version=parent.version),),
        ),
    )


def test_dependency_lock_round_trips():
    parent = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))
    manifest = _with_parent(
        _replace(_manifest(), dependencies=(_lock(parent),), files=ManifestFiles(control=_CONTROL)),
        parent,
    )

    assert Manifest.model_validate_json(manifest.to_json()) == manifest
    jsonschema.validate(manifest.to_dict(), MANIFEST_SCHEMA)


def test_dependency_lock_rejects_duplicate_dataset_versions():
    parent = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))
    with pytest.raises(ValidationError, match="twice"):
        _replace(_manifest(), dependencies=(_lock(parent), _lock(parent)))


def test_manifest_requires_a_lock_row_for_every_declared_parent():
    parent = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))
    with pytest.raises(ValidationError, match="absent from the dependency lock"):
        _with_parent(_manifest(), parent)
