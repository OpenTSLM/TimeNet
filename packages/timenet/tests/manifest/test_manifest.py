from dataclasses import replace

from hypothesis import given, strategies as st
import pint
import pytest

from timenet.errors import TimeNetInvalidManifestError
from timenet.manifest import FilePart, Manifest, ManifestCounts, ManifestFiles
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


def _manifest(*, values_backend: str = "parquet") -> Manifest:
    holter = DataSource(data_source_type="holter_x", name="Holter Monitor X", provider="Acme")
    ecg = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_value=ureg.millivolt,
        data_source=holter,
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
        schema=schema,
        counts=ManifestCounts(
            samples=2,
            annotations=4,
            tasks={"classification": 2},
            time_series_chunks=3,
            time_series_index_rows=3,
            time_series_specs={"ecg_lead": 2},
        ),
        files=ManifestFiles(
            samples=(FilePart("samples.parquet", "sha256:aa", 10),),
            annotations=(FilePart("annotations.parquet", "sha256:bb", 20),),
            time_series_index=(FilePart("time_series_index.parquet", "sha256:cc", 30),),
            tasks=(FilePart("tasks/task=classification/part-0.parquet", "sha256:dd", 40),),
            time_series=(FilePart("time_series/part-00000.parquet", "sha256:ee", 50),),
        ),
        values_backend=values_backend,
    )


def test_files_all_parts_concatenates_in_order():
    files = _manifest().files
    assert files.all_parts() == (
        *(p.path for p in files.samples),
        *(p.path for p in files.annotations),
        *(p.path for p in files.time_series_index),
        *(p.path for p in files.tasks),
        *(p.path for p in files.time_series),
    )


def test_default_format_version():
    assert _manifest().timef_format_version == 1


def test_dict_roundtrip():
    m = _manifest()
    assert Manifest.from_dict(m.to_dict()) == m


def test_json_roundtrip():
    m = _manifest()
    assert Manifest.from_json(m.to_json()) == m


def test_to_dict_shape():
    d = _manifest().to_dict()
    assert d["timef_format_version"] == 1
    assert d["dataset_id"] == "demo/ecg"
    assert d["metadata"]["dataset_version"] == "1.2.0"
    assert d["metadata"]["license"] == "CC-BY-4.0"
    assert d["schema"]["time_series_specs"][0]["unit_value"] == "millivolt"
    # the record sits on the spec, so nothing has to be resolved against a side table on read
    assert d["schema"]["time_series_specs"][0]["data_source"] == {
        "data_source_type": "holter_x",
        "name": "Holter Monitor X",
        "provider": "Acme",
    }
    assert "data_sources" not in d["schema"]
    assert d["schema"]["tasks"] == [{"task_type": "classification"}, {"task_type": "answer"}]
    assert d["counts"]["tasks"] == {"classification": 2}


def test_from_dict_reads_the_data_source_stored_on_the_spec():
    schema = Manifest.from_dict(_manifest().to_dict()).schema
    spec = schema.time_series_specs[0]
    assert spec.data_source == DataSource(data_source_type="holter_x", name="Holter Monitor X", provider="Acme")


def test_from_dict_resolves_tasks_to_real_classes():
    schema = Manifest.from_dict(_manifest().to_dict()).schema
    assert schema.tasks == (ClassificationTask, AnswerTask)


def test_units_roundtrip_as_pint():
    spec = Manifest.from_dict(_manifest().to_dict()).schema.time_series_specs[0]
    assert spec.unit_value == ureg.millivolt
    assert isinstance(spec.unit_value, pint.Unit)


def test_unsupported_format_version_rejected():
    d = _manifest().to_dict()
    d["timef_format_version"] = 99
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


def test_direct_construction_validates_format_version():
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest(
            dataset_id="x",
            metadata=_manifest().metadata,
            files=_manifest().files,
            timef_format_version=99,
        )


def test_dataset_id_must_match_metadata():
    with pytest.raises(TimeNetInvalidManifestError, match="does not match"):
        Manifest(dataset_id="other", metadata=_manifest().metadata, files=_manifest().files)


@pytest.mark.parametrize("missing", ["timef_format_version", "dataset_id", "metadata", "files"])
def test_from_dict_requires_core_blocks(missing):
    d = _manifest().to_dict()
    del d[missing]
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


@pytest.mark.parametrize(
    "entry",
    [
        "oops",  # a bare string where a file descriptor object is required
        {"path": "samples/part-00000000.parquet"},  # missing checksum and size
        {"path": "x", "checksum": "sha256:" + "a" * 64},  # missing size
        {"path": 123, "checksum": "sha256:" + "a" * 64, "size": 10},  # path not a string
        {"path": "", "checksum": "sha256:" + "a" * 64, "size": 10},  # empty path
        {"path": "x", "checksum": "md5:whatever", "size": 10},  # checksum not sha256-prefixed
        {"path": "x", "checksum": "sha256:" + "a" * 64, "size": -1},  # negative size
        {"path": "x", "checksum": "sha256:" + "a" * 64, "size": True},  # bool masquerading as an int
        {"path": "/etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # absolute path
        {"path": "../../etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # traversal above root
        {"path": "samples/../../etc/passwd", "checksum": "sha256:" + "a" * 64, "size": 10},  # traversal mid-path
    ],
)
def test_from_dict_rejects_a_malformed_file_entry(entry):
    # A file group is a list of {path, checksum, size} descriptors; a non-dict entry or one missing a
    # field is a corrupt manifest, surfaced as TimeNetInvalidManifestError rather than a raw TypeError/KeyError.
    d = _manifest().to_dict()
    d["files"]["samples"] = [entry]
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


def test_optional_schema_and_counts_default_empty():
    d = _manifest().to_dict()
    del d["schema"]
    del d["counts"]
    m = Manifest.from_dict(d)
    assert m.schema == DatasetSchema()
    assert m.counts == ManifestCounts()


def test_values_backend_defaults_to_parquet():
    assert _manifest().values_backend == "parquet"
    assert _manifest().to_dict()["values_backend"] == "parquet"


def test_values_backend_absent_reads_as_parquet():
    d = _manifest().to_dict()
    del d["values_backend"]  # a pre-backend manifest
    assert Manifest.from_dict(d).values_backend == "parquet"


def test_values_backend_round_trips():
    m = _manifest(values_backend="parquet")
    assert Manifest.from_json(m.to_json()).values_backend == "parquet"


def test_unknown_values_backend_rejected_when_parsing():
    with pytest.raises(TimeNetInvalidManifestError, match="values_backend"):
        _manifest(values_backend="feather")


def test_unknown_values_backend_rejected():
    data = _manifest().to_dict()
    data["values_backend"] = "hdf5"
    with pytest.raises(TimeNetInvalidManifestError, match="values_backend"):
        Manifest.from_dict(data)


def test_unmodeled_metadata_keys_dropped():
    d = _manifest().to_dict()
    d["metadata"]["concepts"] = ["snomed:80891009"]  # not a modeled field
    m = Manifest.from_dict(d)  # tolerated, dropped
    assert not hasattr(m.metadata, "concepts")


def test_unknown_task_type_rejected():
    d = _manifest().to_dict()
    d["schema"]["tasks"] = [{"task_type": "not_a_task"}]
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


def test_bad_unit_string_rejected():
    d = _manifest().to_dict()
    d["schema"]["time_series_specs"][0]["unit_value"] = "not_a_unit"
    with pytest.raises(TimeNetInvalidManifestError, match="schema"):
        Manifest.from_dict(d)


def test_non_string_dataset_version_rejected():
    d = _manifest().to_dict()
    d["metadata"]["dataset_version"] = 3
    with pytest.raises(TimeNetInvalidManifestError, match="metadata"):
        Manifest.from_dict(d)


@pytest.mark.parametrize("block", ["schema", "counts", "metadata", "files"])
def test_null_block_rejected(block):
    d = _manifest().to_dict()
    d[block] = None
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


@given(
    version=st.tuples(st.integers(0, 50), st.integers(0, 50), st.integers(0, 50)),
    samples=st.integers(0, 10_000),
    task_counts=st.dictionaries(st.sampled_from(["classification", "labeling"]), st.integers(0, 999)),
)
def test_codec_roundtrip_property(version, samples, task_counts):
    manifest = Manifest(
        dataset_id="demo/ds",
        metadata=DatasetMetadata(
            dataset_id="demo/ds",
            dataset_version=Version(*version),
            name="n",
            description="d",
            license=License.MIT,
        ),
        counts=ManifestCounts(samples=samples, tasks=task_counts),
        files=ManifestFiles(
            samples=(FilePart("samples.parquet", "sha256:aa", 10),),
            annotations=(FilePart("annotations.parquet", "sha256:bb", 20),),
            time_series_index=(FilePart("time_series_index.parquet", "sha256:cc", 30),),
        ),
    )
    assert Manifest.from_json(manifest.to_json()) == manifest


@pytest.mark.parametrize(
    ("block", "key"),
    [("metadata", "tags"), ("metadata", "domains"), ("files", "tasks"), ("files", "time_series")],
)
def test_string_for_list_field_rejected(block, key):
    # a bare string where a list is expected must not be silently split into characters
    d = _manifest().to_dict()
    d[block][key] = "oops"
    with pytest.raises(TimeNetInvalidManifestError):
        Manifest.from_dict(d)


@pytest.mark.parametrize("block", ["id_encoding", "derived_from"])
def test_bad_dict_block_names_itself(block):
    # each block gets its own error message, rather than one shared "id_encoding/derived_from" string
    d = _manifest().to_dict()
    d[block] = "oops"
    with pytest.raises(TimeNetInvalidManifestError, match=block):
        Manifest.from_dict(d)


def test_build_env_defaults_to_empty():
    assert _manifest().build_env == {}


def test_build_env_round_trips():
    m = replace(_manifest(), build_env={"python": "3.11.9", "packages": {"timenet": "0.1.0"}})
    assert Manifest.from_dict(m.to_dict()).build_env == m.build_env


def test_to_dict_copies_build_env():
    build_env = {"python": "3.11.9", "packages": {"timenet": "0.1.0"}}
    d = replace(_manifest(), build_env=build_env).to_dict()
    d["build_env"]["python"] = "2.7.0"
    assert build_env["python"] == "3.11.9"
