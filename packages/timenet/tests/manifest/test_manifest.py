from hypothesis import given, strategies as st
import pytest

from timenet.errors import InvalidManifestError
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles
from timenet.types import (
    AnnotationDescriptor,
    AnnotationType,
    ClassificationTask,
    DatasetMetadata,
    DatasetSchema,
    DataSource,
    Domain,
    License,
    QATask,
    TimeSeriesSpec,
    Version,
    ureg,
)


def _manifest(*, values_backend: str = "parquet") -> Manifest:
    holter = DataSource(data_source_type="holter_x", name="Holter Monitor X", provider="Acme")
    ecg = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
        data_source=holter,
    )
    schema = DatasetSchema(
        time_series_specs=(ecg,),
        data_sources=(holter,),
        annotations=(
            AnnotationDescriptor(key="age", annotation_type=AnnotationType.STATIC, value_type="int", unit="years"),
            AnnotationDescriptor(key="artifact", annotation_type=AnnotationType.INTERVAL),
        ),
        tasks=(ClassificationTask, QATask),
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
            samples=("samples.parquet",),
            annotations=("annotations.parquet",),
            time_series_index=("time_series_index.parquet",),
            tasks=("tasks/task=classification/part-0.parquet",),
            time_series=("time_series/shard-00000.parquet",),
        ),
        values_backend=values_backend,
    )


def test_files_all_parts_concatenates_in_order():
    files = _manifest().files
    assert files.all_parts() == (
        *files.samples,
        *files.annotations,
        *files.time_series_index,
        *files.tasks,
        *files.time_series,
    )


def test_default_format_version():
    assert _manifest().timef_format_version == 2


def test_dict_roundtrip():
    m = _manifest()
    assert Manifest.from_dict(m.to_dict()) == m


def test_json_roundtrip():
    m = _manifest()
    assert Manifest.from_json(m.to_json()) == m


def test_to_dict_shape():
    d = _manifest().to_dict()
    assert d["timef_format_version"] == 2
    assert d["dataset_id"] == "demo/ecg"
    assert d["metadata"]["dataset_version"] == "1.2.0"
    assert d["metadata"]["license"] == "CC-BY-4.0"
    assert d["schema"]["time_series_specs"][0]["unit_value"] == "millivolt"
    # spec references its data source by type tag, not by embedding it
    assert d["schema"]["time_series_specs"][0]["data_source"] == "holter_x"
    assert d["schema"]["tasks"] == [{"task_type": "classification"}, {"task_type": "question_and_answer"}]
    assert d["counts"]["tasks"] == {"classification": 2}


def test_from_dict_resolves_data_source_reference():
    schema = Manifest.from_dict(_manifest().to_dict()).schema
    spec = schema.time_series_specs[0]
    assert spec.data_source is not None
    assert spec.data_source.data_source_type == "holter_x"
    assert spec.data_source in schema.data_sources


def test_from_dict_resolves_tasks_to_real_classes():
    schema = Manifest.from_dict(_manifest().to_dict()).schema
    assert schema.tasks == (ClassificationTask, QATask)


def test_units_roundtrip_as_pint():
    spec = Manifest.from_dict(_manifest().to_dict()).schema.time_series_specs[0]
    assert spec.unit_value == ureg.millivolt
    assert spec.unit_sampling_rate == ureg.hertz


def test_unsupported_format_version_rejected():
    d = _manifest().to_dict()
    d["timef_format_version"] = 99
    with pytest.raises(InvalidManifestError):
        Manifest.from_dict(d)


def test_direct_construction_validates_format_version():
    with pytest.raises(InvalidManifestError):
        Manifest(
            dataset_id="x",
            metadata=_manifest().metadata,
            files=_manifest().files,
            timef_format_version=99,
        )


def test_dataset_id_must_match_metadata():
    with pytest.raises(InvalidManifestError, match="does not match"):
        Manifest(dataset_id="other", metadata=_manifest().metadata, files=_manifest().files)


@pytest.mark.parametrize("missing", ["timef_format_version", "dataset_id", "metadata", "files"])
def test_from_dict_requires_core_blocks(missing):
    d = _manifest().to_dict()
    del d[missing]
    with pytest.raises(InvalidManifestError):
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
    with pytest.raises(InvalidManifestError, match="values_backend"):
        _manifest(values_backend="feather")


def test_unmodeled_metadata_keys_dropped():
    d = _manifest().to_dict()
    d["metadata"]["concepts"] = ["snomed:80891009"]  # not a modeled field
    m = Manifest.from_dict(d)  # tolerated, dropped
    assert not hasattr(m.metadata, "concepts")


def test_unknown_task_type_rejected():
    d = _manifest().to_dict()
    d["schema"]["tasks"] = [{"task_type": "not_a_task"}]
    with pytest.raises(InvalidManifestError):
        Manifest.from_dict(d)


def test_bad_unit_string_rejected():
    d = _manifest().to_dict()
    d["schema"]["time_series_specs"][0]["unit_value"] = "not_a_unit"
    with pytest.raises(InvalidManifestError, match="schema"):
        Manifest.from_dict(d)


def test_non_string_dataset_version_rejected():
    d = _manifest().to_dict()
    d["metadata"]["dataset_version"] = 3
    with pytest.raises(InvalidManifestError, match="metadata"):
        Manifest.from_dict(d)


@pytest.mark.parametrize("block", ["schema", "counts", "checksums", "metadata", "files"])
def test_null_block_rejected(block):
    d = _manifest().to_dict()
    d[block] = None
    with pytest.raises(InvalidManifestError):
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
            samples=("samples.parquet",),
            annotations=("annotations.parquet",),
            time_series_index=("time_series_index.parquet",),
        ),
    )
    assert Manifest.from_json(manifest.to_json()) == manifest
