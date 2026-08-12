import json
from pathlib import Path
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.writer import TimeFWriter


def _write(tmp_path, dataset=None, **kwargs) -> Path:
    dataset = dataset if dataset is not None else make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


# ---- round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_full_round_trip(tmp_path, backend):
    original = make_dataset()
    version_dir = _write(tmp_path, dataset=make_dataset(), values_backend=backend)
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert isinstance(restored, TimeFDataset)
    assert_datasets_equal(original, restored)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_round_trip_with_chunk_splitting(tmp_path, backend):
    original = make_dataset()
    version_dir = _write(
        tmp_path, dataset=make_dataset(), values_backend=backend, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_steps_matches_full_series_slice(tmp_path, backend):
    version_dir = _write(
        tmp_path, dataset=make_dataset(), values_backend=backend, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(version_dir) as reader:
        series = next(iter(reader.iter_samples())).time_series[0]
        expected = series.to_arrow().slice(3, 7)
        assert series.read_steps(3, 10).equals(expected)


def test_metadata_and_schema(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        assert reader.metadata.dataset_id == "timenet/hello-world"
        assert {s.spec_type for s in reader.schema.time_series_specs} == {"sine", "cosine"}
        assert ClassificationTask in reader.schema.tasks


def test_shared_series_distinct_objects_same_id(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        samples = {s.sample_id: s for s in reader.read().samples}
    a = next(ts for ts in samples["sample-0"].time_series if ts.time_series_id == "ts-shared")
    b = next(ts for ts in samples["sample-1"].time_series if ts.time_series_id == "ts-shared")
    assert a is not b
    assert a.to_arrow().equals(b.to_arrow())


def test_annotation_value_types_round_trip(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        samples = {s.sample_id: s for s in reader.read().samples}
    anns = {a.key: a for a in samples["sample-0"].annotations}
    assert anns["age"].span is None
    assert anns["age"].value == 64 and isinstance(anns["age"].value, int)
    assert isinstance(anns["stimulus"].span, TimePoint)
    assert isinstance(anns["artifact"].span, TimeInterval)
    artifact_span = anns["artifact"].span
    assert artifact_span is not None
    assert artifact_span.time_series_ids == ("ts-shared",)


def test_task_chain_round_trips(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        tasks = {t.id: t for t in reader.tasks}
    answer = tasks["task-answer-0"]
    assert isinstance(answer, AnswerTask)
    assert answer.from_tasks and answer.from_tasks[0].id == "task-cls-0"


def test_shared_task_frame_round_trips(tmp_path):
    # prompt / rationale / input_annotation_ids live on the base, so they round-trip for every type.
    dataset = make_dataset()
    sample = dataset.samples[0]
    dataset.add_task(sample, AnswerTask(prompt="Any ectopy?", target="No.", id="task-answer-1"))  # rationale=None
    version_dir = _write(tmp_path, dataset=dataset)
    with TimeFReader(version_dir) as reader:
        tasks = {t.id: t for t in reader.tasks}
    with_rationale = tasks["task-answer-0"]
    assert isinstance(with_rationale, AnswerTask)
    assert with_rationale.prompt == "What rhythm?"
    assert with_rationale.rationale == "Regular intervals with one peak per cycle."
    assert with_rationale.input_annotation_ids == ("cohort-shared",)
    assert with_rationale.target == "Normal."
    without_rationale = tasks["task-answer-1"]
    assert without_rationale.rationale is None  # optional field round-trips as None
    assert without_rationale.input_annotation_ids == ()


def test_scope_and_localization_spans_round_trip(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        tasks = {t.id: t for t in reader.tasks}
    assert tasks["task-cls-2"].scope == TimeInterval.seconds(0.0, 0.25, time_series_ids=("ts-window-2",))
    localization = tasks["task-localize-0"]
    assert isinstance(localization, TemporalLocalizationTask)
    assert localization.target == (
        TimePoint.seconds(0.5),  # a point: it decodes to a TimePoint, which has no end bound
        TimeInterval.seconds(0.0, 0.25, time_series_ids=("ts-shared",)),
    )
    assert localization.mode is LocalizationMode.SPARSE


def test_scalar_target_round_trips_as_a_number(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        task = {t.id: t for t in reader.tasks}["task-scalar-0"]
    assert isinstance(task, ScalarPredictionTask)
    assert task.target == pytest.approx(62.0)
    assert task.unit == "bpm" and task.target_name == "mean_rate"


def test_iter_samples_matches_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        streamed = {s.sample_id for s in reader.iter_samples()}
    with TimeFReader(version_dir) as reader:
        read_ids = {s.sample_id for s in reader.read().samples}
    assert streamed == read_ids


# ---- laziness ---------------------------------------------------------------------------------


def test_values_are_lazy(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original_open = pq.ParquetFile
    opens = {"n": 0}

    def counting_open(*args, **kwargs):
        opens["n"] += 1
        return original_open(*args, **kwargs)

    with TimeFReader(version_dir) as reader:
        dataset = reader.read()
        monkeypatch.setattr(pq, "ParquetFile", counting_open)
        ts = dataset.samples[0].time_series[0]
        assert opens["n"] == 0  # building samples opened no shards
        ts.to_arrow()
        assert opens["n"] >= 1  # reading values opened a shard


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_back_dataset_is_picklable(tmp_path, backend):
    # A multi-worker torch DataLoader pickles the dataset to each worker, so lazy loaders must pickle
    # even after values (and thus the backend's handles/caches) have been touched.
    version_dir = _write(tmp_path, values_backend=backend)
    dataset = TimeFReader(version_dir).read()
    first = dataset.samples[0].time_series[0]
    original = first.to_arrow()  # populates the values backend's caches

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.samples[0].time_series[0].to_arrow().equals(original)


# ---- validation -------------------------------------------------------------------------------


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TimeFReader(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        TimeFReader(tmp_path / "empty")


@pytest.mark.parametrize("format_version", [2, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(TimeFFormatError):
        TimeFReader(version_dir)


def test_corrupt_index_locator_has_series_context(tmp_path):
    # Corrupt the artifact rather than the reader's internals: the locator is read from the index on
    # each lookup now, so an in-memory poke would not survive to the read.
    version_dir = _write(tmp_path)
    index_path = version_dir / "time_series_index.parquet"
    table = pq.read_table(index_path)
    bogus = pa.array(["time_series/does-not-exist.parquet"] * table.num_rows)
    pq.write_table(table.set_column(table.schema.get_field_index("chunk_file"), "chunk_file", bogus), index_path)

    with TimeFReader(version_dir) as reader:
        sample = next(iter(reader.iter_samples()))
        series_id = sample.time_series[0].time_series_id
        with pytest.raises(
            TimeFFormatError, match=f"failed to read series {series_id!r} for sample {sample.sample_id!r}"
        ):
            reader._load_values(sample.sample_id, series_id)


def test_missing_listed_file_raises(tmp_path):
    version_dir = _write(tmp_path)
    (version_dir / "samples.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        TimeFReader(version_dir)


def test_verify_passes_on_an_intact_dataset(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        reader.verify()  # must not raise


def test_verify_detects_a_corrupted_shard(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        # a shard: read lazily, so __init__ still succeeds and verify() is what catches it
        rel = next(r for r in reader._manifest.checksums if r.startswith("time_series/shard-"))
    target = version_dir / rel
    target.write_bytes(target.read_bytes() + b"corruption")
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="checksum mismatch"):
        reader.verify()


def test_verify_detects_a_deleted_file(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        rel = next(r for r in reader._manifest.checksums if r.startswith("time_series/shard-"))
    (version_dir / rel).unlink()
    # __init__ already refuses a manifest-listed file that is gone
    with pytest.raises(FileNotFoundError):
        TimeFReader(version_dir)


def test_corrupt_task_partition_raises_format_error(tmp_path):
    # An unknown task partition name is corrupt on-disk data, so it must surface as TimeFFormatError
    # rather than the bare ValueError that TaskType() happens to raise.
    version_dir = _write(tmp_path)
    tasks_dir = next((version_dir / "tasks").iterdir())
    tasks_dir.rename(tasks_dir.parent / "task=not_a_real_task_type")
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tasks"] = [
        p.replace(tasks_dir.name, "task=not_a_real_task_type") for p in manifest["files"]["tasks"]
    ]
    manifest["checksums"] = {
        k.replace(tasks_dir.name, "task=not_a_real_task_type"): v for k, v in manifest["checksums"].items()
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TimeFFormatError):
        TimeFReader(version_dir)


def test_start_time_round_trips_exactly(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        samples = {s.sample_id: s for s in reader.iter_samples()}
    # The fixture anchor is not representable in float64, so this catches any float coercion.
    assert samples["sample-0"].start_time == 9_007_199_254_740_993
    assert samples["sample-1"].start_time is None
    assert samples["sample-2"].start_time is None


def test_samples_file_without_start_time_column_reads_as_none(tmp_path):
    version_dir = _write(tmp_path)
    samples_path = version_dir / "samples.parquet"
    table = pq.read_table(samples_path)
    pq.write_table(table.drop_columns(["start_time_us"]), samples_path)
    with TimeFReader(version_dir) as reader:
        assert all(s.start_time is None for s in reader.iter_samples())


def _corrupt_first_series(version_dir, field, value):
    """Set a field on the first series' struct in samples.parquet, simulating on-disk corruption."""
    samples_path = version_dir / "samples.parquet"
    table = pq.read_table(samples_path)
    rows = table.to_pylist()
    rows[0]["time_series"][0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples_path)


def test_ordinal_row_carrying_regular_columns_raises_format_error(tmp_path):
    # The tag and the columns disagree: an ordinal series must have no period or start index.
    version_dir = _write(tmp_path)
    _corrupt_first_series(version_dir, "axis_type", "ordinal")
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="carries regular- or"):
        list(reader.iter_samples())


def test_regular_row_with_a_zero_denominator_raises_format_error(tmp_path):
    # A zero denominator would raise a raw ZeroDivisionError from Fraction; it must surface as format error.
    version_dir = _write(tmp_path)
    _corrupt_first_series(version_dir, "period_denominator", 0)
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="unbuildable regular axis"):
        list(reader.iter_samples())


def _corrupt_descriptor(version_dir, key, field, value):
    """Rewrite one annotation descriptor field in the manifest, simulating on-disk corruption."""
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for descriptor in manifest["schema"]["annotations"]:
        if descriptor["key"] == key:
            descriptor[field] = value
    manifest_path.write_text(json.dumps(manifest))


def test_annotation_shape_disagreeing_with_its_descriptor_raises_format_error(tmp_path):
    # "artifact" is an interval; a descriptor that calls it static no longer matches the decoded span.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "artifact", "annotation_type", "static")
    with pytest.raises(TimeFFormatError, match="decodes to shape"):
        TimeFReader(version_dir)


def test_annotation_value_type_disagreeing_with_its_descriptor_raises_format_error(tmp_path):
    # "age" is an int; a descriptor that calls it a str no longer matches the decoded value.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "age", "value_type", "str")
    with pytest.raises(TimeFFormatError, match="value type"):
        TimeFReader(version_dir)


def test_annotation_span_outside_the_series_raises_format_error(tmp_path):
    # A stored span that no longer fits the series it resolves to is corruption, not a caller mistake.
    version_dir = _write(tmp_path)
    ann_path = version_dir / "annotations.parquet"
    table = pq.read_table(ann_path)
    rows = table.to_pylist()
    for row in rows:
        # Only stretch an interval's end; nulling a point's would change its shape instead.
        if row["span"] is not None and row["span"]["end_us"] is not None:
            row["span"]["end_us"] = 10**15  # far past any series window
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), ann_path)
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="falls outside sample"):
        list(reader.iter_samples())


def _time_span_dataset(tmp_path) -> Path:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/time-span",
            dataset_version=Version(1, 0, 0),
            name="T",
            description="d",
            license=License.MIT,
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless),
        channel="c",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=3,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )
    sample = dataset.add_sample(time_series=(series,), time_span=TimeInterval.seconds(0.0, 5.0))
    sample.add_annotation(Annotation(key="note", span=TimePoint.seconds(2.0)))  # unscoped, inside [0, 5) s
    return _write(tmp_path, dataset)


def _corrupt_first_time_span(version_dir, struct):
    """Replace the first sample's time_span struct in samples.parquet, simulating on-disk corruption."""
    samples_path = version_dir / "samples.parquet"
    table = pq.read_table(samples_path)
    rows = table.to_pylist()
    rows[0]["time_span"] = struct
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples_path)


def test_time_span_with_reversed_bounds_raises_format_error(tmp_path):
    # A stored time_span whose end is not past its start fails Span validation on decode; it must surface
    # as a format error, not the raw ValueError that validation raises.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_first_time_span(version_dir, {"start_us": 6_000_000, "end_us": 5_000_000, "time_series_ids": None})
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="must be > start"):
        list(reader.iter_samples())


def test_point_shaped_time_span_raises_format_error(tmp_path):
    # A time_span must be an interval covering the whole sample. A corrupt point-shaped one (no end) is
    # rejected as a format error, not left to reach the unscoped-span check and raise a bare TypeError.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_first_time_span(version_dir, {"start_us": 0, "end_us": None, "time_series_ids": None})
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="must be a TimeInterval"):
        list(reader.iter_samples())
