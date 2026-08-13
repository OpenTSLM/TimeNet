import json
from pathlib import Path
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
import timenet.reader.parts as parts_mod
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    IntervalSpan,
    LocalizationMode,
    PointSpan,
    ScalarPredictionTask,
    TemporalLocalizationTask,
)
import timenet.values_backends.parquet.writer as parquet_writer
from timenet.writer import TimeFWriter
import timenet.writer.writer as writer_mod


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


def test_round_trip_with_control_sharding(tmp_path):
    # A 1-byte control target splits every control table into per-row parts under its own subdir.
    original = make_dataset()
    version_dir = _write(tmp_path, dataset=make_dataset(), control_shard_target_bytes=1)
    files = Manifest.from_json((version_dir / "manifest.json").read_text()).files
    assert len(files.samples) > 1
    assert len(files.annotations) > 1
    assert len(files.time_series_index) > 1
    assert sum("task=classification" in f for f in files.tasks) > 1  # two classification tasks split
    for rel in (*files.samples, *files.annotations, *files.time_series_index, *files.tasks):
        assert (version_dir / rel).exists()
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


def test_index_resolves_when_a_series_straddles_parts(tmp_path):
    # Small chunks give a series several index rows; a 1-byte control target puts them in separate
    # parts. The reader concatenates parts in manifest order and bisects, so the series still resolves.
    original = make_dataset()
    version_dir = _write(
        tmp_path, dataset=make_dataset(), control_shard_target_bytes=1, chunk_max_bytes=64, row_group_target_bytes=64
    )
    files = Manifest.from_json((version_dir / "manifest.json").read_text()).files
    assert len(files.time_series_index) > 1  # index rows spread across parts, some series straddling
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


def test_reads_a_legacy_flat_layout(tmp_path, monkeypatch):
    # A dataset written before this change used flat single files, {:05d} shards and part-0 task files.
    # The reader resolves the paths stored in the manifest, so it still reads them with no version gate.
    monkeypatch.setattr(writer_mod, "SAMPLES_TEMPLATE", "samples.parquet")
    monkeypatch.setattr(writer_mod, "ANNOTATIONS_TEMPLATE", "annotations.parquet")
    monkeypatch.setattr(writer_mod, "INDEX_TEMPLATE", "time_series_index.parquet")
    monkeypatch.setattr(writer_mod, "TASK_PART_TEMPLATE", "tasks/task={task_type}/part-0.parquet")
    monkeypatch.setattr(parquet_writer, "SHARD_TEMPLATE", "time_series/shard-{:05d}.parquet")

    original = make_dataset()
    version_dir = _write(tmp_path, dataset=make_dataset())
    assert (version_dir / "samples.parquet").exists()  # legacy flat names on disk
    assert (version_dir / "time_series/shard-00000.parquet").exists()
    assert (version_dir / "tasks/task=classification/part-0.parquet").exists()
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
    assert isinstance(anns["stimulus"].span, PointSpan)
    assert isinstance(anns["artifact"].span, IntervalSpan)
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
    assert tasks["task-cls-2"].scope == IntervalSpan.seconds(0.0, 0.25, time_series_ids=("ts-window-2",))
    localization = tasks["task-localize-0"]
    assert isinstance(localization, TemporalLocalizationTask)
    assert localization.target == (
        PointSpan.seconds(0.5),  # a point: end_s stays None rather than becoming 0.0
        IntervalSpan.seconds(0.0, 0.25, time_series_ids=("ts-shared",)),
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
    index_path = version_dir / "time_series_index/part-00000000.parquet"
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
    (version_dir / "samples/part-00000000.parquet").unlink()
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
    samples_path = version_dir / "samples/part-00000000.parquet"
    table = pq.read_table(samples_path)
    pq.write_table(table.drop_columns(["start_time_us"]), samples_path)
    with TimeFReader(version_dir) as reader:
        assert all(s.start_time is None for s in reader.iter_samples())


def _corrupt_first_series(version_dir, field, value):
    """Set a field on the first series' struct in samples/part-00000000.parquet, simulating on-disk corruption."""
    samples_path = version_dir / "samples/part-00000000.parquet"
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
    ann_path = version_dir / "annotations/part-00000000.parquet"
    table = pq.read_table(ann_path)
    rows = table.to_pylist()
    for row in rows:
        # Only stretch an interval's end; nulling a point's would change its shape instead.
        if row["span"] is not None and row["span"]["end_us"] is not None:
            row["span"]["end_us"] = 10**15  # far past any series window
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), ann_path)
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="falls outside sample"):
        list(reader.iter_samples())


# ---- index part skip ---------------------------------------------------------------------------


def test_index_skip_loads_only_covering_parts(tmp_path, monkeypatch):
    version_dir = _write(
        tmp_path, dataset=make_dataset(), control_shard_target_bytes=1, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(version_dir) as reader:
        index_parts = {str(version_dir / rel) for rel in reader._manifest.files.time_series_index}
        assert len(index_parts) > 1
        seen: list[str] = []
        original = parts_mod.pq.read_table
        monkeypatch.setattr(parts_mod.pq, "read_table", lambda p, *a, **k: seen.append(str(p)) or original(p, *a, **k))
        sample = next(reader.iter_samples())
        values = sample.time_series[0].loader().to_numpy(zero_copy_only=False)
        assert len(values) > 0
    touched = [s for s in seen if s in index_parts]
    assert 0 < len(touched) < len(index_parts)


def test_index_straddle_still_resolves(tmp_path):
    original = make_dataset()
    version_dir = _write(
        tmp_path, dataset=make_dataset(), control_shard_target_bytes=1, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


def test_missing_index_part_stats_raises(tmp_path):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["part_stats"].pop("time_series_index")
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(TimeFFormatError):
        TimeFReader(version_dir)


def test_overlapping_index_parts_raise(tmp_path):
    version_dir = _write(tmp_path, dataset=make_dataset(), control_shard_target_bytes=1, chunk_max_bytes=64)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    entries = data["part_stats"]["time_series_index"]
    if len(entries) > 1:  # force a strict overlap: part 0 last key jumps past part 1 first key
        entries[0]["last_key"] = entries[-1]["last_key"]
        manifest_path.write_text(json.dumps(data))
        with pytest.raises(TimeFFormatError):
            TimeFReader(version_dir)


def test_get_sample_by_index_and_id_match_iteration(tmp_path):
    # A 1-byte control target puts every sample in its own part, so the offset/id-range skip has
    # several parts to pick between.
    version_dir = _write(tmp_path, dataset=make_dataset(), control_shard_target_bytes=1)
    with TimeFReader(version_dir) as reader:
        streamed = list(reader.iter_samples())
        assert len(reader) == len(streamed)
        for i, expected in enumerate(streamed):
            assert reader.get_sample(i).sample_id == expected.sample_id
            assert reader.get_sample_by_id(expected.sample_id).sample_id == expected.sample_id


def test_get_sample_out_of_range_and_unknown_id(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        with pytest.raises(IndexError):
            reader.get_sample(len(reader))
        with pytest.raises(IndexError):
            reader.get_sample(-1)
        with pytest.raises(KeyError):
            reader.get_sample_by_id("no-such-sample")


def test_get_sample_by_id_loads_only_the_covering_part(tmp_path, monkeypatch):
    version_dir = _write(tmp_path, dataset=make_dataset(), control_shard_target_bytes=1)
    with TimeFReader(version_dir) as reader:
        sample_parts = {str(version_dir / rel) for rel in reader._manifest.files.samples}
        assert len(sample_parts) > 1
        wanted = list(reader.iter_samples())[1].sample_id
        seen: list[str] = []
        original = parts_mod.pq.read_table
        monkeypatch.setattr(parts_mod.pq, "read_table", lambda p, *a, **k: seen.append(str(p)) or original(p, *a, **k))
        assert reader.get_sample_by_id(wanted).sample_id == wanted
    touched = [s for s in seen if s in sample_parts]
    assert 0 < len(touched) < len(sample_parts)
