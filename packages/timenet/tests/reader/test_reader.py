import json
from pathlib import Path
import pickle

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
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
from timenet.writer import TimeFWriter
import timenet.writer.writer as writer_module


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


# ---- filtered sample reads --------------------------------------------------------------------


def test_iter_samples_returns_only_the_requested_ids(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        selected = list(reader.iter_samples(sample_ids=["sample-2", "sample-0"]))
    assert [s.sample_id for s in selected] == ["sample-0", "sample-2"]  # stored order, not asked order
    assert selected[0].time_series[0].to_arrow() is not None


def test_iter_samples_with_a_single_id_still_reads_its_values(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        expected = {ts.time_series_id: ts.to_arrow() for ts in next(iter(reader.iter_samples())).time_series}
    with TimeFReader(version_dir) as reader:
        (only,) = reader.iter_samples(sample_ids=["sample-0"])
        assert {ts.time_series_id: ts.to_arrow() for ts in only.time_series}.keys() == expected.keys()
        for ts in only.time_series:
            assert ts.to_arrow().equals(expected[ts.time_series_id])


def test_iter_samples_with_an_unknown_id_raises(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFValidationError, match="no such sample"):
        list(reader.iter_samples(sample_ids=["sample-0", "sample-nope"]))


def test_iter_samples_with_an_empty_id_list_yields_nothing(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        assert list(reader.iter_samples(sample_ids=[])) == []


# ---- laziness ---------------------------------------------------------------------------------


def _count_control_plane_reads(monkeypatch) -> list[str]:
    """Record every file the reader opens or decodes, whole-table, filtered, or by handle."""
    read = []
    original_read_table, original_dataset, original_file = pq.read_table, pads.dataset, pq.ParquetFile

    def counting_read_table(source, *args, **kwargs):
        read.append(str(source))
        return original_read_table(source, *args, **kwargs)

    def counting_dataset(source, *args, **kwargs):
        read.extend(str(p) for p in source) if isinstance(source, list) else read.append(str(source))
        return original_dataset(source, *args, **kwargs)

    def counting_file(source, *args, **kwargs):
        read.append(str(source))
        return original_file(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", counting_read_table)
    monkeypatch.setattr(pads, "dataset", counting_dataset)
    monkeypatch.setattr(pq, "ParquetFile", counting_file)
    return read


def test_open_decodes_no_control_plane_table(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    read = _count_control_plane_reads(monkeypatch)
    reader = TimeFReader(version_dir)
    assert read == []  # only manifest.json, which is plain JSON
    assert reader._tasks is None
    assert reader._annotation_table is None
    assert reader._index_directory is None
    assert reader.metadata.dataset_id  # metadata comes from the manifest, still free


def test_tasks_are_decoded_on_first_access_and_cached(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    read = _count_control_plane_reads(monkeypatch)
    with TimeFReader(version_dir) as reader:
        assert read == []
        first = reader.tasks
        opened = [p for p in read if "/tasks/" in p]
        assert opened  # first access decodes the task partitions
        assert reader.tasks is first  # second access re-uses the cached tuple
        assert [p for p in read if "/tasks/" in p] == opened


def test_annotations_are_decoded_only_when_a_sample_resolves_them(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    read = _count_control_plane_reads(monkeypatch)
    with TimeFReader(version_dir) as reader:
        assert not [p for p in read if p.endswith("annotations.parquet")]
        sample = next(iter(reader.iter_samples()))
        assert sample.annotations
        assert [p for p in read if p.endswith("annotations.parquet")]


def test_index_lookup_decodes_only_the_row_groups_that_can_match(tmp_path, monkeypatch):
    # One row group per two index rows, so the statistics on the sample_id column have something to
    # rule out. Without pruning, one lookup would decode every row group in the file.
    monkeypatch.setattr(writer_module, "DEFAULT_CONTROL_PLANE_BATCH_ROWS", 2)
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        sample = next(iter(reader.iter_samples()))
        for series in sample.time_series:
            series.to_arrow()
        total = len(reader._index_groups())
        assert total > 1, "the fixture must span several row groups for this to mean anything"
        assert len(reader._index_cache) < total


def test_index_row_groups_are_pruned_by_their_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, "DEFAULT_CONTROL_PLANE_BATCH_ROWS", 2)
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        groups = reader._index_groups()
        assert all(g.min_sample_id is not None for g in groups), "the index must carry sample_id statistics"
        assert [g.may_hold("sample-0") for g in groups].count(False) > 0


def test_getstate_drops_every_control_plane_cache(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        next(iter(reader.iter_samples())).time_series[0].to_arrow()
        _ = reader.tasks
        state = reader.__getstate__()
    assert state["_tasks"] is None
    assert state["_annotation_table"] is None
    assert state["_annotation_rows"] is None
    assert state["_annotation_cache"] == {}
    assert state["_index_directory"] is None
    assert state["_index_bisectable"] is False
    assert state["_index_files"] == {}
    assert state["_index_cache"] == {}
    assert state["_samples_data"] is None
    assert state["_values"] is None


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


def test_corrupt_task_partition_raises_format_error_on_first_task_access(tmp_path):
    # An unknown task partition name is corrupt on-disk data, so it must surface as TimeFFormatError
    # rather than the bare ValueError that TaskType() happens to raise. Tasks are decoded on first
    # access, so that is where it surfaces — construction no longer touches the partition.
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
    reader = TimeFReader(version_dir)  # construction is happy: it read only the manifest
    with pytest.raises(TimeFFormatError):
        _ = reader.tasks


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
    # Annotations decode when a sample resolves them, so that is where the disagreement surfaces.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "artifact", "annotation_type", "static")
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="decodes to shape"):
        list(reader.iter_samples())


def test_annotation_value_type_disagreeing_with_its_descriptor_raises_format_error(tmp_path):
    # "age" is an int; a descriptor that calls it a str no longer matches the decoded value.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "age", "value_type", "str")
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="value type"):
        list(reader.iter_samples())


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


def test_shuffled_sample_access_does_not_thrash_the_index_cache(tmp_path, monkeypatch):
    # TNET-84 names shuffled-epoch reads as the standard sleep-staging pattern. A count-bounded index
    # cache made them 6.1x slower than in-order at 24 row groups, because each lookup evicted, decoded
    # a whole row group and rebuilt its offset map. The cache is bounded by bytes so both patterns hold
    # the same working set. This asserts the cache retains groups rather than timing anything.
    monkeypatch.setattr(writer_module, "DEFAULT_CONTROL_PLANE_BATCH_ROWS", 4)
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        sample_ids = [s.sample_id for s in reader.iter_samples()]
        series = {s.sample_id: [ts.time_series_id for ts in s.time_series] for s in reader.iter_samples()}
        assert len(reader._index_groups()) > 1  # the batch size really did split the index

        for sample_id in reversed(sample_ids):  # reverse order is the cheapest stand-in for shuffled
            for series_id in series[sample_id]:
                assert reader._index_rows(sample_id, series_id)
        # every decoded group is still resident: nothing was evicted to serve the pass
        assert len(reader._index_cache) == len(reader._index_groups())


def test_corrupt_index_data_page_raises_format_error(tmp_path):
    # A footer that parses but a garbage data page: the lazy index decode must surface as
    # TimeFFormatError, not a raw OSError from pyarrow, or a caller catching corruption misses it.
    version_dir = _write(tmp_path)
    idx = version_dir / "time_series_index.parquet"
    raw = bytearray(idx.read_bytes())
    for i in range(4, min(64, len(raw) - 8)):
        raw[i] = 0
    idx.write_bytes(raw)
    with TimeFReader(version_dir) as reader:
        sample = next(iter(reader.iter_samples()))
        with pytest.raises(TimeFFormatError):
            sample.time_series[0].to_arrow()


def test_iter_samples_unknown_id_raises_on_full_consumption(tmp_path):
    # The guarantee holds when the iterator is drained; an early-stopping consumer is served what
    # exists and never reaches the check, which the docstring now states explicitly.
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        got = next(reader.iter_samples(sample_ids=["sample-0", "no-such-sample"]))
        assert got.sample_id == "sample-0"  # early stop: no raise
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFValidationError, match="no-such-sample"):
        list(reader.iter_samples(sample_ids=["sample-0", "no-such-sample"]))
