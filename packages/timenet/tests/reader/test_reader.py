from fractions import Fraction
import json
from pathlib import Path
import pickle
from typing import cast

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    Domain,
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert isinstance(restored, TimeFDataset)
    assert_datasets_equal(original, restored)


def _registered_annotation_dataset() -> TimeFDataset:
    """A recording-as-sample dataset whose task references an annotation no sample carries.

    Two calls produce equal datasets (fixed ids), so it works as a round-trip fixture.

    Returns:
        The dataset.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/registered-annotation",
            dataset_version=Version(1, 0, 0),
            name="Registered",
            description="A task references an annotation no sample carries.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        channel="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    sample = dataset.add_sample(time_series=(series,), sample_id="rec-0")
    options = Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno")
    dataset.register_annotations([options])
    dataset.add_task(
        sample,
        AnswerTask(prompt="Rhythm?", target="yes", input_annotation_ids=(options.id,), id="qa-0"),
    )
    return dataset


def test_registered_annotation_round_trips(tmp_path):
    original = _registered_annotation_dataset()
    version_dir = _write(tmp_path, dataset=_registered_annotation_dataset())
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)  # compares registered_annotations too
    assert [ann.id for ann in restored.registered_annotations] == ["opts-yesno"]
    assert restored.tasks[0].input_annotation_ids == ("opts-yesno",)  # the ref survived, resolves in the table


def test_write_rejects_an_annotation_both_registered_and_sample_carried(tmp_path):
    # A registered annotation writes with empty sample_ids and the reader restores it from that; an id
    # also carried by a sample would write non-empty and be lost on read, so the writer rejects it.
    dataset = _registered_annotation_dataset()
    dataset.samples[0].add_annotations([Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno")])
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError, match="both registered and carried by a sample"):
        _write(tmp_path, dataset=dataset)


def _tasks_dataset(*, streaming: bool) -> TimeFDataset:
    """A recording-as-sample dataset whose QA tasks are added batched or via a stream (same content).

    Args:
        streaming: Feed the tasks through :meth:`TimeFDataset.set_task_stream` when true, else
            :meth:`add_tasks`.

    Returns:
        The dataset.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/streamed-tasks",
            dataset_version=Version(1, 0, 0),
            name="Streamed",
            description="Many QA tasks over one recording.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        channel="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    sample = dataset.add_sample(time_series=(series,), sample_id="rec-0")
    options = Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno")
    dataset.register_annotations([options])
    prompts = [f"Question {i}?" for i in range(5)]
    if streaming:
        tasks = [
            AnswerTask(
                prompt=prompt,
                target="yes",
                rationale=f"reason {i}",
                input_annotation_ids=(options.id,),
                id=f"qa-{i}",
                sample_ids=(sample.sample_id,),  # streamed tasks carry their own sample_ids
            )
            for i, prompt in enumerate(prompts)
        ]
        dataset.set_task_stream([AnswerTask], lambda tasks=tasks: iter(tasks))
    else:
        dataset.add_tasks(
            sample,
            [
                AnswerTask(
                    prompt=prompt,
                    target="yes",
                    rationale=f"reason {i}",
                    input_annotation_ids=(options.id,),
                    id=f"qa-{i}",
                )
                for i, prompt in enumerate(prompts)
            ],
        )
    return dataset


def _read(version_dir: Path) -> TimeFDataset:
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        return reader.read()


def test_streaming_tasks_round_trip(tmp_path):
    restored = _read(_write(tmp_path, dataset=_tasks_dataset(streaming=True)))
    assert len(restored.tasks) == 5
    task = next(t for t in restored.tasks if t.id == "qa-3")
    assert task.prompt == "Question 3?"
    assert task.target == "yes"
    assert task.rationale == "reason 3"
    assert task.sample_ids == ("rec-0",)
    assert task.input_annotation_ids == ("opts-yesno",)
    assert [ann.id for ann in restored.registered_annotations] == ["opts-yesno"]
    # After read(), streamed tasks back-populate their samples, so tasks_for() resolves them (this is
    # what tasks_for and the torch view rely on).
    rec0 = restored.samples[0]
    expected = {t.id for t in restored.tasks if rec0.sample_id in t.sample_ids}
    assert expected  # the sample really does carry streamed tasks
    assert {t.id for t in restored.tasks_for(rec0)} == expected


def test_streaming_tasks_match_batched(tmp_path):
    # The same tasks, added batched vs streamed, read back to the same tasks.
    batched = _read(_write(tmp_path / "batch", dataset=_tasks_dataset(streaming=False)))
    streamed = _read(_write(tmp_path / "stream", dataset=_tasks_dataset(streaming=True)))
    assert {t.id: t for t in batched.tasks} == {t.id: t for t in streamed.tasks}


def test_streaming_writer_writes_a_single_task_from_a_one_shot_source(tmp_path):
    # A source that hands out one non-re-iterable iterator (against the contract) must still write its
    # only task: iterating twice would let the peek consume it and the write find nothing.
    dataset = _tasks_dataset(streaming=True)
    one_shot = iter(list(dataset.iter_tasks())[:1])
    dataset._task_stream = lambda: one_shot
    restored = _read(_write(tmp_path, dataset=dataset))
    assert len(restored.tasks) == 1


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_round_trip_with_chunk_splitting(tmp_path, backend):
    original = make_dataset()
    version_dir = _write(
        tmp_path, dataset=make_dataset(), values_backend=backend, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_steps_matches_full_series_slice(tmp_path, backend):
    version_dir = _write(
        tmp_path, dataset=make_dataset(), values_backend=backend, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        series = next(iter(reader.iter_samples())).time_series[0]
        expected = series.to_arrow().slice(3, 7)
        assert series.read_steps(3, 10).equals(expected)


def test_metadata_and_schema(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader.metadata.dataset_id == "timenet/hello-world"
        assert {s.spec_type for s in reader.schema.time_series_specs} == {"sine", "cosine"}
        assert ClassificationTask in reader.schema.tasks


def test_shared_series_distinct_objects_same_id(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        samples = {s.sample_id: s for s in reader.read().samples}
    a = next(ts for ts in samples["sample-0"].time_series if ts.time_series_id == "ts-shared")
    b = next(ts for ts in samples["sample-1"].time_series if ts.time_series_id == "ts-shared")
    assert a is not b
    assert a.to_arrow().equals(b.to_arrow())


def test_annotation_value_types_round_trip(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        task = {t.id: t for t in reader.tasks}["task-scalar-0"]
    assert isinstance(task, ScalarPredictionTask)
    assert task.target == pytest.approx(62.0)
    assert task.unit == "bpm" and task.target_name == "mean_rate"


def test_iter_samples_matches_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        streamed = {s.sample_id for s in reader.iter_samples()}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    dataset = TimeFReader(DatasetVersion.open_local(version_dir)).read()
    first = dataset.samples[0].time_series[0]
    original = first.to_arrow()  # populates the values backend's caches

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.samples[0].time_series[0].to_arrow().equals(original)


def test_tasks_are_decoded_on_first_access_and_cached(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    reads: list[str] = []
    original_read_table = pq.read_table

    def counting_read_table(source, *args, **kwargs):
        reads.append(str(source))
        return original_read_table(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", counting_read_table)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader._tasks is None
        assert not [p for p in reads if "/tasks/" in p]  # construction decoded no task partition
        first = reader.tasks
        opened = [p for p in reads if "/tasks/" in p]
        assert opened  # first access decodes the task partitions
        assert reader.tasks is first  # second access reuses the cached tuple
        assert [p for p in reads if "/tasks/" in p] == opened


# ---- filtered sample reads --------------------------------------------------------------------


def test_iter_samples_returns_only_the_requested_ids(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        selected = list(reader.iter_samples(sample_ids=["sample-2", "sample-0"]))
    assert [s.sample_id for s in selected] == ["sample-0", "sample-2"]  # stored order, not asked order
    assert selected[0].time_series[0].to_arrow() is not None


def test_iter_samples_with_a_single_id_still_reads_its_values(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        expected = {ts.time_series_id: ts.to_arrow() for ts in next(iter(reader.iter_samples())).time_series}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (only,) = reader.iter_samples(sample_ids=["sample-0"])
        assert {ts.time_series_id: ts.to_arrow() for ts in only.time_series}.keys() == expected.keys()
        for ts in only.time_series:
            assert ts.to_arrow().equals(expected[ts.time_series_id])


def test_iter_samples_with_an_unknown_id_raises(tmp_path):
    version_dir = _write(tmp_path)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFValidationError, match="no such sample"),
    ):
        list(reader.iter_samples(sample_ids=["sample-0", "sample-nope"]))


def test_iter_samples_with_an_empty_id_list_yields_nothing(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert list(reader.iter_samples(sample_ids=[])) == []


def test_iter_samples_unknown_id_raises_on_full_consumption(tmp_path):
    # The guarantee holds when the iterator is drained; an early-stopping consumer is served what
    # exists and never reaches the check, which the docstring states explicitly.
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        got = next(reader.iter_samples(sample_ids=["sample-0", "no-such-sample"]))
        assert got.sample_id == "sample-0"  # early stop: no raise
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFValidationError, match="no-such-sample"),
    ):
        list(reader.iter_samples(sample_ids=["sample-0", "no-such-sample"]))


# ---- pruning ----------------------------------------------------------------------------------


def _count_control_plane_reads(monkeypatch) -> list[str]:
    """Record every file the reader opens or decodes, whole-table, filtered, or by handle."""
    read: list[str] = []
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
    reader = TimeFReader(DatasetVersion.open_local(version_dir))
    assert read == []  # only manifest.json, which is plain JSON
    assert reader._tasks is None
    assert reader._annotations is None
    assert reader._index is None
    assert reader.metadata.dataset_id  # metadata comes from the manifest, still free


def test_annotations_are_decoded_only_when_a_sample_resolves_them(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    read = _count_control_plane_reads(monkeypatch)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert not [p for p in read if "/annotations/" in p]
        sample = next(iter(reader.iter_samples()))
        assert sample.annotations
        assert [p for p in read if "/annotations/" in p]


def test_annotation_lookup_decodes_only_the_row_groups_that_can_match(tmp_path):
    # A small row-group target splits the annotations, so the id statistics have something to rule out.
    # Resolving one annotation id must decode only the group that can hold it, not the whole file.
    version_dir = _write(tmp_path, row_group_target_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        annotations = reader._annotations_table()
        total = len(annotations.groups())
        assert total > 1, "the fixture must span several annotation row groups for this to mean anything"
        assert reader._resolve_annotation("sample-0", "age-0").value == 64  # one id, one row group decoded
        assert len(annotations._cache) < total


def test_annotation_resolution_builds_no_whole_table_id_map(tmp_path):
    # The old reader built a {id: Annotation} map over the whole table on open. The pruned reader visits
    # only the row group(s) that can hold the id and caches just the decoded annotation, never the table.
    # A small row-group target makes the pruning observable, so this cannot pass with pruning broken.
    version_dir = _write(tmp_path, row_group_target_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        annotations = reader._annotations_table()
        total = len(annotations.groups())
        assert total > 1, "the fixture must span several annotation row groups for this to mean anything"
        assert reader._resolve_annotation("sample-0", "age-0").value == 64
        assert len(annotations._cache) < total  # only the matching group was decoded, not the whole table
        assert list(reader._annotation_cache) == ["age-0"]  # one decoded annotation cached, not an id map
        for table, keys in annotations._cache.values():
            assert len(keys) == table.num_rows  # each cache entry is one group's rows, not a per-id map


def test_index_lookup_decodes_only_the_row_groups_that_can_match(tmp_path):
    # A small row-group target splits the index, so the sample_id statistics have something to rule out.
    # Without pruning, one lookup would decode every row group in the file.
    version_dir = _write(tmp_path, row_group_target_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample = next(iter(reader.iter_samples()))
        for series in sample.time_series:
            series.to_arrow()
        index = reader._index_table()
        total = len(index.groups())
        assert total > 1, "the fixture must span several row groups for this to mean anything"
        assert len(index._cache) < total


def test_index_row_groups_are_pruned_by_their_statistics(tmp_path):
    version_dir = _write(tmp_path, row_group_target_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        probe = cast("str | bytes", reader._codec.encode("sample_id", "sample-0"))  # the stored form pruned on
        groups = reader._index_table().groups()
        assert all(g.min_key is not None for g in groups), "the index must carry sample_id statistics"
        assert [g.may_hold(probe) for g in groups].count(False) > 0  # some groups provably cannot match


def test_getstate_drops_every_control_plane_cache(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        next(iter(reader.iter_samples())).time_series[0].to_arrow()
        _ = reader.tasks
        state = reader.__getstate__()
    assert state["_tasks"] is None
    assert state["_annotations"] is None
    assert state["_annotation_cache"] == {}
    assert state["_index"] is None
    assert state["_samples_data"] is None
    assert state["_values"] is None


def test_shuffled_sample_access_does_not_thrash_the_index_cache(tmp_path):
    # A byte-bounded index cache holds the same working set for in-order and shuffled reads. A
    # count-bounded cache evicted and re-decoded a whole row group per lookup; this asserts the cache
    # retains groups rather than timing anything.
    version_dir = _write(tmp_path, row_group_target_bytes=64)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample_ids = [s.sample_id for s in reader.iter_samples()]
        series = {s.sample_id: [ts.time_series_id for ts in s.time_series] for s in reader.iter_samples()}
        index = reader._index_table()
        assert len(index.groups()) > 1  # the small row-group target really did split the index

        for sample_id in reversed(sample_ids):  # reverse order is the cheapest stand-in for shuffled
            for series_id in series[sample_id]:
                assert reader._index_rows(sample_id, series_id)
        # every decoded group is still resident: nothing was evicted to serve the pass
        assert len(index._cache) == len(index.groups())


# ---- validation -------------------------------------------------------------------------------


def test_missing_root_raises(tmp_path):
    # Building the handle reads manifest.json; a directory that does not exist has no version to open.
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(tmp_path / "empty")


@pytest.mark.parametrize("format_version", [2, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(TimeFFormatError):
        TimeFReader(DatasetVersion.open_local(version_dir))


def test_corrupt_index_locator_has_series_context(tmp_path):
    # Corrupt the artifact rather than the reader's internals: the locator is read from the index on
    # each lookup now, so an in-memory poke would not survive to the read.
    version_dir = _write(tmp_path)
    index_path = version_dir / "time_series_index/part-00000000.parquet"
    table = pq.read_table(index_path)
    bogus = pa.array(["time_series/does-not-exist.parquet"] * table.num_rows)
    pq.write_table(table.set_column(table.schema.get_field_index("chunk_file"), "chunk_file", bogus), index_path)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample = next(iter(reader.iter_samples()))
        series_id = sample.time_series[0].time_series_id
        with pytest.raises(
            TimeFFormatError, match=f"failed to read series {series_id!r} for sample {sample.sample_id!r}"
        ):
            reader._load_values(sample.sample_id, series_id)


def test_corrupt_index_data_page_raises_format_error(tmp_path):
    # A footer that parses but a garbage data page: the lazy index decode must surface as
    # TimeFFormatError, not a raw OSError from pyarrow, or a caller catching corruption misses it.
    version_dir = _write(tmp_path)
    idx = version_dir / "time_series_index/part-00000000.parquet"
    raw = bytearray(idx.read_bytes())
    for i in range(4, min(64, len(raw) - 8)):
        raw[i] = 0
    idx.write_bytes(raw)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        sample = next(iter(reader.iter_samples()))
        with pytest.raises(TimeFFormatError):
            sample.time_series[0].to_arrow()


def test_missing_listed_file_fails_lazily_on_first_access(tmp_path):
    # __init__ no longer stat-sweeps (an O(files) HEAD storm on an object store); a missing file now
    # surfaces on the first read that touches it, which the lazy stack already accepts.
    version_dir = _write(tmp_path)
    (version_dir / "samples/part-00000000.parquet").unlink()
    reader = TimeFReader(DatasetVersion.open_local(version_dir))  # open is happy: it never touches samples
    with pytest.raises(FileNotFoundError):
        list(reader.iter_samples())


def test_verify_passes_on_an_intact_dataset(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        reader.verify()  # must not raise


def test_verify_detects_a_corrupted_shard(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        # a shard: read lazily, so __init__ still succeeds and verify() is what catches it
        rel = next(p.path for p in reader._manifest.files.time_series if p.path.startswith("time_series/part-"))
    target = version_dir / rel
    original = target.read_bytes()
    # Same size, flipped content: exercises the checksum check, not the size check verify() also does.
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="checksum mismatch"),
    ):
        reader.verify()


def test_verify_detects_a_size_mismatch(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        rel = next(p.path for p in reader._manifest.files.time_series if p.path.startswith("time_series/part-"))
    target = version_dir / rel
    target.write_bytes(target.read_bytes()[:-1])  # truncate, so size disagrees before a checksum is even computed
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="size mismatch"),
    ):
        reader.verify()


def test_verify_detects_a_deleted_file(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        rel = next(p.path for p in reader._manifest.files.time_series if p.path.startswith("time_series/part-"))
    (version_dir / rel).unlink()
    # A value shard is read lazily, so __init__ no longer stat-sweeps it; verify() reopens every listed
    # file through the handle and is where a deleted one now surfaces.
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="missing file"),
    ):
        reader.verify()


def test_corrupt_task_partition_raises_format_error_on_first_task_access(tmp_path):
    # An unknown task partition name is corrupt on-disk data, so it must surface as TimeFFormatError
    # rather than the bare ValueError that TaskType() happens to raise. Tasks decode on first access,
    # so that is where it surfaces; construction never touches the task partition.
    version_dir = _write(tmp_path)
    tasks_dir = next((version_dir / "tasks").iterdir())
    tasks_dir.rename(tasks_dir.parent / "task=not_a_real_task_type")
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tasks"] = [
        {**part, "path": part["path"].replace(tasks_dir.name, "task=not_a_real_task_type")}
        for part in manifest["files"]["tasks"]
    ]
    manifest_path.write_text(json.dumps(manifest))
    reader = TimeFReader(DatasetVersion.open_local(version_dir))  # construction does not touch the task partition
    with pytest.raises(TimeFFormatError):
        _ = reader.tasks


def test_start_time_round_trips_exactly(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert all(s.start_time is None for s in reader.iter_samples())


def _corrupt_first_series(version_dir, field, value):
    """Set a field on the first series' struct in samples.parquet, simulating on-disk corruption."""
    samples_path = version_dir / "samples/part-00000000.parquet"
    table = pq.read_table(samples_path)
    rows = table.to_pylist()
    rows[0]["time_series"][0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples_path)


def test_ordinal_row_carrying_regular_columns_raises_format_error(tmp_path):
    # The tag and the columns disagree: an ordinal series must have no period or start index.
    version_dir = _write(tmp_path)
    _corrupt_first_series(version_dir, "axis_type", "ordinal")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="carries regular- or"),
    ):
        list(reader.iter_samples())


def test_regular_row_with_a_zero_denominator_raises_format_error(tmp_path):
    # A zero denominator would raise a raw ZeroDivisionError from Fraction; it must surface as format error.
    version_dir = _write(tmp_path)
    _corrupt_first_series(version_dir, "period_denominator", 0)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="unbuildable regular axis"),
    ):
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
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="decodes to shape"),
    ):
        list(reader.iter_samples())


def test_annotation_value_type_disagreeing_with_its_descriptor_raises_format_error(tmp_path):
    # "age" is an int; a descriptor that calls it a str no longer matches the decoded value.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "age", "value_type", "str")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="value type"),
    ):
        list(reader.iter_samples())


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
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="falls outside sample"),
    ):
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
    samples_path = version_dir / "samples/part-00000000.parquet"
    table = pq.read_table(samples_path)
    rows = table.to_pylist()
    rows[0]["time_span"] = struct
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), samples_path)


def test_time_span_with_reversed_bounds_raises_format_error(tmp_path):
    # A stored time_span whose end is not past its start fails Span validation on decode; it must surface
    # as a format error, not the raw ValueError that validation raises.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_first_time_span(version_dir, {"start_us": 6_000_000, "end_us": 5_000_000, "time_series_ids": None})
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="must be > start"),
    ):
        list(reader.iter_samples())


def test_point_shaped_time_span_raises_format_error(tmp_path):
    # A time_span must be an interval covering the whole sample. A corrupt point-shaped one (no end) is
    # rejected as a format error, not left to reach the unscoped-span check and raise a bare TypeError.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_first_time_span(version_dir, {"start_us": 0, "end_us": None, "time_series_ids": None})
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="must be a TimeInterval"),
    ):
        list(reader.iter_samples())


def test_task_partition_missing_optional_column_reads_as_none(tmp_path):
    """A task partition written before an optional column existed reads back with that field None.

    Task payload columns are derived from the live dataclass, so adding an optional field to a task
    type would make every already-published partition of that type raise on read. The reader must
    tolerate a missing payload column instead, so additive task fields never force a recuration.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/defensive-reads",
            dataset_version=Version(1, 0, 0),
            name="Defensive",
            description="One classification task carrying an optional schema column.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        channel="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    sample = dataset.add_sample(time_series=(series,), sample_id="rec-0")
    dataset.add_task(sample, ClassificationTask(target="afib", target_schema="scp5", id="cls-0"))
    version_dir = _write(tmp_path, dataset=dataset)

    # Simulate an older partition: drop the optional payload column from the task Parquet on disk.
    task_parquets = [p for p in version_dir.rglob("*.parquet") if "task=classification" in str(p)]
    assert len(task_parquets) == 1
    table = pq.read_table(task_parquets[0])
    assert "target_schema" in table.column_names
    pq.write_table(table.drop_columns(["target_schema"]), task_parquets[0])

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    task = restored.tasks[0]
    assert isinstance(task, ClassificationTask)
    assert task.target == "afib"
    assert task.target_schema is None
