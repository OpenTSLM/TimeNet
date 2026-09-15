from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
import pickle

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.control_plane.reader import ControlPlaneReader
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import SpanOutsideWindowWarning, TimeFFormatError, TimeFValidationError
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
    TSCorrespondenceTask,
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


@contextmanager
def _control_db(version_dir: Path):
    """Open a committed version's control database read-write, to simulate on-disk corruption."""
    connection = duckdb.connect(str(version_dir / "control.duckdb"))
    try:
        yield connection
    finally:
        connection.close()


def _record_key(reader: TimeFReader, record_id: str) -> int:
    """Return one record's surrogate id, which is what the reader's internals join on."""
    found, _missing = reader._control_plane().resolve_record_ids([record_id])
    return found[0]


def _spy(monkeypatch, method: str) -> list:
    """Record every call to one ControlPlaneReader method, and return the recorded arguments."""
    calls: list = []
    original = getattr(ControlPlaneReader, method)

    def counting(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ControlPlaneReader, method, counting)
    return calls


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
    """A recording-as-record dataset whose task references an annotation no record carries.

    Two calls produce equal datasets (fixed ids), so it works as a round-trip fixture.

    Returns:
        The dataset.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/registered-annotation",
            dataset_version=Version(1, 0, 0),
            name="Registered",
            description="A task references an annotation no record carries.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    options = Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno")
    dataset.register_annotations([options])
    dataset.add_task(
        record,
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


def test_write_rejects_an_annotation_both_registered_and_record_carried(tmp_path):
    # A registered annotation writes with empty record_ids and the reader restores it from that; an id
    # also carried by a record would write non-empty and be lost on read, so the writer rejects it.
    dataset = _registered_annotation_dataset()
    dataset.records[0].add_annotations([Annotation(key="answer_options", value=["yes", "no"], id="opts-yesno")])
    dataset.derive_schema()
    with pytest.raises(TimeFValidationError, match="both registered and carried by a record"):
        _write(tmp_path, dataset=dataset)


def _tasks_dataset(*, streaming: bool) -> TimeFDataset:
    """A recording-as-record dataset whose QA tasks are added batched or via a stream (same content).

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
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
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
                record_ids=(record.record_id,),  # streamed tasks carry their own record_ids
            )
            for i, prompt in enumerate(prompts)
        ]
        dataset.set_task_stream([AnswerTask], lambda tasks=tasks: iter(tasks))
    else:
        dataset.add_tasks(
            record,
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
    assert task.record_ids == ("rec-0",)
    assert task.input_annotation_ids == ("opts-yesno",)
    assert [ann.id for ann in restored.registered_annotations] == ["opts-yesno"]
    # After read(), streamed tasks back-populate their records, so tasks_for() resolves them (this is
    # what tasks_for and the torch view rely on).
    rec0 = restored.records[0]
    expected = {t.id for t in restored.tasks if rec0.record_id in t.record_ids}
    assert expected  # the record really does carry streamed tasks
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
        series = next(iter(reader.iter_records())).time_series[0]
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
        records = {s.record_id: s for s in reader.read().records}
    a = next(ts for ts in records["record-0"].time_series if ts.time_series_id == "ts-shared")
    b = next(ts for ts in records["record-1"].time_series if ts.time_series_id == "ts-shared")
    assert a is not b
    assert a.to_arrow().equals(b.to_arrow())


def test_annotation_value_types_round_trip(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        records = {s.record_id: s for s in reader.read().records}
    anns = {a.key: a for a in records["record-0"].annotations}
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
    record = dataset.records[0]
    dataset.add_task(record, AnswerTask(prompt="Any ectopy?", target="No.", id="task-answer-1"))  # rationale=None
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


def test_iter_records_matches_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        streamed = {s.record_id for s in reader.iter_records()}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        read_ids = {s.record_id for s in reader.read().records}
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
        ts = dataset.records[0].time_series[0]
        assert opens["n"] == 0  # building records opened no shards
        ts.to_arrow()
        assert opens["n"] >= 1  # reading values opened a shard


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_back_dataset_is_picklable(tmp_path, backend):
    # A multi-worker torch DataLoader pickles the dataset to each worker, so lazy loaders must pickle
    # even after values (and thus the backend's handles/caches) have been touched.
    version_dir = _write(tmp_path, values_backend=backend)
    dataset = TimeFReader(DatasetVersion.open_local(version_dir)).read()
    first = dataset.records[0].time_series[0]
    original = first.to_arrow()  # populates the values backend's caches

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.records[0].time_series[0].to_arrow().equals(original)


def test_tasks_are_decoded_on_first_access_and_cached(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    queries = _spy(monkeypatch, "tasks")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert reader._tasks is None
        assert not queries  # construction queried no task table
        first = reader.tasks
        assert len(queries) == 1  # first access decodes every task in one round of queries
        assert reader.tasks is first  # second access reuses the cached tuple
        assert len(queries) == 1


# ---- filtered record reads --------------------------------------------------------------------


def test_iter_records_returns_only_the_requested_ids(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        selected = list(reader.iter_records(record_ids=["record-2", "record-0"]))
    assert [s.record_id for s in selected] == ["record-0", "record-2"]  # stored order, not asked order
    assert selected[0].time_series[0].to_arrow() is not None


def test_iter_records_with_a_single_id_still_reads_its_values(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        expected = {ts.time_series_id: ts.to_arrow() for ts in next(iter(reader.iter_records())).time_series}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        (only,) = reader.iter_records(record_ids=["record-0"])
        assert {ts.time_series_id: ts.to_arrow() for ts in only.time_series}.keys() == expected.keys()
        for ts in only.time_series:
            assert ts.to_arrow().equals(expected[ts.time_series_id])


def test_iter_records_with_an_unknown_id_raises(tmp_path):
    version_dir = _write(tmp_path)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFValidationError, match="no such record"),
    ):
        list(reader.iter_records(record_ids=["record-0", "record-nope"]))


def test_iter_records_with_an_empty_id_list_yields_nothing(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert list(reader.iter_records(record_ids=[])) == []


def test_iter_records_unknown_id_raises_on_full_consumption(tmp_path):
    # The guarantee holds when the iterator is drained; an early-stopping consumer is served what
    # exists and never reaches the check, which the docstring states explicitly.
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        got = next(reader.iter_records(record_ids=["record-0", "no-such-record"]))
        assert got.record_id == "record-0"  # early stop: no raise
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFValidationError, match="no-such-record"),
    ):
        list(reader.iter_records(record_ids=["record-0", "no-such-record"]))


# ---- control-plane laziness -------------------------------------------------------------------


def test_open_queries_no_control_plane_table(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    connects: list = []
    original = duckdb.connect
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: connects.append(a) or original(*a, **k))
    reader = TimeFReader(DatasetVersion.open_local(version_dir))
    assert connects == []  # only manifest.json, which is plain JSON
    assert reader._control is None
    assert reader._tasks is None
    assert reader.metadata.dataset_id  # metadata comes from the manifest, still free


def test_annotations_are_read_only_when_a_record_resolves_them(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    queries = _spy(monkeypatch, "record_annotations")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        assert not queries
        record = next(iter(reader.iter_records()))
        assert record.annotations
        assert queries


def test_a_values_only_read_resolves_no_annotation(tmp_path, monkeypatch):
    # A caller that reads values pays a query and a JSON parse per annotation for data it never
    # touches. with_annotations=False must leave the annotations table unread.
    version_dir = _write(tmp_path)
    queries = _spy(monkeypatch, "record_annotations")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(iter(reader.iter_records(with_annotations=False)))
        assert record.annotations == ()
        assert not queries
        # The values are still there, so the record is usable for what the caller asked for.
        assert record.time_series
        assert len(record.time_series[0].to_numpy())


def test_a_values_only_read_keeps_every_other_record_field(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        full = {r.record_id: r for r in reader.iter_records()}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        lean = {r.record_id: r for r in reader.iter_records(with_annotations=False)}
    assert set(full) == set(lean)
    for record_id, whole in full.items():
        thin = lean[record_id]
        assert thin.subject_ids == whole.subject_ids
        assert thin.task_ids == whole.task_ids
        assert thin.start_time == whole.start_time
        assert thin.time_span == whole.time_span
        assert [ts.time_series_id for ts in thin.time_series] == [ts.time_series_id for ts in whole.time_series]
        for lean_series, whole_series in zip(thin.time_series, whole.time_series, strict=True):
            assert lean_series.to_numpy().tolist() == whole_series.to_numpy().tolist()


def test_a_record_reads_its_chunk_locators_once_for_all_of_its_series(tmp_path, monkeypatch):
    # One query returns every locator of a record, so a per-series lookup would pay a scan each.
    version_dir = _write(tmp_path)
    queries = _spy(monkeypatch, "record_chunks")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(iter(reader.iter_records(with_annotations=False)))
        for series in record.time_series:
            series.to_numpy()  # force the lazy loader, which is what asks for the locators
        assert len(record.time_series) > 1, "the fixture needs a multi-series record to mean anything"
        assert len(queries) == 1, f"one query per record, got {len(queries)} for one record"


def test_the_locator_memo_returns_what_the_control_plane_holds(tmp_path):
    # A small chunk cap splits one series across many chunks. A memo that ignored chunk order, or
    # the record it was keyed on, would still pass on a fixture with one chunk per series.
    version_dir = _write(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64)
    widest = 0
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        control = reader._control_plane()
        for record in reader.iter_records(with_annotations=False):
            key = _record_key(reader, record.record_id)
            stored = {}
            for row in control.record_chunks(key):
                stored.setdefault(row["time_series_id"], []).append(row)
            for series in record.time_series:
                memoized = reader._index_rows(key, series.time_series_id)
                assert memoized == stored[series.time_series_id]
                assert [row["chunk_idx"] for row in memoized] == sorted(row["chunk_idx"] for row in memoized)
                widest = max(widest, len(memoized))
    assert widest > 1, "the fixture needs a series split across several chunks"


def test_closing_the_reader_drops_the_locator_memo(tmp_path):
    version_dir = _write(tmp_path)
    reader = TimeFReader(DatasetVersion.open_local(version_dir))
    record = next(iter(reader.iter_records(with_annotations=False)))
    reader._index_rows(_record_key(reader, record.record_id), record.time_series[0].time_series_id)
    assert reader._index_rows_cache

    reader.close()
    assert not reader._index_rows_cache
    assert reader._control is None


def test_getstate_drops_every_control_plane_cache(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        next(iter(reader.iter_records())).time_series[0].to_arrow()
        _ = reader.tasks
        state = reader.__getstate__()
    assert state["_tasks"] is None
    assert state["_control"] is None  # a DuckDB connection does not cross a process boundary
    assert state["_values"] is None
    assert state["_axis_cache"] == {}
    assert state["_index_rows_cache"] == {}


# ---- validation -------------------------------------------------------------------------------


def test_missing_root_raises(tmp_path):
    # Building the handle reads manifest.json; a directory that does not exist has no version to open.
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        DatasetVersion.open_local(tmp_path / "empty")


@pytest.mark.parametrize("format_version", [3, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(TimeFFormatError):
        TimeFReader(DatasetVersion.open_local(version_dir))


def test_corrupt_chunk_locator_has_series_context(tmp_path):
    # Corrupt the artifact rather than the reader's internals: the locator is read from the control
    # plane on each lookup, so an in-memory poke would not survive to the read.
    version_dir = _write(tmp_path)
    with _control_db(version_dir) as db:
        db.execute("UPDATE values_artifacts SET chunk_file = 'time_series/does-not-exist.parquet'")

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(iter(reader.iter_records()))
        series_id = record.time_series[0].time_series_id
        with pytest.raises(
            TimeFFormatError, match=f"failed to read series {series_id!r} for record {record.record_id!r}"
        ):
            record.time_series[0].to_arrow()


def test_a_truncated_control_database_raises_format_error(tmp_path):
    # A file that is no longer a database: the lazy open must surface as TimeFFormatError, not a raw
    # duckdb.Error, or a caller catching corruption misses it.
    version_dir = _write(tmp_path)
    db_path = version_dir / "control.duckdb"
    db_path.write_bytes(db_path.read_bytes()[: 1 << 12])
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader, pytest.raises(TimeFFormatError):
        list(reader.iter_records())


def test_missing_listed_file_fails_lazily_on_first_access(tmp_path):
    # __init__ does not stat-sweep (an O(files) HEAD storm on an object store); a missing file
    # surfaces on the first read that touches it, which the lazy stack already accepts.
    version_dir = _write(tmp_path)
    (version_dir / "control.duckdb").unlink()
    reader = TimeFReader(DatasetVersion.open_local(version_dir))  # open is happy: it never touches it
    with pytest.raises(TimeFFormatError, match="no control database"):
        list(reader.iter_records())


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


def test_unknown_stored_task_type_raises_format_error_on_first_task_access(tmp_path):
    # An unknown task type is corrupt on-disk data, so it must surface as TimeFFormatError rather
    # than the bare ValueError that TaskType() happens to raise. Tasks decode on first access, so
    # that is where it surfaces; construction never touches the task tables.
    version_dir = _write(tmp_path)
    with _control_db(version_dir) as db:
        db.execute("UPDATE tasks SET task_type = 'not_a_real_task_type'")
    reader = TimeFReader(DatasetVersion.open_local(version_dir))  # construction does not read the tasks
    with pytest.raises(TimeFFormatError):
        _ = reader.tasks


def test_start_time_round_trips_exactly(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        records = {s.record_id: s for s in reader.iter_records()}
    # The fixture anchor is not representable in float64, so this catches any float coercion.
    assert records["record-0"].start_time == 9_007_199_254_740_993
    assert records["record-1"].start_time is None
    assert records["record-2"].start_time is None


def _corrupt_axes(version_dir, column, value):
    """Set one column on every stored axis, simulating on-disk corruption."""
    with _control_db(version_dir) as db:
        db.execute(f"UPDATE axes SET {column} = ?", [value])  # noqa: S608


def test_ordinal_row_carrying_regular_columns_raises_format_error(tmp_path):
    # The tag and the columns disagree: an ordinal series must have no period or start index.
    version_dir = _write(tmp_path)
    _corrupt_axes(version_dir, "axis_type", "ordinal")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="carries regular- or"),
    ):
        list(reader.iter_records())


def test_regular_row_with_a_zero_denominator_raises_format_error(tmp_path):
    # A zero denominator would raise a raw ZeroDivisionError from Fraction; it must surface as format error.
    version_dir = _write(tmp_path)
    _corrupt_axes(version_dir, "period_denominator", 0)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="unbuildable regular axis"),
    ):
        list(reader.iter_records())


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
    # Annotations decode when a record resolves them, so that is where the disagreement surfaces.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "artifact", "annotation_type", "static")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="decodes to shape"),
    ):
        list(reader.iter_records())


def test_annotation_value_type_disagreeing_with_its_descriptor_raises_format_error(tmp_path):
    # "age" is an int; a descriptor that calls it a str no longer matches the decoded value.
    version_dir = _write(tmp_path)
    _corrupt_descriptor(version_dir, "age", "value_type", "str")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="value type"),
    ):
        list(reader.iter_records())


def test_annotation_span_outside_the_series_warns_and_reads_back_unchanged(tmp_path):
    # The writer keeps a span that runs past its series, so the reader gives it back unchanged.
    version_dir = _write(tmp_path)
    with _control_db(version_dir) as db:
        # Only stretch an interval's end; nulling a point's would change its shape instead.
        db.execute("UPDATE annotations SET span_end_us = 1000000000000000 WHERE span_end_us IS NOT NULL")
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.warns(SpanOutsideWindowWarning, match="falls outside record"),
    ):
        records = list(reader.iter_records())
    # Only an interval was stretched; a point has no end to stretch.
    ends = [a.span.end_us for s in records for a in s.annotations if isinstance(a.span, TimeInterval)]
    assert 10**15 in ends, "the stretched span must survive the read"


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
        signal="c",
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=3,
        loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
    )
    record = dataset.add_record(time_series=(series,), time_span=TimeInterval.seconds(0.0, 5.0))
    record.add_annotation(Annotation(key="note", span=TimePoint.seconds(2.0)))  # unscoped, inside [0, 5) s
    return _write(tmp_path, dataset)


def _corrupt_time_span(version_dir, start_us, end_us):
    """Replace every record's stored time_span bounds, simulating on-disk corruption."""
    with _control_db(version_dir) as db:
        db.execute("UPDATE records SET time_span_start_us = ?, time_span_end_us = ?", [start_us, end_us])


def test_time_span_with_reversed_bounds_raises_format_error(tmp_path):
    # A stored time_span whose end is not past its start fails Span validation on decode; it must surface
    # as a format error, not the raw ValueError that validation raises.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_time_span(version_dir, 6_000_000, 5_000_000)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="must be > start"),
    ):
        list(reader.iter_records())


def test_point_shaped_time_span_raises_format_error(tmp_path):
    # A time_span must be an interval covering the whole record. A corrupt point-shaped one (no end) is
    # rejected as a format error, not left to reach the unscoped-span check and raise a bare TypeError.
    version_dir = _time_span_dataset(tmp_path)
    _corrupt_time_span(version_dir, 0, None)
    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="must be a TimeInterval"),
    ):
        list(reader.iter_records())


def test_task_payload_field_written_before_it_existed_reads_as_none(tmp_path):
    """A task written before an optional payload field existed reads back with that field None.

    A payload field is one row per field, so a version written before the field existed simply has
    no row for it. Deleting the row is what an older build looks like, and the reader must fall back
    to the dataclass default rather than raise, so additive task fields never force a recuration.
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
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    dataset.add_task(record, ClassificationTask(target="afib", target_schema="scp5", id="cls-0"))
    version_dir = _write(tmp_path, dataset=dataset)

    # Simulate an older build: drop the optional payload field's row from the control database.
    with _control_db(version_dir) as db:
        assert db.execute("SELECT count(*) FROM task_fields WHERE field = 'target_schema'").fetchone()[0] == 1
        db.execute("DELETE FROM task_fields WHERE field = 'target_schema'")

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    task = restored.tasks[0]
    assert isinstance(task, ClassificationTask)
    assert task.target == "afib"
    assert task.target_schema is None


def _annotation_dataset(annotations):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/annotations",
            dataset_version=Version(1, 0, 0),
            name="Annotations",
            description="One record carrying annotations under test.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    for ann in annotations:
        record.add_annotation(ann)
    return dataset


def test_annotation_map_value_round_trips(tmp_path):
    ds = _annotation_dataset([Annotation(key="panas", value={"pa": 30, "na": 12}, id="ann-map")])
    restored = _read(_write(tmp_path, dataset=ds))
    assert restored.records[0].annotations[0].value == {"pa": 30, "na": 12}


def test_annotation_source_round_trips(tmp_path):
    ds = _annotation_dataset([Annotation(key="stage", value="N2", source="rater-A", id="ann-src")])
    restored = _read(_write(tmp_path, dataset=ds))
    assert restored.records[0].annotations[0].source == "rater-A"


def _registered_source_dataset(source):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/registered-source",
            dataset_version=Version(1, 0, 0),
            name="RegisteredSource",
            description="A registered annotation carrying a source that no record carries.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    options = Annotation(key="answer_options", value=["yes", "no"], source=source, id="opts-src")
    dataset.register_annotations([options])
    dataset.add_task(
        record,
        AnswerTask(prompt="Rhythm?", target="yes", input_annotation_ids=(options.id,), id="qa-0"),
    )
    return dataset


def test_registered_annotation_source_round_trips(tmp_path):
    restored = _read(_write(tmp_path, dataset=_registered_source_dataset("rater-A")))
    assert restored.registered_annotations[0].source == "rater-A"


def test_temporal_localization_empty_target_round_trips(tmp_path):
    """A localization ``target=()`` ("searched, found none") round-trips as ``()``, not ``None``."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/localization-empty",
            dataset_version=Version(1, 0, 0),
            name="Localization",
            description="A localization task whose answer is that there are no events.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    dataset.add_task(record, TemporalLocalizationTask(prompt="Mark every P-wave", target=(), id="loc-0"))
    restored = _read(_write(tmp_path, dataset=dataset))
    task = restored.tasks[0]
    assert isinstance(task, TemporalLocalizationTask)
    assert task.target == ()


def test_correspondence_time_series_answer_round_trips(tmp_path):
    """A correspondence task that answers with series ids round-trips those ids, resolved on the record."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="test/correspondence",
            dataset_version=Version(1, 0, 0),
            name="Correspondence",
            description="Which signals correspond, answered with time-series ids.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
        )
    )
    series = TimeSeries(
        spec=TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt),
        signal="I",
        time_axis=RegularAxis.from_rate_hz(Fraction(500)),
        loader=lambda: pa.array([0.0, 1.0, 2.0], type=pa.float32()),
        source_id="rec-0",
        time_series_id="ecg-rec-0-I",
        n_values=3,
    )
    record = dataset.add_record(time_series=(series,), record_id="rec-0")
    dataset.add_task(record, TSCorrespondenceTask(target_time_series_ids=("ecg-rec-0-I",), id="corr-0"))
    restored = _read(_write(tmp_path, dataset=dataset))
    task = restored.tasks[0]
    assert isinstance(task, TSCorrespondenceTask)
    assert task.target_time_series_ids == ("ecg-rec-0-I",)
