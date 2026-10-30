from contextlib import contextmanager
from dataclasses import fields, replace
import shutil

import numpy as np
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset, composition
from timenet.dataset.composition import ImportedDataset, ImportState
from timenet.dataset.edit import edit_version, remove_records, remove_signals
from timenet.errors import TimeFNotFoundError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import PackedValueReader, StoredOffsets, StoredText
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import (
    TASKS,
    Access,
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    InputModality,
    License,
    LocalizationMode,
    ObjectKind,
    ScalarPredictionTask,
    Split,
    StepInterval,
    StepPoint,
    StepSpan,
    Task,
    TemporalLocalizationTask,
    TextHandle,
    TextListHandle,
    TimeInterval,
    TimeOrigin,
    TimePoint,
    TimeSeriesSpec,
    TimeSpan,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    Version,
    read_value,
    ureg,
)
from timenet.types.access import AccessRequirement, access_requirements, effective_access
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


_ALIAS = ParentAlias("base")
_SPEC = TimeSeriesSpec(spec_type="sine", name="Sine", unit_value=ureg.dimensionless)


def store(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _next_version(dataset, patch=1):
    """Give a dataset read back from storage a later patch version, so that it writes beside its release."""
    dataset._metadata = dataset.metadata.model_copy(update={"dataset_version": Version(1, 0, patch)})


def _metadata(dataset_id, **update):
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version(1, 0, 0),
        name=dataset_id,
        description="d",
        license=License.MIT,
    )
    return metadata.model_copy(update=update)


def _ref(reader):
    return DatasetRef(dataset_id=reader.metadata.dataset_id, version=reader.metadata.dataset_version)


def _child(*readers, dataset_id="test/child", aliases=(_ALIAS,)):
    """Return an empty child whose card declares each reader's release under its alias."""
    parents = {alias: _ref(reader) for alias, reader in zip(aliases, readers, strict=True)}
    return TimeFDataset(metadata=_metadata(dataset_id, parents=parents))


def _signal(name, *, offset_s=0):
    """Return 16 values at 16 Hz, covering one second from ``offset_s``."""
    return Signal.from_values(
        np.arange(16, dtype=np.float32),
        spec=_SPEC,
        name=name,
        time_axis=RegularAxis.from_rate_hz(16, offset_us=offset_s * 1_000_000),
    )


def _ids(dataset):
    """Return the ID the parent dataset gave each of its Records and Signals, by name.

    The Records are named ``rec`` and ``other``, or ``r`` alone, in the order they were added.
    """
    names = ("rec", "other") if len(dataset.records) == 2 else ("r",)
    records = {name: record.id for name, record in zip(names, dataset.records, strict=True)}
    return records | {signal.name: signal.id for record in dataset.records for signal in record.signals}


def _tree(dataset_id="test/tree", *, time_span=None, **update):
    """Return a parent whose Record ``rec`` nests Sources and has annotations at every level.

    ``sig-a`` covers 0 to 1 s below ``monitor/leads``, and ``sig-b`` covers 2 to 3 s directly in
    ``monitor``. The annotations named ``late`` lie at 2.5 s, so only ``sig-b`` places them. The
    Source ``aux`` holds no Signal. The second Record, ``other``, holds ``sig-c``.
    """
    dataset = TimeFDataset(metadata=_metadata(dataset_id, **update))
    signal_a, signal_b = _signal("sig-a"), _signal("sig-b", offset_s=2)
    aux = Source(name="aux")
    root = Source(
        name="monitor",
        metadata={"model": "m1"},
        signals=(signal_b,),
        sources=(Source(name="leads", signals=(signal_a,)), aux),
    )
    record = dataset.add_record(record=Record(metadata={"site": "north"}, sources=(root,), time_span=time_span))
    record.add_annotations(
        [
            Annotation(key="age", value=64),
            Annotation(key="early", span=TimePoint.seconds(0.5)),
            Annotation(key="late", span=TimePoint.seconds(2.5)),
        ]
    )
    for annotation in (
        Annotation(key="device", value="m1"),
        Annotation(key="src-early", span=TimePoint.seconds(0.5)),
        Annotation(key="src-late", span=TimePoint.seconds(2.5)),
    ):
        root.annotate(annotation)
    aux.annotate(Annotation(key="aux-note", value="spare"))
    signal_a.annotate(Annotation(key="artifact", span=TimeInterval.seconds(0.0, 0.25)))
    dataset.add_record(record=Record(sources=(Source(name="monitor", signals=(_signal("sig-c"),)),)))
    dataset.annotate(Annotation(key="release-note", value="first release"))
    return dataset


@contextmanager
def _reader(root, dataset):
    with TimeFReader(open_local_version(store(root, dataset))) as reader:
        yield reader


def _keys(annotated):
    return sorted(annotation.key for annotation in annotated.annotations)


def _all_keys(record):
    return [("record", _keys(record)), *sorted((source.name, _keys(source)) for source in record.walk_sources())]


def _indexes(dataset):
    names = ("_records_by_id", "_signals_by_id", "_signals_by_record_id", "_records_by_signal_id")
    return {name: dict(getattr(dataset, name)) for name in names} | {
        "import_map": dict(dataset._import_state.import_map),
        "_sources": set(dataset._sources),
        "_records": list(dataset._records),
        "last_ids": dataset.last_ids,
    }


def _rows(release_dir):
    with connect_control(release_dir / "control.duckdb", read_only=True) as connection:
        return connection.execute(
            "SELECT object_kind, parent_id, object_id FROM imported_objects ORDER BY object_kind, parent_id"
        ).fetchall()


def test_a_signal_import_copies_its_source_chain_and_the_annotations_its_window_places(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, signals=[ids["sig-a"]])
        (record,) = child.records
        record.metadata["site"] = "edited"
        assert reader.get(Record, ids["rec"]).metadata == {"site": "north"}
    (root,) = record.sources
    (leads,) = root.sources
    (signal,) = leads.signals
    assert imported.signals == {ids["sig-a"]: signal}
    assert imported.records == {}
    assert (root.name, leads.name, root.signals) == ("monitor", "leads", ())
    assert root.metadata == {"model": "m1"}
    # The annotations at 2.5 s lie only on sig-b, which was not imported.
    assert _keys(record) == ["age", "early"]
    assert _keys(root) == ["device", "src-early"]
    assert _keys(leads) == []
    assert _keys(signal) == ["artifact"]
    assert child.annotations == ()
    # The copies took the first IDs of the child's own counters.
    assert (record.id, signal.id) == (1, 1)
    assert child._import_state.import_map == {
        (_ALIAS, ObjectKind.RECORD, ids["rec"]): 1,
        (_ALIAS, ObjectKind.SIGNAL, ids["sig-a"]): 1,
    }


def test_copies_take_the_childs_own_ids_and_only_the_map_links_them(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        # The child's first Record is the copy of the parent's second, so the two numberings differ.
        other = child.import_from(reader, records=[ids["other"]]).records[ids["other"]]
        rec = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        parent_other = reader.get(Record, ids["other"])
        again = child.import_from(reader, records=[ids["other"], ids["rec"]])
    assert (other.id, rec.id) == (1, 2)
    assert other.id != ids["other"]
    assert rec.id != ids["rec"]
    # A copy is a new object, never the parent's: the parent's read Records keep their own IDs.
    assert parent_other.id == ids["other"]
    assert other is not parent_other
    assert child.records == (other, rec)
    assert again.records == {ids["other"]: other, ids["rec"]: rec}
    assert sorted(signal.name for signal in rec.signals) == ["sig-a", "sig-b"]
    assert [signal.name for signal in other.signals] == ["sig-c"]
    assert child._import_state.import_map == {
        (_ALIAS, ObjectKind.RECORD, ids["other"]): other.id,
        (_ALIAS, ObjectKind.RECORD, ids["rec"]): rec.id,
        (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]): other.signals[0].id,
        **{(_ALIAS, ObjectKind.SIGNAL, ids[signal.name]): signal.id for signal in rec.signals},
    }
    assert all(isinstance(value, int) for value in child._import_state.import_map.values())


def test_copies_share_an_origin_their_parent_records_share(tmp_path):
    parent = TimeFDataset(metadata=_metadata("test/clocks"))
    shared, alone = TimeOrigin(1_700_000_000_000_000), TimeOrigin(1_700_000_000_000_000)
    for name, origin in (("a", shared), ("b", shared), ("c", alone)):
        parent.add_record(record=Record(start_time=origin, sources=(Source(name="s", signals=(_signal(name),)),)))
    a, b, c = (record.id for record in parent.records)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        # Two calls on one reader: the copy of b joins the origin the copy of a already holds.
        first = child.import_from(reader, records=[a]).records[a]
        second, third = (child.import_from(reader, records=[b, c]).records[key] for key in (b, c))
        read_a = reader.get(Record, a)
    assert first.start_time is second.start_time is read_a.start_time
    # Equal timestamps do not make two origins one: c keeps its own.
    assert third.start_time is not first.start_time
    assert third.start_time.timestamp == first.start_time.timestamp
    child_dir = store(tmp_path, child)
    with connect_control(child_dir / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM clocks").fetchone() == (2,)
    with TimeFReader(open_local_version(child_dir)) as reader:
        copy_a, copy_b, copy_c = reader.read().records
    assert copy_a.start_time is copy_b.start_time
    assert copy_c.start_time is not copy_a.start_time


def _clock_count(release_dir):
    with connect_control(release_dir / "control.duckdb", read_only=True) as connection:
        return connection.execute("SELECT count(*) FROM clocks").fetchone()


def _clocks_parent(root, timestamp):
    """Store a parent whose Records ``a``, ``b``, and ``c`` share one clock, and ``d`` has its own.

    Both clocks have the timestamp ``timestamp``.

    Returns:
        The release directory, and the IDs of ``a``, ``b``, ``c``, and ``d``.
    """
    parent = TimeFDataset(metadata=_metadata("test/clocks"))
    shared, alone = TimeOrigin(timestamp), TimeOrigin(timestamp)
    for name, origin in (("a", shared), ("b", shared), ("c", shared), ("d", alone)):
        parent.add_record(record=Record(start_time=origin, sources=(Source(name="s", signals=(_signal(name),)),)))
    return store(root, parent), tuple(record.id for record in parent.records)


@pytest.mark.parametrize("timestamp", [1_700_000_000_000_000, None])
def test_copies_share_a_parent_clock_across_imports_and_reloads(tmp_path, timestamp):
    parent_dir, (a, b, c, d) = _clocks_parent(tmp_path, timestamp)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=[a]).records[a]
    # A second reader builds new origins. The copy of b still takes the origin of the copy of a.
    with TimeFReader(open_local_version(parent_dir)) as reader:
        second = child.import_from(reader, records=[b, d]).records
    assert second[b].start_time is first.start_time
    # Equal timestamps do not make two clocks one: d keeps its own.
    assert second[d].start_time is not first.start_time
    child_dir = store(tmp_path, child)
    assert _clock_count(child_dir) == (2,)
    # After a write and a read back, the copy of c takes the origin of the copies of a and b.
    with (
        TimeFReader(open_local_version(child_dir)) as stored,
        TimeFReader(open_local_version(parent_dir)) as reader,
    ):
        reread = stored.read()
        third = reread.import_from(reader, records=[c]).records[c]
        _next_version(reread)
        again_dir = store(tmp_path, reread)
    copies = {record.signals[0].name: record for record in reread.records}
    assert third.start_time is copies["a"].start_time is copies["b"].start_time
    assert copies["d"].start_time is not copies["a"].start_time
    assert _clock_count(again_dir) == (2,)


def test_a_record_import_extends_a_partial_copy_in_place(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
        (record,) = child.records
        (root,) = record.sources
        (leads,) = root.sources
        (signal_a,) = leads.signals
        record.metadata["site"] = "edited"
        root.metadata["model"] = "edited"
        imported = child.import_from(reader, records=[ids["rec"]])
        again = child.import_from(reader, signals=[ids["sig-b"]])
    assert imported.records == {ids["rec"]: record}
    assert (child.records, record.sources, leads.signals) == ((record,), (root,), (signal_a,))
    assert (record.metadata, root.metadata) == ({"site": "edited"}, {"model": "edited"})
    (signal_b,) = root.signals
    assert again.signals == {ids["sig-b"]: signal_b}
    # The Signal the copy gained took the child's next Signal ID, and the dataset indexed it, so
    # a task can scope a span to it.
    assert signal_b.id == 2
    scope = TimeInterval.seconds(2.0, 2.5, time_series=(signal_b,))
    child.add_task(task=ClassificationTask(inputs=(record,), targets=("late",), scope=scope))
    aux = next(source for source in root.sources if source.name == "aux")
    assert {source.name for source in root.sources} == {"leads", "aux"}
    assert _keys(record) == ["age", "early", "late"]
    assert _keys(root) == ["device", "src-early", "src-late"]
    assert _keys(aux) == ["aux-note"]
    assert _keys(signal_a) == ["artifact"]


def test_growing_a_partial_copy_adds_only_what_the_new_signal_places(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    parent.records[0].annotate(Annotation(key="report", value="A long clinical report. " * 300))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
    (record,) = child.records
    (root,) = record.sources
    # The child edits and clears the copied annotations.
    record.annotations = ()
    record.annotate(Annotation(key="age", value=65))
    root.annotations = ()

    def unexpected_read(*args):
        pytest.fail("growing a copy must not read stored text")

    with TimeFReader(open_local_version(parent_dir)) as reader, monkeypatch.context() as patch:
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        child.import_from(reader, signals=[ids["sig-b"]])
    assert [(annotation.key, annotation.value) for annotation in record.annotations] == [("age", 65), ("late", None)]
    assert _keys(root) == ["src-late"]


def test_a_task_import_gives_its_records_the_origin_of_a_held_copy_on_their_clock(tmp_path):
    parent = TimeFDataset(metadata=_metadata("test/clocks"))
    shared = TimeOrigin(None)
    a, b = (
        parent.add_record(record=Record(start_time=shared, sources=(Source(name="s", signals=(_signal(name),)),)))
        for name in ("a", "b")
    )
    task = parent.add_task(task=AnswerTask(inputs=(b,), targets=("x",)))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=[a.id]).records[a.id]
    with TimeFReader(open_local_version(parent_dir)) as reader:
        (copy,) = child.import_from(reader, tasks=[task.id]).tasks.values()
    assert copy.inputs[0].start_time is first.start_time


def test_a_partial_copy_takes_the_origin_of_a_held_copy_on_its_parent_clock(tmp_path):
    parent_dir, (a, b, _, _) = _clocks_parent(tmp_path, None)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=[a]).records[a]
    with TimeFReader(open_local_version(parent_dir)) as reader:
        (signal_b,) = reader.get(Record, b).signals
        child.import_from(reader, signals=[signal_b.id])
    (_, partial) = child.records
    assert partial.start_time is first.start_time


def test_an_eager_signal_import_loads_only_the_selected_signal(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    parent.records[0].annotate(Annotation(key="report", value="A long clinical report. " * 300))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        loads = _count_array_loads(monkeypatch)
        copy = child.import_from(reader, signals=[ids["sig-a"]], load=LoadPlan.ALL_EAGER).signals[ids["sig-a"]]
        # The parent Record holds sig-a and sig-b. Only sig-a loads.
        assert len(loads) == 1
    (record,) = child.records
    # The parent release is gone, so only values that the import loaded can still be read.
    shutil.rmtree(parent_dir)
    assert copy.to_numpy().tolist() == list(range(16))
    assert next(annotation for annotation in record.annotations if annotation.key == "report").value.read() == (
        "A long clinical report. " * 300
    )


def test_an_eager_import_loads_the_signals_it_adds_to_a_held_copy(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        held = child.import_from(reader, signals=[ids["sig-a"]]).signals[ids["sig-a"]]
        loads = _count_array_loads(monkeypatch)
        added = child.import_from(reader, signals=[ids["sig-b"]], load=LoadPlan.ALL_EAGER).signals[ids["sig-b"]]
        # Only sig-b loads. The call did not select sig-a, so its copy stays lazy.
        assert len(loads) == 1
    assert not held._values.is_loaded
    shutil.rmtree(parent_dir)
    assert added.to_numpy().tolist() == list(range(16))


def test_growing_a_copy_past_its_time_span_raises_and_leaves_the_copy(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
        (record,) = child.records
        # sig-a covers 0 to 1 s, and sig-b covers 2 to 3 s.
        record.time_span = TimeInterval.seconds(0.0, 1.0)
        keys, indexes = _all_keys(record), _indexes(child)
        with pytest.raises(TimeFValidationError, match="must contain every series' window, but 'sig-b'"):
            child.import_from(reader, signals=[ids["sig-b"]])
        assert (_all_keys(record), _indexes(child)) == (keys, indexes)
        assert [signal.name for signal in record.signals] == ["sig-a"]
        record.time_span = TimeInterval.seconds(0.0, 3.0)
        child.import_from(reader, signals=[ids["sig-b"]])
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_complete_copy_keeps_annotations_its_windows_do_not_place(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    # The parent keeps this annotation with a warning: 1.5 s lies between sig-a and sig-b.
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        partial = _child(reader)
        partial.import_from(reader, signals=[ids["sig-a"]])
        assert "gap" not in _keys(partial.records[0])
        # Each import completes the copy, but only the first attaches "gap".
        for _ in range(2):
            partial.import_from(reader, records=[ids["rec"]], signals=[ids["sig-b"]])
        whole = _child(reader)
        whole.import_from(reader, records=[ids["rec"]])
    assert _keys(partial.records[0]) == ["age", "early", "gap", "late"]
    assert "gap" in _keys(whole.records[0])


def _two_source_parent():
    """Return a parent whose Record ``r`` holds the Sources ``lead`` and ``dev``.

    ``lead`` holds ``sig-w1`` and ``sig-w2`` (4 to 5 s). ``dev`` holds ``sig-x`` (0 to 1 s) and
    ``sig-y`` (2 to 3 s), and it has an annotation in the window of each.
    """
    parent = TimeFDataset(metadata=_metadata("test/two"))
    lead = Source(name="lead", signals=(_signal("sig-w1", offset_s=4), _signal("sig-w2", offset_s=4)))
    dev = Source(name="dev", signals=(_signal("sig-x"), _signal("sig-y", offset_s=2)))
    parent.add_record(record=Record(sources=(lead, dev)))
    dev.annotate(Annotation(key="x-note", span=TimePoint.seconds(0.5)))
    dev.annotate(Annotation(key="y-note", span=TimePoint.seconds(2.5)))
    return parent


def _named(record, name):
    (source,) = (source for source in record.sources if source.name == name)
    return source


def _hierarchy_of(record):
    return [(source, source.signals, source.annotations) for source in record.walk_sources()]


def test_an_extension_that_fails_leaves_the_copy_and_the_indexes_as_they_were(tmp_path):
    parent = _two_source_parent()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-w1"], ids["sig-x"]])
        (record,) = child.records
        lead = _named(record, "lead")
        # The child adds a second root Source named lead by hand. Only the check that indexes the
        # copy again finds it.
        record.sources = (*record.sources, Source(name="lead"))
        hierarchy = [(record.sources, record.annotations), *_hierarchy_of(record)]
        indexes = _indexes(child)
        # The call adds sig-w2 to the copy of lead and sig-y to the copy of dev in place. Then the
        # check fails, and the copy goes back to what it was.
        with pytest.raises(TimeFValidationError, match="record 1 holds more than one Source named 'lead'"):
            child.import_from(reader, signals=[ids["sig-w2"], ids["sig-y"]])
    assert [signal.name for signal in lead.signals] == ["sig-w1"]
    assert [(record.sources, record.annotations), *_hierarchy_of(record)] == hierarchy
    assert _indexes(child) == indexes


def test_a_record_that_fails_keeps_the_parent_recorded(tmp_path, monkeypatch):
    parent = _tree(access=Access.RESTRICTED, access_url="https://r.example")
    ids = _ids(parent)

    def refuse(self, record, *, keep_ids):
        raise TimeFValidationError("refused for the test")

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFDataset, "_register_record", refuse)
        with pytest.raises(TimeFValidationError, match="refused for the test"):
            child.import_from(reader, records=[ids["rec"], ids["other"]])
    assert list(child.parent_cards) == [_ALIAS]
    assert set(child.lineage) == {"test/tree@1.0.0"}
    assert (child.metadata.access, child.metadata.access_url) == (Access.OPEN, None)
    assert child.records == ()
    assert child._import_state.import_map == {}


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_partial_copy_of_a_record_with_a_session_span_keeps_every_annotation_in_it(tmp_path):
    parent = _tree(time_span=TimeInterval.seconds(0.0, 3.0))
    ids = _ids(parent)
    parent.records[0].annotate(Annotation(key="after", span=TimePoint.seconds(5.0)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
        (record,) = child.records
        # The session span places "late", which only sig-b's window covers; "after" lies past it.
        assert _keys(record) == ["age", "early", "late"]
        child.import_from(reader, records=[ids["rec"]])
    assert _keys(record) == ["after", "age", "early", "late"]


def test_reimporting_a_whole_record_changes_nothing_and_keeps_child_edits(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=[ids["rec"]])
        record = first.records[ids["rec"]]
        # A connector may replace the copied annotations; a later import leaves them so.
        record.annotations = ()
        hierarchy = [(source, source.signals, source.annotations) for source in record.walk_sources()]
        second = child.import_from(reader, records=[ids["rec"]], signals=[ids["sig-a"]])
    assert second.records == {ids["rec"]: record}
    assert second.signals[ids["sig-a"]] is next(signal for signal in record.signals if signal.name == "sig-a")
    assert child.records == (record,)
    assert record.annotations == ()
    assert [(source, source.signals, source.annotations) for source in record.walk_sources()] == hierarchy
    assert len(hierarchy) == 3


def test_copies_reuse_the_parent_payloads_of_irregular_offsets_and_packed_text(tmp_path, monkeypatch):
    text = "A long clinical report. " * 300
    offsets = [0, 1_000, 5_000, 9_000]
    parent = TimeFDataset(metadata=_metadata("test/irregular"))
    signal = Signal.from_irregular(np.arange(4, dtype=np.float32), time_offsets_us=offsets, spec=_SPEC, name="env")
    record = parent.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))
    record.annotate(Annotation(key="report", value=text))
    parent_dir = store(tmp_path, parent)

    def unexpected_read(*args):
        pytest.fail("an import must not read or copy stored values")

    with TimeFReader(open_local_version(parent_dir)) as reader, monkeypatch.context() as patch:
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        patch.setattr(TimeFReader, "_read_array", unexpected_read)
        child = _child(reader)
        child.import_from(reader, records=[record.id])
        child_dir = store(tmp_path, child)
    assert sorted(path.name for path in child_dir.iterdir()) == ["control.duckdb", "manifest.json"]
    with TimeFReader(open_local_version(child_dir)) as reader:
        (copy,) = reader.read().records
        (report,) = copy.annotations
        assert isinstance(report.value, StoredText)
        assert isinstance(copy.signals[0]._time_offsets, StoredOffsets)
        assert report.value._location.source == "test/irregular@1.0.0"
        assert copy.signals[0].time_offsets_us().tolist() == offsets
        assert copy.signals[0].to_numpy().tolist() == [0, 1, 2, 3]
        assert report.value.read() == text


def test_an_eager_import_gives_copies_that_read_without_the_parent(tmp_path):
    text = "A long clinical report. " * 300
    offsets = [0, 1_000, 5_000, 9_000]
    parent = TimeFDataset(metadata=_metadata("test/irregular"))
    signal = Signal.from_irregular(np.arange(4, dtype=np.float32), time_offsets_us=offsets, spec=_SPEC, name="env")
    record = parent.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))
    record.annotate(Annotation(key="report", value=text))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        (copy,) = child.import_from(reader, records=[record.id], load=LoadPlan.ALL_EAGER).records.values()
    # The parent release is gone, so only values that the import loaded can still be read.
    shutil.rmtree(parent_dir)
    (report,) = copy.annotations
    assert isinstance(report.value, StoredText)
    assert isinstance(copy.signals[0]._time_offsets, StoredOffsets)
    assert copy.signals[0].to_numpy().tolist() == [0, 1, 2, 3]
    assert copy.signals[0].time_offsets_us().tolist() == offsets
    assert report.value.read() == text


def _count_array_loads(monkeypatch):
    """Record each Signal array that a reader loads from storage, from now on.

    Returns:
        The list that collects the location of each load.
    """
    loads = []
    read_array = TimeFReader._read_array

    def counting(self, location, spec):
        loads.append(location)
        return read_array(self, location, spec)

    monkeypatch.setattr(TimeFReader, "_read_array", counting)
    return loads


def test_a_repeated_eager_import_loads_only_the_values_of_new_copies(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        loads = _count_array_loads(monkeypatch)
        first = child.import_from(reader, records=[ids["rec"]], load=LoadPlan.ALL_EAGER).records[ids["rec"]]
        assert len(loads) == 2
        loads.clear()
        again = child.import_from(reader, records=[ids["rec"], ids["other"]], load=LoadPlan.ALL_EAGER).records
        # Only sig-c of the new copy of other loads. The held copy of rec holds its values already.
        assert len(loads) == 1
        loads.clear()
        child.import_from(reader, records=[ids["rec"], ids["other"]], load=LoadPlan.ALL_EAGER)
        assert loads == []
    assert again[ids["rec"]] is first
    assert all(signal._values.is_loaded for copy in again.values() for signal in copy.signals)


def test_an_eager_import_loads_the_values_of_a_lazy_copy_the_child_holds(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        loads = _count_array_loads(monkeypatch)
        lazy = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        child.import_from(reader, records=[ids["rec"]])
        assert loads == []
        eager = child.import_from(reader, records=[ids["rec"]], load=LoadPlan.ALL_EAGER).records[ids["rec"]]
        # The copy loads each of its two arrays once, through its own value handles.
        assert len(loads) == 2
    assert eager is lazy
    # The parent release is gone, so only values that the import loaded can still be read.
    shutil.rmtree(parent_dir)
    assert {signal.name: signal.to_numpy().tolist() for signal in eager.signals} == {
        "sig-a": list(range(16)),
        "sig-b": list(range(16)),
    }


def test_a_written_child_reads_its_copies_by_parent_id(tmp_path, monkeypatch):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        ref = _ref(reader)
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"]], signals=[ids["sig-c"]])
        imported.records[ids["rec"]].metadata["site"] = "edited"
        child_dir = store(tmp_path, child)
    # Sources and annotations are reached through their owners, so only Records and Signals get
    # rows, which map the parent's IDs to the child's.
    expected = sorted(
        (kind.value, parent_id, child_id) for (_, kind, parent_id), child_id in child._import_state.import_map.items()
    )
    assert _rows(child_dir) == expected
    assert [(kind, parent_id) for kind, parent_id, _ in expected] == sorted(
        [("Record", ids["rec"]), ("Record", ids["other"])]
        + [("Signal", ids[name]) for name in ("sig-a", "sig-b", "sig-c")]
    )
    with TimeFReader(open_local_version(child_dir)) as stored:

        def unexpected_read(**_):
            raise AssertionError("a get by ID must not read the whole dataset")

        monkeypatch.setattr(stored, "read", unexpected_read)
        assert stored.get(Record, ids["rec"], parent=_ALIAS).metadata["site"] == "edited"
        assert stored.get(Signal, ids["sig-c"], parent=ref).id == imported.signals[ids["sig-c"]].id
        signals = stored.get_many(Signal, [ids["sig-a"], ids["sig-b"]], parent=_ALIAS)
        assert {signal.name for signal in signals} == {"sig-a", "sig-b"}


def test_an_import_reads_only_the_selected_records(tmp_path, monkeypatch):
    batches = []
    read_records = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        requested = None if record_ids is None else tuple(record_ids)
        batches.append(requested)
        return read_records(self, requested, **kwargs)

    def unexpected_read(*args, **kwargs):
        pytest.fail("an import must not read the whole parent")

    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "read", unexpected_read)
        monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
        child.import_from(reader, records=[ids["rec"]], signals=[ids["sig-c"]])
    assert batches == [(ids["rec"], ids["other"])]


def test_an_import_that_selects_nothing_still_records_its_parent(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        assert child.import_from(reader) == ImportedDataset(alias=_ALIAS, parent=_ref(reader))
        child_dir = store(tmp_path, child)
    with connect_control(child_dir / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT alias, parent_ref FROM imports").fetchall() == [("base", "test/tree@1.0.0")]
    assert _rows(child_dir) == []
    assert list(open_local_version(child_dir).manifest.parent_cards) == [_ALIAS]


def test_an_import_into_a_child_read_back_rederives_its_schema(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    other = TimeFDataset(metadata=_metadata("test/other"))
    cosine = TimeSeriesSpec(spec_type="cosine", name="Cosine", unit_value=ureg.dimensionless)
    signal = Signal.from_values(
        np.zeros(4, dtype=np.float32), spec=cosine, name="c", time_axis=RegularAxis.from_rate_hz(4)
    )
    other_record = other.add_record(record=Record(sources=(Source(name="monitor", signals=(signal,)),)))
    with _reader(tmp_path, parent) as tree, _reader(tmp_path, other) as second:
        aliases = (_ALIAS, ParentAlias("other"))
        child = _child(tree, second, aliases=aliases)
        child.import_from(tree, records=[ids["rec"]])
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            assert reread.schema is not None
            assert [spec.spec_type for spec in reread.schema.time_series_specs] == ["sine"]
            reread._metadata = reread.metadata.model_copy(update={"dataset_version": Version(1, 0, 1)})
            reread.import_from(second, records=[other_record.id])
            assert reread.schema is None
            # The copy continues the counters that the stored version kept.
            assert [record.id for record in reread.records] == [1, 2]
            reread_dir = reread.write(path=tmp_path)
    specs = open_local_version(reread_dir).manifest.dataset_schema.time_series_specs
    assert sorted(spec.spec_type for spec in specs) == ["cosine", "sine"]


def test_a_child_read_back_keeps_its_import_state_and_reuses_its_copies(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    selected = [ids["rec"], ids["other"]]
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=selected)
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            # Written again with no import in between, the child keeps its import rows and parent cards.
            _next_version(reread)
            again_dir = store(tmp_path, reread)
            records = reread.records
            again = reread.import_from(reader, records=selected)
    assert _rows(again_dir) == _rows(child_dir)
    manifest = open_local_version(again_dir).manifest
    assert manifest.parent_cards == child.parent_cards
    assert {str(card.source): card for card in manifest.lineage} == child.lineage
    # The import finds the copies that the read built through the restored map, and copies nothing.
    assert len(reread.records) == 2
    assert all(copy is record for copy, record in zip(reread.records, records, strict=True))
    assert all(any(copy is record for record in records) for copy in again.records.values())
    assert {parent_id: copy.id for parent_id, copy in again.records.items()} == {
        parent_id: child_id
        for (_, kind, parent_id), child_id in child._import_state.import_map.items()
        if kind is ObjectKind.RECORD
    }
    assert reread._import_state.import_map == child._import_state.import_map


def test_a_child_read_back_and_written_again_reads_every_copy_by_parent_id(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=[ids["rec"]])
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            assert (reread._import_state.import_map, reread.parent_cards, reread.lineage) == (
                child._import_state.import_map,
                child.parent_cards,
                child.lineage,
            )
            second = reread.import_from(reader, records=[ids["other"]])
            _next_version(reread)
            again_dir = store(tmp_path, reread)
    expected = sorted(
        (kind.value, parent_id, child_id) for (_, kind, parent_id), child_id in reread._import_state.import_map.items()
    )
    assert _rows(again_dir) == expected
    assert [(kind, parent_id) for kind, parent_id, _ in expected] == sorted(
        [("Record", ids["rec"]), ("Record", ids["other"])]
        + [("Signal", ids[name]) for name in ("sig-a", "sig-b", "sig-c")]
    )
    with TimeFReader(open_local_version(again_dir)) as stored:
        assert stored.get(Record, ids["rec"], parent=_ALIAS).id == first.records[ids["rec"]].id
        assert stored.get(Record, ids["other"], parent=_ALIAS).id == second.records[ids["other"]].id
        assert {signal.name for signal in stored.get_many(Signal, [ids["sig-a"], ids["sig-c"]], parent=_ALIAS)} == {
            "sig-a",
            "sig-c",
        }


def test_an_import_into_a_child_read_back_leaves_its_reader_as_it_is(tmp_path):
    parent, second_parent = _tree(), _tree("test/second")
    ids, second_ids = _ids(parent), _ids(second_parent)
    with _reader(tmp_path, parent) as reader, _reader(tmp_path, second_parent) as second:
        child = _child(reader, second, aliases=(_ALIAS, ParentAlias("second")))
        child.import_from(reader, records=[ids["rec"]])
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            reread.import_from(second, records=[second_ids["other"]])
            assert list(reread.parent_cards) == [_ALIAS, "second"]
            # The reader's own parent cards are not the dataset's, so they stay as stored.
            assert list(stored.read().parent_cards) == [_ALIAS]
            with pytest.raises(TimeFNotFoundError, match="declares no parent alias 'second'"):
                stored.get(Record, second_ids["other"], parent=ParentAlias("second"))


def test_an_unknown_parent_id_raises_before_the_child_changes(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        with pytest.raises(TimeFNotFoundError, match="999"):
            child.import_from(reader, records=[ids["rec"], 999])
        with pytest.raises(TimeFNotFoundError, match="999"):
            child.import_from(reader, signals=[ids["sig-a"], 999])
        with pytest.raises(TimeFNotFoundError, match="999"):
            child.import_from(reader, records=[ids["rec"]], tasks=[999])
    assert (child.records, child.parent_cards, child.lineage, child._import_state.import_map) == ((), {}, {}, {})


@pytest.mark.parametrize("selection", ["records", "signals", "tasks"])
def test_a_selection_must_be_a_collection_of_int_parent_ids(tmp_path, selection):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        with pytest.raises(TimeFValidationError, match="not the string 'rec'"):
            child.import_from(reader, **{selection: "rec"})
        # A string is no ID, even one that spells a parent's ID; nor is a bool or zero.
        for identity in (str(ids["rec"]), True, 0):
            with pytest.raises(TimeFValidationError, match="must be a positive int64 id"):
                child.import_from(reader, **{selection: [identity]})
    assert child.records == ()


@pytest.mark.parametrize("alias", [None, _ALIAS])
def test_a_parent_the_card_does_not_declare_is_rejected(tmp_path, alias):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = TimeFDataset(metadata=_metadata("test/child"))
        with pytest.raises(TimeFValidationError, match=r"declare it in metadata\.parents"):
            child.import_from(reader, alias=alias, records=[ids["rec"]])
    assert (child.records, child.parent_cards) == ((), {})


def test_an_alias_must_declare_the_parent(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        with pytest.raises(
            TimeFValidationError, match=r"parent alias 'other' of test/child does not declare .*\['base'\]"
        ):
            child.import_from(reader, alias=ParentAlias("other"), records=[ids["rec"]])
    assert (child.records, child.parent_cards) == ((), {})


def test_a_parent_declared_under_two_aliases_is_imported_under_one(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader, reader, aliases=(ParentAlias("a"), ParentAlias("b")))
        with pytest.raises(TimeFValidationError, match=r"under the parent aliases \['a', 'b'\]; pass alias="):
            child.import_from(reader, records=[ids["rec"]])
        child.import_from(reader, alias=ParentAlias("a"), records=[ids["rec"]])
        with pytest.raises(TimeFValidationError, match=r"already imports test/tree@1\.0\.0 under the parent alias 'a'"):
            child.import_from(reader, alias=ParentAlias("b"), records=[ids["other"]])
        # The check reads the parent cards, not the import map: a dataset with the cards and no map
        # is refused too.
        carded = _child(reader, reader, aliases=(ParentAlias("a"), ParentAlias("b")))
        carded._import_state = ImportState(parent_cards=dict(child.parent_cards))
        with pytest.raises(TimeFValidationError, match="under the parent alias 'a'"):
            carded.import_from(reader, alias=ParentAlias("b"), records=[ids["other"]])
    assert len(child.records) == 1
    assert carded.records == ()


def test_an_alias_that_imports_another_release_is_rejected(tmp_path):
    with _reader(tmp_path, _tree()) as first:
        card = ParentDatasetCard(metadata=first.metadata)
    newer = _tree(dataset_version=Version(1, 0, 1))
    ids = _ids(newer)
    with _reader(tmp_path, newer) as second:
        # The card declares the newer release, while the parent cards it kept name the older one.
        child = _child(second)
        child._import_state.parent_cards[_ALIAS] = card
        with pytest.raises(TimeFValidationError, match=r"already imports test/tree@1\.0\.0, not test/tree@1\.0\.1"):
            child.import_from(second, records=[ids["rec"]])
    assert child.records == ()


def _effective(dataset):
    return effective_access(dataset.metadata, dataset.lineage.values())


def _import_all(tmp_path, parents, **update):
    """Import each parent, in turn, into a new child whose card takes ``update``.

    Returns:
        The child, and after each import the access and the access page of its card and its
        effective access level.
    """
    readers = {alias: TimeFReader(open_local_version(store(tmp_path, parent))) for alias, parent in parents.items()}
    try:
        child = _child(*readers.values(), aliases=tuple(readers))
        child._metadata = child.metadata.model_copy(update=update)
        steps = []
        for reader in readers.values():
            child.import_from(reader)
            steps.append((child.metadata.access, child.metadata.access_url, _effective(child)))
    finally:
        for reader in readers.values():
            reader.close()
    return child, steps


def test_an_import_keeps_the_card_of_the_child_and_its_effective_access_is_the_strictest(tmp_path):
    parents = {
        ParentAlias("credentialed"): _tree("test/c", access=Access.CREDENTIALED, access_url="https://c.example"),
        ParentAlias("restricted"): _tree("test/r", access=Access.RESTRICTED, access_url="https://r.example"),
        ParentAlias("open"): _tree("test/o"),
    }

    _, steps = _import_all(tmp_path, parents)

    # The card of the child stays open. The strictest level of its lineage applies to its data.
    assert steps == [
        (Access.OPEN, None, Access.CREDENTIALED),
        (Access.OPEN, None, Access.RESTRICTED),
        (Access.OPEN, None, Access.RESTRICTED),
    ]


def _requirement(release, access, access_url):
    return AccessRequirement(DatasetRef.parse(release), access, access_url)


def test_a_child_with_its_own_access_lists_itself_and_its_stricter_parent(tmp_path):
    parents = {ParentAlias("a"): _tree("test/a", access=Access.RESTRICTED, access_url="https://a.example")}

    child, steps = _import_all(tmp_path, parents, access=Access.CREDENTIALED, access_url="https://child.example")

    assert steps == [(Access.CREDENTIALED, "https://child.example", Access.RESTRICTED)]
    assert access_requirements(child.metadata, child.lineage.values()) == (
        _requirement("test/child@1.0.0", Access.CREDENTIALED, "https://child.example"),
        _requirement("test/a@1.0.0", Access.RESTRICTED, "https://a.example"),
    )


def test_two_parents_with_different_access_pages_are_both_listed(tmp_path):
    parents = {
        ParentAlias("a"): _tree("test/a", access=Access.CREDENTIALED, access_url="https://a.example"),
        ParentAlias("b"): _tree("test/b", access=Access.CREDENTIALED, access_url="https://b.example"),
    }

    child, steps = _import_all(tmp_path, parents)

    assert steps[-1] == (Access.OPEN, None, Access.CREDENTIALED)
    assert access_requirements(child.metadata, child.lineage.values()) == (
        _requirement("test/a@1.0.0", Access.CREDENTIALED, "https://a.example"),
        _requirement("test/b@1.0.0", Access.CREDENTIALED, "https://b.example"),
    )


def test_a_child_of_a_restricted_parent_reads_back_with_its_own_card(tmp_path):
    parent = _tree(access=Access.RESTRICTED, access_url="https://r.example")
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=[ids["rec"]])
        child_dir = store(tmp_path, child)
    manifest = open_local_version(child_dir).manifest
    assert (manifest.metadata.access, manifest.metadata.access_url) == (Access.OPEN, None)
    with TimeFReader(open_local_version(child_dir)) as reader:
        reread = reader.read()
    assert reread.metadata == manifest.metadata
    assert _effective(reread) is Access.RESTRICTED
    assert access_requirements(reread.metadata, reread.lineage.values()) == (
        _requirement("test/tree@1.0.0", Access.RESTRICTED, "https://r.example"),
    )


def test_a_grandparent_access_page_reaches_the_grandchild_through_its_lineage(tmp_path):
    grandparent = _tree("test/grand", access=Access.CREDENTIALED, access_url="https://grand.example")
    ids = _ids(grandparent)
    with _reader(tmp_path, grandparent) as reader:
        parent = _child(reader, dataset_id="test/parent")
        parent.import_from(reader, records=[ids["rec"]])
        parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        grandchild = _child(reader, dataset_id="test/grandchild", aliases=(ParentAlias("parent"),))
        grandchild.import_from(reader, records=[reader.get(Record, ids["rec"], parent=_ALIAS).id])
        grandchild_dir = store(tmp_path, grandchild)
    manifest = open_local_version(grandchild_dir).manifest
    assert (manifest.metadata.access, manifest.metadata.access_url) == (Access.OPEN, None)
    # The cards of the parent and the grandchild stay open, so only the grandparent is listed. Its
    # level reaches the grandchild through the lineage.
    assert effective_access(manifest.metadata, manifest.lineage) is Access.CREDENTIALED
    assert access_requirements(manifest.metadata, manifest.lineage) == (
        _requirement("test/grand@1.0.0", Access.CREDENTIALED, "https://grand.example"),
    )


def test_a_grandchild_records_every_ancestor_in_its_lineage(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=[ids["rec"]])
        child_dir = store(tmp_path, child)
    with TimeFReader(open_local_version(child_dir)) as reader:
        grandchild = _child(reader, dataset_id="test/grandchild", aliases=(ParentAlias("child"),))
        grandchild.import_from(reader, records=[reader.get(Record, ids["rec"], parent=_ALIAS).id])
        grandchild_dir = store(tmp_path, grandchild)
    assert set(grandchild.lineage) == {"test/tree@1.0.0", "test/child@1.0.0"}
    manifest = open_local_version(grandchild_dir).manifest
    assert list(manifest.parent_cards) == ["child"]
    assert sorted(str(card.source) for card in manifest.lineage) == ["test/child@1.0.0", "test/tree@1.0.0"]
    with TimeFReader(open_local_version(grandchild_dir)) as reader:
        reread = reader.read()
        (record,) = reread.records
        assert _keys(record) == ["age", "early", "late"]
        assert [signal.to_numpy().tolist() for signal in record.signals] == [list(range(16))] * 2
        # Read back, the grandchild keeps its parent cards and lineage, and writes them again.
        assert (reread.parent_cards, reread.lineage) == (grandchild.parent_cards, grandchild.lineage)
        _next_version(reread)
        again_dir = store(tmp_path, reread)
    assert open_local_version(again_dir).manifest.lineage == manifest.lineage


def test_the_writer_rejects_copies_imported_under_an_alias_without_a_parent_card(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=[ids["rec"]])
        del child._import_state.parent_cards[_ALIAS]
        with pytest.raises(TimeFValidationError, match=r"parent aliases \['base'\] without a parent card"):
            store(tmp_path, child)


def test_an_import_copies_again_what_the_child_no_longer_holds(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    selected = [ids["rec"], ids["other"]]
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=selected)
        edited = remove_records(child, [imported.records[ids["rec"]].id])
        (kept,) = edited.records
        again = edited.import_from(reader, records=selected)
        edited_dir = store(tmp_path, edited)
    assert again.records[ids["other"]] is kept
    assert edited.records == (kept, again.records[ids["rec"]])
    assert again.records[ids["rec"]] is not imported.records[ids["rec"]]
    # The new copy takes a new ID; the edit kept the counters, so the removed copy's ID is not reused.
    assert again.records[ids["rec"]].id > max(imported.records[identity].id for identity in selected)
    assert {signal.name for signal in again.records[ids["rec"]].signals} == {"sig-a", "sig-b"}
    with TimeFReader(open_local_version(edited_dir)) as stored:
        assert stored.get(Record, ids["rec"], parent=_ALIAS).metadata == {"site": "north"}


def test_an_import_state_copy_without_removed_copies_leaves_the_state_as_it_is():
    card = ParentDatasetCard(metadata=_metadata("test/tree"))
    state = ImportState(
        parent_cards={_ALIAS: card},
        lineage={"test/tree@1.0.0": card},
        import_map={
            (_ALIAS, ObjectKind.RECORD, 10): 1,
            (_ALIAS, ObjectKind.SIGNAL, 20): 1,
            (_ALIAS, ObjectKind.SIGNAL, 21): 2,
            (_ALIAS, ObjectKind.TASK, 30): 2,
        },
    )
    trimmed = state.without({ObjectKind.SIGNAL: {1}, ObjectKind.TASK: {5}})
    # Child IDs are per kind: Signal 1 goes, and Record 1 and task 2 stay.
    assert trimmed.import_map == {
        (_ALIAS, ObjectKind.RECORD, 10): 1,
        (_ALIAS, ObjectKind.SIGNAL, 21): 2,
        (_ALIAS, ObjectKind.TASK, 30): 2,
    }
    assert (trimmed.parent_cards, trimmed.lineage) == (state.parent_cards, state.lineage)
    # The copy holds its own dicts, so a change to it leaves the state as it is.
    for part in (trimmed.parent_cards, trimmed.lineage, trimmed.import_map):
        part.clear()
    assert (len(state.parent_cards), len(state.lineage), len(state.import_map)) == (1, 1, 4)


def test_a_later_import_finds_the_copy_of_a_source_by_its_name(tmp_path):
    parent = _two_source_parent()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["r"]]).records[ids["r"]]
        trimmed = remove_signals(child, [signal.id for signal in _named(record, "dev").signals])
        (copy,) = trimmed.records
        dev = _named(copy, "dev")
        assert (dev.signals, dev.annotations) == ((), ())
        # No imported Signal identifies the copy of dev now. Sibling Sources have distinct names, so
        # the name of dev finds its copy.
        trimmed.import_from(reader, signals=[ids["sig-y"]])
    assert sorted(source.name for source in copy.sources) == ["dev", "lead"]
    assert ([signal.name for signal in dev.signals], _keys(dev)) == (["sig-y"], ["y-note"])


def test_an_edit_drops_the_import_entries_of_what_it_removes(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"], ids["other"]])
        edited = remove_records(child, [imported.records[ids["rec"]].id])
        edited_dir = store(tmp_path, edited)
    kept = imported.records[ids["other"]]
    # The map links only the copies the edit holds, and the writer stores those.
    assert edited._import_state.import_map == {
        (_ALIAS, ObjectKind.RECORD, ids["other"]): kept.id,
        (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]): kept.signals[0].id,
    }
    assert _rows(edited_dir) == [("Record", ids["other"], kept.id), ("Signal", ids["sig-c"], kept.signals[0].id)]
    manifest = open_local_version(edited_dir).manifest
    assert manifest.parent_cards == child.parent_cards
    assert {str(card.source): card for card in manifest.lineage} == child.lineage
    with TimeFReader(open_local_version(edited_dir)) as stored:
        assert stored.read()._import_state.import_map == edited._import_state.import_map
        with pytest.raises(TimeFNotFoundError):
            stored.get(Record, ids["rec"], parent=_ALIAS)


def test_the_writer_ignores_stale_entries_of_an_alias_the_child_dropped(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"]])
        edited = remove_records(child, [imported.records[ids["rec"]].id])
        # By hand: the map still names the removed copies, and the parent's card is gone.
        edited._import_state = ImportState(import_map=dict(child._import_state.import_map))
        edited_dir = store(tmp_path, edited)
    assert _rows(edited_dir) == []


def _partial_rec(reader, ids):
    """Return a child holding a copy of ``rec`` with ``sig-a`` only, whose metadata and annotations it edited."""
    child = _child(reader)
    child.import_from(reader, signals=[ids["sig-a"]])
    (partial,) = child.records
    partial.metadata["site"] = "edited"
    partial.annotations = tuple(annotation for annotation in partial.annotations if annotation.key != "age")
    return child


def test_a_child_read_back_extends_its_partial_copy_and_keeps_its_edits(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        in_memory = _partial_rec(reader, ids)
        in_memory.import_from(reader, records=[ids["rec"]])
        child_dir = store(tmp_path, _partial_rec(reader, ids))
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            (record,) = reread.records
            whole = reread.import_from(reader, records=[ids["rec"]])
            _next_version(reread)
            again_dir = store(tmp_path, reread)
    assert whole.records == {ids["rec"]: record}
    assert reread.records == (record,)
    assert record.metadata == {"site": "edited"}
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]
    # The copy of sig-b takes the next Signal ID of the counters the stored version kept.
    assert sorted(signal.id for signal in record.signals) == [1, 2]
    # The extension attaches the annotations that sig-b places, as it does in memory, and age,
    # which the child removed, stays removed.
    assert _all_keys(record) == _all_keys(in_memory.records[0])
    assert _keys(record) == ["early", "late"]
    with TimeFReader(open_local_version(again_dir)) as stored:
        copy = stored.get(Record, ids["rec"], parent=_ALIAS)
        assert (copy.id, copy.metadata) == (record.id, {"site": "edited"})
        assert sorted(signal.name for signal in copy.signals) == ["sig-a", "sig-b"]


def test_an_edited_version_keeps_the_parent_cards_and_the_import_map(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"], ids["other"]])
        child_dir = store(tmp_path, child)
        edited_dir = edit_version(
            child_dir, dataset_version=Version(1, 0, 1), remove_record_ids=[imported.records[ids["other"]].id]
        )
        assert open_local_version(edited_dir).manifest.parent_cards == child.parent_cards
        assert [(kind, parent_id) for kind, parent_id, _ in _rows(edited_dir)] == sorted(
            [("Record", ids["rec"]), ("Signal", ids["sig-a"]), ("Signal", ids["sig-b"])]
        )
        with TimeFReader(open_local_version(edited_dir)) as stored:
            assert stored.get(Record, ids["rec"], parent=_ALIAS).id == imported.records[ids["rec"]].id
            with pytest.raises(
                TimeFNotFoundError, match=rf"no Record {ids['other']} imported under parent alias 'base'"
            ):
                stored.get(Record, ids["other"], parent=_ALIAS)
            # Read back, the edited version extends the copy it holds and copies the removed one again.
            edited = stored.read()
            (kept,) = edited.records
            again = edited.import_from(reader, records=[ids["rec"], ids["other"]])
    assert {str(card.source): card for card in open_local_version(edited_dir).manifest.lineage} == child.lineage
    assert again.records[ids["rec"]] is kept
    assert edited.records == (kept, again.records[ids["other"]])
    # The new copy takes a new ID: the stored counters keep the removed copy's from being reused.
    assert again.records[ids["other"]].id > imported.records[ids["other"]].id


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_removed_signal_comes_back_with_its_annotations_once(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    # No window places gap, so a copy holds it while it holds every Signal.
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        whole = _child(reader, dataset_id="test/whole")
        whole.import_from(reader, records=[ids["rec"]])
        child = _child(reader)
        sig_a = child.import_from(reader, signals=[ids["sig-a"], ids["sig-b"]]).signals[ids["sig-a"]].id
        # In memory. Removing sig-a drops early and src-early, which only it places, gap, which only
        # the complete copy places, and its entry.
        trimmed = remove_signals(child, [sig_a])
        assert _all_keys(trimmed.records[0]) == [
            ("record", ["age", "late"]),
            ("leads", []),
            ("monitor", ["device", "src-late"]),
        ]
        assert (_ALIAS, ObjectKind.SIGNAL, ids["sig-a"]) not in trimmed._import_state.import_map
        trimmed.import_from(reader, records=[ids["rec"]])
        # Stored. edit_version removes it from a written child, and an import copies it back after
        # a read-back.
        child_dir = store(tmp_path, child)
        edited_dir = edit_version(child_dir, dataset_version=Version(1, 0, 1), remove_signal_ids=[sig_a])
        assert [(kind, parent_id) for kind, parent_id, _ in _rows(edited_dir)] == [
            ("Record", ids["rec"]),
            ("Signal", ids["sig-b"]),
        ]
        with TimeFReader(open_local_version(edited_dir)) as stored:
            reread = stored.read()
            reread.import_from(reader, records=[ids["rec"]])
            _next_version(reread, patch=2)
            again_dir = store(tmp_path, reread)
        with TimeFReader(open_local_version(again_dir)) as stored:
            (written,) = stored.read().records
    # Each time, the copy gets back sig-a and each annotation once, as one import of the whole
    # Record gives.
    for copy in (trimmed.records[0], reread.records[0], written):
        assert _all_keys(copy) == _all_keys(whole.records[0])
        assert sorted((signal.name, _keys(signal)) for signal in copy.signals) == [
            ("sig-a", ["artifact"]),
            ("sig-b", []),
        ]
    # Copied back, sig-a takes a new Signal ID.
    assert trimmed._import_state.import_map[_ALIAS, ObjectKind.SIGNAL, ids["sig-a"]] > sig_a
    assert child._import_state.import_map[_ALIAS, ObjectKind.SIGNAL, ids["sig-a"]] == sig_a


def test_a_new_copy_of_a_removed_record_grows_as_a_first_copy_does(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        fresh = _child(reader, dataset_id="test/fresh")
        fresh.import_from(reader, signals=[ids["sig-b"]])
        fresh.import_from(reader, records=[ids["rec"]])
        child = _child(reader)
        removed = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]].id
        child_dir = store(tmp_path, child)
        # In memory. The edit drops the entries of the removed copy and its Signals.
        edited = remove_records(child, [removed])
        assert edited._import_state.import_map == {}
        edited.import_from(reader, signals=[ids["sig-b"]])
        edited.import_from(reader, records=[ids["rec"]])
        # The same through edit_version, with a write and a read-back between the two imports.
        edited_dir = edit_version(child_dir, dataset_version=Version(1, 0, 1), remove_record_ids=[removed])
        with TimeFReader(open_local_version(edited_dir)) as stored:
            partial = stored.read()
            partial.import_from(reader, signals=[ids["sig-b"]])
            _next_version(partial, patch=2)
            partial_dir = store(tmp_path, partial)
        with TimeFReader(open_local_version(partial_dir)) as stored:
            reread = stored.read()
            reread.import_from(reader, records=[ids["rec"]])
    # Each copy gets early and src-early, which only sig-a places, when the whole Record comes.
    assert _keys(fresh.records[0]) == ["age", "early", "late"]
    assert _all_keys(edited.records[0]) == _all_keys(fresh.records[0])
    assert _all_keys(reread.records[0]) == _all_keys(fresh.records[0])


def _containers(dataset):
    """Return what each Record and Source of a dataset holds, as it holds it now."""
    return [(record.sources, record.annotations) for record in dataset.records] + [
        (source.sources, source.signals, source.annotations)
        for record in dataset.records
        for source in record.walk_sources()
    ]


def _state(dataset):
    return _containers(dataset), _indexes(dataset), dict(dataset.parent_cards), dict(dataset.lineage)


def test_an_import_into_an_edited_dataset_leaves_the_original_as_it_is(tmp_path):
    parent, ids = _tasks_tree()
    second_parent = _tree("test/second")
    second_ids = _ids(second_parent)
    with _reader(tmp_path, parent) as reader, _reader(tmp_path, second_parent) as second:
        child = _child(reader, second, aliases=(_ALIAS, ParentAlias("second")))
        imported = child.import_from(reader, signals=[ids["sig-a"], ids["sig-b"]], records=[ids["other"]])
        other = imported.records[ids["other"]]
        # Removing sig-a leaves the nested Source leads empty.
        trimmed = remove_signals(child, [imported.signals[ids["sig-a"]].id])
        before = {id(dataset): _state(dataset) for dataset in (child, trimmed)}
        edited = remove_records(trimmed, [other.id])
        # The import extends the copy of rec in place, copying sig-a back into leads and adding
        # aux, and copies other and the tasks again. Another parent adds a card and an ancestor.
        again = edited.import_from(reader, records=[ids["rec"], ids["other"]], tasks=[ids["question"]])
        edited.import_from(second, records=[second_ids["other"]])
    assert again.records[ids["rec"]] is edited.records[0]
    assert sorted(signal.name for signal in edited.records[0].signals) == ["sig-a", "sig-b"]
    assert (len(edited.records), len(edited.tasks), list(edited.parent_cards)) == (3, 2, [_ALIAS, "second"])
    for dataset in (child, trimmed):
        assert _state(dataset) == before[id(dataset)]
        assert (dataset.tasks, dataset.annotations, list(dataset.parent_cards)) == ((), (), [_ALIAS])
    (partial,) = (record for record in trimmed.records if record.id != other.id)
    assert [signal.name for signal in partial.signals] == ["sig-b"]


def test_an_import_into_a_dataset_that_remove_signals_built_leaves_the_original_as_it_is(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        other = child.import_from(reader, signals=[ids["sig-a"]], records=[ids["other"]]).records[ids["other"]]
        (partial,) = (record for record in child.records if record is not other)
        before = _state(child)
        # The edit leaves the partial copy of rec as it is, and still copies it, so the import that
        # extends it extends the edit's copy.
        trimmed = remove_signals(child, [other.signals[0].id])
        trimmed.import_from(reader, records=[ids["rec"]])
    assert _state(child) == before
    assert [signal.name for signal in partial.signals] == ["sig-a"]
    (extended,) = (record for record in trimmed.records if record.id == partial.id)
    assert sorted(signal.name for signal in extended.signals) == ["sig-a", "sig-b"]


@pytest.mark.parametrize("edit", [remove_records, remove_signals])
def test_a_task_import_into_an_edited_dataset_leaves_the_signals_of_the_original_as_they_are(tmp_path, edit):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"], ids["other"]])
        record, other = imported.records[ids["rec"]], imported.records[ids["other"]]
        signal_a = _signal_named(record, "sig-a")
        # question references artifact on sig-a. The child's copy of sig-a no longer holds it, so
        # the import attaches a copy of the parent's to the edit's copy of sig-a.
        object.__setattr__(signal_a, "annotations", ())
        edited = edit(child, [other.id] if edit is remove_records else [other.signals[0].id])
        (question,) = edited.import_from(reader, tasks=[ids["question"]]).tasks.values()
    (copy,) = (copy for copy in edited.records if copy.id == record.id)
    (artifact,) = _signal_named(copy, "sig-a").annotations
    assert question.input_annotations[0] is artifact
    assert (signal_a.annotations, child.tasks) == ((), ())


def test_remove_signals_drops_the_entries_of_the_task_copies_it_removes(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=[ids["edit"]])
        targets = imported.tasks[ids["edit"]].targets
        assert targets is not None
        target, _ = targets
        assert isinstance(target, Signal)
        trimmed = remove_signals(child, [target.id], cascade=True)
        trimmed_dir = store(tmp_path, trimmed)
    assert trimmed.tasks == ()
    assert (_ALIAS, ObjectKind.SIGNAL, ids["sig-e1"]) not in trimmed._import_state.import_map
    assert _task_imports(trimmed) == {}
    assert not [row for row in _rows(trimmed_dir) if row[0] == "Task"]
    assert set(_task_imports(child)) == {(_ALIAS, ObjectKind.TASK, ids["edit"])}


def test_a_child_read_back_reuses_its_task_copies(tmp_path, monkeypatch):
    def unexpected_read(*args):
        pytest.fail("an import must not read a task the child holds a copy of")

    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, tasks=[ids["question"]])
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            (record,) = reread.records
            (question,) = reread.tasks_of(AnswerTask)
            tasks, annotations = reread.tasks, reread.annotations
            with monkeypatch.context() as patch:
                patch.setattr(DuckDBControlReader, "_read_tasks_with_records", unexpected_read)
                again = reread.import_from(reader, tasks=[ids["question"], ids["label"]])
            (locate,) = reread.import_from(reader, tasks=[ids["locate"]]).tasks.values()
            _next_version(reread)
            again_dir = store(tmp_path, reread)
    assert again.tasks[ids["question"]] is question
    assert again.tasks[ids["label"]] is question.from_tasks[0]
    assert (locate.inputs, locate.from_tasks) == ((record,), (question,))
    # The copy continues the task counter the stored version kept.
    assert locate.id == 3
    assert reread.tasks == (*tasks, locate)
    assert (reread.records, reread.annotations) == ((record,), annotations)
    with TimeFReader(open_local_version(again_dir)) as stored:
        assert stored.get(Task, ids["locate"], parent=_ALIAS).from_tasks[0].id == question.id


def test_a_cascading_removal_drops_task_copies_that_a_later_import_brings_back(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["other"]], tasks=[ids["question"]])
        other = imported.records[ids["other"]]
        (record,) = imported.tasks[ids["question"]].inputs
        edited = remove_records(child, [record.id], cascade=True)
        assert ([copy.id for copy in edited.records], edited.tasks) == ([other.id], ())
        # The map links only the copies the edit holds: no Record, Signal, or task copy it removed.
        assert set(edited._import_state.import_map) == {
            (_ALIAS, ObjectKind.RECORD, ids["other"]),
            (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]),
        }
        edited_dir = store(tmp_path, edited)
        (question,) = edited.import_from(reader, tasks=[ids["question"]]).tasks.values()
    # The written edit has rows only for the copies it holds, none for the removed tasks.
    assert [(kind, parent_id) for kind, parent_id, _ in _rows(edited_dir)] == [
        ("Record", ids["other"]),
        ("Signal", ids["sig-c"]),
    ]
    # The new copies take new IDs: the edit kept the counters.
    assert question.id > imported.tasks[ids["question"]].id
    assert edited.records[0].id == other.id
    assert edited.records[1].id > record.id
    assert [task.id for task in edited.tasks] == [question.id, question.from_tasks[0].id]
    # The dataset annotation that the first import brought stays, and the new copy references it.
    (note,) = edited.annotations
    assert question.input_annotations[2] is note
    assert set(_task_imports(edited)) == {
        (_ALIAS, ObjectKind.TASK, ids["label"]),
        (_ALIAS, ObjectKind.TASK, ids["question"]),
    }
    assert len(child.tasks) == 2


def test_get_by_parent_id_finds_no_copy_an_edit_removed(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["other"]], tasks=[ids["question"]])
        (record,) = imported.tasks[ids["question"]].inputs
        edited_dir = store(tmp_path, remove_records(child, [record.id], cascade=True))
    with TimeFReader(open_local_version(edited_dir)) as stored:
        for kind, name in ((Record, "rec"), (Signal, "sig-a"), (Task, "question")):
            with pytest.raises(
                TimeFNotFoundError, match=rf"no {kind.__name__} {ids[name]} imported under parent alias 'base'"
            ):
                stored.get(kind, ids[name], parent=_ALIAS)
        with pytest.raises(TimeFNotFoundError, match=rf"no Record {ids['rec']} imported"):
            stored.get_many(Record, [ids["other"], ids["rec"]], parent=_ALIAS)
        assert stored.get(Record, ids["other"], parent=_ALIAS).id == imported.records[ids["other"]].id


def test_two_children_of_one_parent_keep_metadata_separate_and_reuse_payloads(tmp_path):
    original = make_dataset()
    original_id = original.records[0].id
    base_path = store(tmp_path, original)
    with TimeFReader(open_local_version(base_path)) as reader:
        children = []
        for name in ("left", "right"):
            child = _child(reader, dataset_id=f"test/{name}")
            imported = child.import_from(reader, records=[original_id])
            imported.records[original_id].metadata["branch"] = name
            children.append((store(tmp_path, child), imported.records[original_id].id))
    readers = [TimeFReader(open_local_version(path)) for path, _ in children]
    try:
        merged = _child(*readers, dataset_id="test/merged", aliases=(ParentAlias("left"), ParentAlias("right")))
        for reader, (_, identity) in zip(readers, children, strict=True):
            merged.import_from(reader, records=[identity])
        # Both children numbered their copy 1; the merged child gives each copy an ID of its own.
        assert [identity for _, identity in children] == [1, 1]
        assert [record.id for record in merged.records] == [1, 2]
        merged.records[0].metadata["branch"] = "edited"
        assert merged.records[1].metadata["branch"] == "right"
        merged_path = store(tmp_path, merged)
    finally:
        for reader in readers:
            reader.close()
    version = open_local_version(merged_path)
    assert not version.manifest.files.time_series
    sources = version.manifest.payload_sources.values()
    assert {source.release.dataset_id for source in sources} == {original.metadata.dataset_id}
    for path in [base_path, *(path for path, _ in children)]:
        (path / "control.duckdb").rename(path / "unavailable.duckdb")
    with TimeFReader(version) as reader:
        for record in reader.read().records:
            assert record.signals[0].to_arrow().equals(original.records[0].signals[0].to_arrow())


def _annotation(annotated, key):
    return next(annotation for annotation in annotated.annotations if annotation.key == key)


def _signal_named(record, name):
    return next(signal for signal in record.signals if signal.name == name)


def _tasks_tree():
    """Return ``_tree()`` with tasks over it, two candidate Records, and a Record of edited Signals.

    ``label`` reads ``rec`` and its ``age``. ``question`` derives from ``label``, scopes ``sig-a``,
    and references ``sig-a``'s ``artifact``, ``monitor``'s ``device``, and the dataset's
    ``release-note``; ``locate`` derives from ``question``. ``match`` picks ``pool-1`` from two
    candidates, ``edit`` targets ``sig-e1`` of ``edited``, and ``pooled`` has no candidates, which
    ``from-pooled`` derives from. ``second`` references ``age``, ``late``, and ``release-note``. The
    dataset's ``unused-note`` is referenced by no task.

    Returns:
        The dataset, and the ID it gave each of its Records, Signals, and tasks, by name.
    """
    dataset = _tree()
    rec, other = dataset.records
    (root,) = rec.sources
    signal_a = _signal_named(rec, "sig-a")
    pools = [
        dataset.add_record(record=Record(sources=(Source(name="monitor", signals=(_signal(f"sig-p{index}"),)),)))
        for index in (1, 2)
    ]
    edited = dataset.add_record(
        record=Record(sources=(Source(name="monitor", signals=(_signal("sig-e1"), _signal("sig-e2"))),))
    )
    (release_note,) = dataset.annotations
    dataset.annotate(Annotation(key="unused-note", value="not referenced"))
    label = ClassificationTask(
        inputs=(rec,),
        targets=("normal",),
        target_schema="scp",
        split=Split.TRAIN,
        input_annotations=(_annotation(rec, "age"),),
        metadata={"rater": "a"},
    )
    label.annotate(Annotation(key="quality", value="good"))
    question = AnswerTask(
        inputs=(rec,),
        prompt="Is lead I clean?",
        rationale="No artifact after 0.25 s.",
        targets=("yes",),
        scope=TimeInterval.seconds(0.0, 0.5, time_series=(signal_a,)),
        input_annotations=(_annotation(signal_a, "artifact"), _annotation(root, "device"), release_note),
        from_tasks=(label,),
    )
    locate = TemporalLocalizationTask(
        inputs=(rec,),
        targets=(TimeInterval.seconds(2.0, 2.5, time_series=(_signal_named(rec, "sig-b"),)),),
        mode=LocalizationMode.EXHAUSTIVE,
        from_tasks=(question,),
    )
    pooled = TSCorrespondenceTask(inputs=(other,), targets=("none",))
    tasks = {
        "label": label,
        "question": question,
        "locate": locate,
        "match": TSCorrespondenceTask(inputs=(other,), candidate_records=tuple(pools), targets=(pools[0],)),
        "edit": TSEditingTask(inputs=(other,), targets=(edited.signals[0], "denoised")),
        "pooled": pooled,
        "from-pooled": AnswerTask(inputs=(other,), targets=("x",), from_tasks=(pooled,)),
        "second": AnswerTask(
            inputs=(rec,),
            targets=("no",),
            input_annotations=(_annotation(rec, "age"), _annotation(rec, "late"), release_note),
        ),
    }
    dataset.add_tasks(tasks=tasks.values())
    records = {"rec": rec, "other": other, "pool-1": pools[0], "pool-2": pools[1], "edited": edited}
    ids = {name: record.id for name, record in records.items()}
    ids |= {signal.name: signal.id for record in dataset.records for signal in record.signals}
    return dataset, ids | {name: task.id for name, task in tasks.items()}


def _task_imports(dataset):
    return {key: value for key, value in dataset._import_state.import_map.items() if key[1] is ObjectKind.TASK}


def test_a_task_import_brings_its_inputs_whole_and_the_annotations_it_references(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=[ids["question"]])
        parent_keys = _all_keys(reader.get(Record, ids["rec"]))
    (question,) = imported.tasks.values()
    (record,) = child.records
    (root,) = record.sources
    signal_a = _signal_named(record, "sig-a")
    assert list(imported.tasks) == [ids["question"]]
    assert imported.records == imported.signals == {}
    assert isinstance(question, AnswerTask)
    # The copies took the first IDs of the child's task counter, and the map holds them.
    (label,) = question.from_tasks
    assert (question.id, label.id) == (1, 2)
    assert _task_imports(child) == {
        (_ALIAS, ObjectKind.TASK, ids["question"]): 1,
        (_ALIAS, ObjectKind.TASK, ids["label"]): 2,
    }
    assert question.inputs == (record,)
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]
    assert _all_keys(record) == parent_keys
    # Each reference is the copy on its owner's copy, and the dataset's own annotation is the child's.
    artifact, device, note = question.input_annotations
    assert artifact is _annotation(signal_a, "artifact")
    assert device is _annotation(root, "device")
    assert child.annotations == (note,)
    assert note.value == "first release"
    assert question.scope is not None
    assert question.scope.time_series is not None
    assert question.scope.time_series[0] is signal_a
    assert (question.scope.start_us, question.scope.exclusive_end) == (0, 500_000)
    assert (question.prompt, question.rationale, question.targets) == (
        "Is lead I clean?",
        "No artifact after 0.25 s.",
        ("yes",),
    )
    assert child.tasks_for(record) == child.tasks


def test_a_task_import_brings_the_tasks_it_derives_from_transitively(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=[ids["locate"]])
        (parent_quality,) = reader.get(Task, ids["label"]).annotations
    (locate,) = imported.tasks.values()
    (question,) = locate.from_tasks
    (label,) = question.from_tasks
    assert child.tasks == (locate, question, label)
    assert (type(locate), type(question), type(label)) == (TemporalLocalizationTask, AnswerTask, ClassificationTask)
    (record,) = child.records
    assert locate.inputs == question.inputs == label.inputs == (record,)
    assert locate.targets is not None
    (target,) = locate.targets
    assert isinstance(target, TimeInterval)
    assert target.time_series is not None
    (named,) = target.time_series
    assert named is _signal_named(record, "sig-b")
    # Every field comes, including those of a task class's own.
    assert locate.mode is LocalizationMode.EXHAUSTIVE
    assert (label.target_schema, label.split, label.metadata, label.targets) == (
        "scp",
        Split.TRAIN,
        {"rater": "a"},
        ("normal",),
    )
    assert label.input_annotations[0] is _annotation(record, "age")
    # The task's own annotation is a copy of the parent's.
    (quality,) = label.annotations
    assert quality.value == "good"
    assert quality is not parent_quality


def test_a_task_copy_holds_copies_of_the_tasks_own_annotations(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    read_tasks = TimeFReader._read_tasks_with_records
    parent_tasks = {}

    def keeping(self, task_ids, *, load):
        read = read_tasks(self, task_ids, load=load)
        # The read returns the tasks by ID first.
        parent_tasks.update(read[0])
        return read

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "_read_tasks_with_records", keeping)
        (label,) = child.import_from(reader, tasks=[ids["label"]]).tasks.values()
    # The parent annotation is the object that the import read, not the object of another read.
    (original,) = parent_tasks[ids["label"]].annotations
    (copy,) = label.annotations
    assert copy is not original
    assert copy == original
    assert (copy.confidence, copy.occurrence_metadata) == (original.confidence, original.occurrence_metadata)


def test_a_task_import_brings_candidate_records_and_record_targets_whole(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        (match,) = child.import_from(reader, tasks=[ids["match"]]).tasks.values()
        copies = {
            name: child.import_from(reader, records=[ids[name]]).records[ids[name]]
            for name in ("other", "pool-1", "pool-2")
        }
    assert child.records == tuple(copies.values())
    assert match.inputs == (copies["other"],)
    assert match.candidate_records == (copies["pool-1"], copies["pool-2"])
    assert match.targets == (copies["pool-1"],)


def test_a_signal_target_brings_a_partial_record(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        (edit,) = child.import_from(reader, tasks=[ids["edit"]]).tasks.values()
        other = child.import_from(reader, records=[ids["other"]]).records[ids["other"]]
        signal = child.import_from(reader, signals=[ids["sig-e1"]]).signals[ids["sig-e1"]]
    (edited,) = (record for record in child.records if record is not other)
    assert edit.inputs == (other,)
    assert edit.targets == (signal, "denoised")
    assert edited.signals == (signal,)


def test_only_the_dataset_annotations_a_task_references_come_along(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=[ids["rec"]], tasks=[ids["label"]])
        assert child.annotations == ()
        child.import_from(reader, tasks=[ids["question"]])
        (note,) = child.annotations
        child.import_from(reader, tasks=[ids["locate"]], records=[ids["other"]])
    assert child.annotations == (note,)
    assert note.key == "release-note"


def test_a_repeated_task_import_reuses_the_copies(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        first = child.import_from(reader, tasks=[ids["question"]])
        (record,) = child.records
        tasks, annotations, hierarchy = child.tasks, child.annotations, _all_keys(record)

        def unexpected_read(*args):
            pytest.fail("an import must not read a task the child holds a copy of")

        with monkeypatch.context() as patch:
            patch.setattr(DuckDBControlReader, "_read_tasks_with_records", unexpected_read)
            again = child.import_from(reader, tasks=[ids["question"], ids["label"]])
        assert again.tasks[ids["question"]] is first.tasks[ids["question"]]
        assert again.tasks[ids["label"]] is first.tasks[ids["question"]].from_tasks[0]
        assert (child.tasks, child.annotations, _all_keys(record)) == (tasks, annotations, hierarchy)
        (locate,) = child.import_from(reader, tasks=[ids["locate"]]).tasks.values()
    assert locate.from_tasks == (first.tasks[ids["question"]],)
    assert child.tasks == (*tasks, locate)
    assert child.records == (record,)
    assert locate.id == 3
    assert set(_task_imports(child)) == {
        (_ALIAS, ObjectKind.TASK, ids[name]) for name in ("label", "question", "locate")
    }


def test_a_task_imported_again_keeps_the_childs_edits_of_its_copy(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        question = child.import_from(reader, tasks=[ids["question"]]).tasks[ids["question"]]
        # The child edits its copy and drops one of the copy's references.
        question.input_annotations = question.input_annotations[:1]
        question.metadata["checked"] = True
        tasks, annotations = child.tasks, child.annotations
        again = child.import_from(reader, tasks=[ids["question"]]).tasks[ids["question"]]
    assert again is question
    assert (len(again.input_annotations), again.metadata) == (1, {"checked": True})
    assert (child.tasks, child.annotations) == (tasks, annotations)


def test_a_task_import_after_a_record_import_reuses_the_copies(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        record.metadata["site"] = "edited"
        hierarchy = [
            (container, container.annotations) for container in (record, *record.walk_sources(), *record.signals)
        ]
        (question,) = child.import_from(reader, tasks=[ids["question"]]).tasks.values()
    assert child.records == (record,)
    assert record.metadata == {"site": "edited"}
    assert [
        (container, container.annotations) for container in (record, *record.walk_sources(), *record.signals)
    ] == hierarchy
    (root,) = record.sources
    artifact, device, _ = question.input_annotations
    assert question.inputs == question.from_tasks[0].inputs == (record,)
    assert artifact is _annotation(_signal_named(record, "sig-a"), "artifact")
    assert device is _annotation(root, "device")
    assert question.from_tasks[0].input_annotations[0] is _annotation(record, "age")


def test_an_annotation_the_child_removed_comes_back_for_a_task_that_references_it(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        # A connector may replace the copied annotations with its own.
        record.annotations = ()
        record.annotate(Annotation(key="note", value="mine"))
        (label,) = child.import_from(reader, tasks=[ids["label"]]).tasks.values()
        store(tmp_path, child)
    (age,) = label.input_annotations
    assert child.records == (record,)
    (held,) = child.records
    assert [(annotation.key, annotation.value) for annotation in held.annotations] == [("note", "mine"), ("age", 64)]
    assert age is held.annotations[-1]


def test_an_annotation_removed_beside_others_of_its_key_comes_back(tmp_path):
    parent = _tree()
    rec = parent.records[0]
    labels = [rec.annotate(Annotation(key="label", value=value)) for value in ("NORM", "MI")]
    mi_task = parent.add_task(task=AnswerTask(inputs=(rec,), targets=("x",), input_annotations=(labels[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[rec.id]).records[rec.id]
        record.annotations = tuple(annotation for annotation in record.annotations if annotation.value != "MI")
        (task,) = child.import_from(reader, tasks=[mi_task.id]).tasks.values()
        store(tmp_path, child)
    (mi,) = task.input_annotations
    assert mi is record.annotations[-1]
    assert [annotation.value for annotation in record.annotations if annotation.key == "label"] == ["NORM", "MI"]


def test_an_annotation_of_equal_content_the_child_attached_by_hand_serves_a_task(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        # The child holds the content of the parent's age, attached without annotate(): an
        # occurrence is the object itself, so it is the copy the task references.
        mine = Annotation(key="age", value=64)
        record.annotations = (mine,)
        (label,) = child.import_from(reader, tasks=[ids["label"]]).tasks.values()
    (age,) = label.input_annotations
    assert age is mine
    assert record.annotations == (mine,)


def _edit_age(record):
    record.annotations = tuple(
        replace(annotation, value=65) if annotation.key == "age" else annotation for annotation in record.annotations
    )


def _replace_age(record):
    record.annotations = tuple(annotation for annotation in record.annotations if annotation.key != "age")
    record.annotate(Annotation(key="age", value=65))


@pytest.mark.parametrize("change", [_edit_age, _replace_age], ids=["edited", "replaced"])
def test_a_task_that_references_an_annotation_the_child_changed_gets_a_copy_of_the_parents(tmp_path, change):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        # The child changes age from 64 to 65.
        change(record)
        (label,) = child.import_from(reader, tasks=[ids["label"]]).tasks.values()
        child_dir = store(tmp_path, child)
    # The Record holds both, and the task references the parent's 64.
    assert sorted(annotation.value for annotation in record.annotations if annotation.key == "age") == [64, 65]
    (age,) = label.input_annotations
    assert (age.value, age is record.annotations[-1]) == (64, True)
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(Task, ids["label"], parent=_ALIAS)
        assert [annotation.value for annotation in written.input_annotations] == [64]


def test_task_copies_that_fail_to_register_leave_no_task_and_no_map_entry(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    validate = TimeFDataset._validate_task_batch

    def refuse(self, batch):
        validate(self, batch)
        raise TimeFValidationError("refused for the test")

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        (root,) = record.sources
        signal_a = _signal_named(record, "sig-a")
        # Every reference of question, label, and second attaches a copy again: to a Signal, a
        # Source, the dataset, and two to the Record.
        record.annotations = root.annotations = ()
        object.__setattr__(signal_a, "annotations", ())
        imports, last_ids = dict(child._import_state.import_map), child.last_ids
        with monkeypatch.context() as patch:
            patch.setattr(TimeFDataset, "_validate_task_batch", refuse)
            with pytest.raises(TimeFValidationError, match="refused for the test"):
                child.import_from(reader, tasks=[ids["question"], ids["second"]], records=[ids["other"]])
        assert child.tasks == ()
        assert child.tasks_for(record) == ()
        assert (record.annotations, root.annotations, signal_a.annotations, child.annotations) == ((), (), (), ())
        # The Records stay imported, and no task is in the map or took an ID.
        other = child._import_state.import_map[_ALIAS, ObjectKind.RECORD, ids["other"]]
        assert [copy.id for copy in child.records] == [record.id, other]
        assert child._import_state.import_map.keys() - imports.keys() == {
            (_ALIAS, ObjectKind.RECORD, ids["other"]),
            (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]),
        }
        assert _task_imports(child) == {}
        assert child.last_ids[ObjectKind.TASK] == last_ids[ObjectKind.TASK]
        # Once the cause is gone, the same import completes.
        question, _ = child.import_from(reader, tasks=[ids["question"], ids["second"]]).tasks.values()
    assert question.id == 1
    assert len(child.tasks) == 3
    assert set(_task_imports(child)) == {
        (_ALIAS, ObjectKind.TASK, ids[name]) for name in ("label", "question", "second")
    }


def test_a_correspondence_task_drawing_on_the_whole_parent_is_rejected(tmp_path):
    parent, ids = _tasks_tree()
    pooled = rf"TSCorrespondenceTask {ids['pooled']} of test/tree@1\.0\.0 has no candidate_records"
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        with pytest.raises(TimeFValidationError, match=rf"{pooled}.*explicit candidate_records"):
            child.import_from(reader, records=[ids["rec"]], tasks=[ids["match"], ids["pooled"]])
        # It holds for a task that a selected task derives from, too.
        with pytest.raises(TimeFValidationError, match=pooled):
            child.import_from(reader, tasks=[ids["from-pooled"]])
    assert (child.records, child.tasks, child.parent_cards, child._import_state.import_map) == ((), (), {}, {})


def test_a_child_that_streams_its_tasks_cannot_import_tasks(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.set_task_stream([AnswerTask], lambda: iter(()))
        with pytest.raises(TimeFValidationError, match="streams its tasks"):
            child.import_from(reader, records=[ids["rec"]], tasks=[ids["question"]])
        assert (child.records, child.parent_cards, child._import_state.import_map) == ((), {}, {})
        child.import_from(reader, records=[ids["rec"]])
    assert len(child.records) == 1


def test_a_task_import_reads_only_the_tasks_and_the_records_it_needs(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    batches = []
    read_records = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        requested = None if record_ids is None else tuple(record_ids)
        batches.append(requested)
        return read_records(self, requested, **kwargs)

    def unexpected_read(*args, **kwargs):
        pytest.fail("an import must not read the whole parent")

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "read", unexpected_read)
        monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
        child.import_from(reader, records=[ids["rec"], ids["pool-2"]], tasks=[ids["locate"], ids["edit"]])
    # The task read builds the Records the tasks reference, and the Record that holds the Signal
    # target, and the import copies those; it reads only the other selected Records. The tasks and
    # the tasks they derive from build together, so their Records arrive in one read.
    assert batches == [(ids["rec"], ids["other"]), (ids["edited"],), (ids["pool-2"],)]


def test_a_written_child_reads_its_task_copies_by_parent_id(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=[ids["locate"], ids["edit"]])
        child_dir = store(tmp_path, child)
    tasks = _task_imports(child)
    assert {parent_id for _, _, parent_id in tasks} == {ids[name] for name in ("edit", "label", "locate", "question")}
    assert [row for row in _rows(child_dir) if row[0] == "Task"] == sorted(
        ("Task", parent_id, child_id) for (_, _, parent_id), child_id in tasks.items()
    )
    with TimeFReader(open_local_version(child_dir)) as stored:
        question = stored.get(AnswerTask, ids["question"], parent=_ALIAS)
        assert question.prompt == "Is lead I clean?"
        assert [annotation.key for annotation in question.input_annotations] == ["artifact", "device", "release-note"]
        assert question.from_tasks[0].id == imported.tasks[ids["locate"]].from_tasks[0].from_tasks[0].id
        targets = stored.get(Task, ids["edit"], parent=_ALIAS).targets
        assert targets is not None
        signal, _ = targets
        assert isinstance(signal, Signal)
        assert signal.to_numpy().tolist() == list(range(16))
        with pytest.raises(TimeFNotFoundError, match="the stored object is a TemporalLocalizationTask"):
            stored.get(AnswerTask, ids["locate"], parent=_ALIAS)


def _annotated_elsewhere():
    """Return a parent whose tasks read Record ``rec``, target ``sig-n`` of Record ``far``, and reference far.

    ``far`` holds the Source ``near`` with ``sig-n`` (0 to 1 s), the Source ``tail`` with ``sig-l``
    (2 to 3 s), and the Signal-less Source ``spare``. Each task but ``plain`` references one
    annotation of far as context: one on ``far`` itself without a span (``static``) and one at
    2.5 s (``late``), one on ``near`` without a span (``near-note``), one on ``sig-l``
    (``sig-note``), and one on ``spare`` (``spare-note``). ``plain`` references ``rec-note`` of rec.

    Returns:
        The parent, and the ID it gave ``far`` and each task, by name.
    """
    parent = TimeFDataset(metadata=_metadata("test/far"))
    rec = parent.add_record(record=Record(sources=(Source(name="m", signals=(_signal("sig-r"),)),)))
    near = Source(name="near", signals=(_signal("sig-n"),))
    tail = Source(name="tail", signals=(_signal("sig-l", offset_s=2),))
    spare = Source(name="spare")
    far = parent.add_record(record=Record(sources=(near, tail, spare)))
    annotations = {
        "static": far.annotate(Annotation(key="static", value="far")),
        "late": far.annotate(Annotation(key="late", span=TimePoint.seconds(2.5))),
        "near": near.annotate(Annotation(key="near-note", value="near")),
        "sig": tail.signals[0].annotate(Annotation(key="sig-note", value="tail")),
        "spare": spare.annotate(Annotation(key="spare-note", value="spare")),
    }
    # Each task references far only through its Signal target.
    tasks = {
        name: TSEditingTask(inputs=(rec,), targets=(near.signals[0],), input_annotations=(annotation,))
        for name, annotation in annotations.items()
    }
    rec_note = rec.annotate(Annotation(key="rec-note", value="rec"))
    tasks["plain"] = TSEditingTask(inputs=(rec,), targets=(near.signals[0],), input_annotations=(rec_note,))
    parent.add_tasks(tasks=tasks.values())
    return parent, {"far": far.id} | {name: task.id for name, task in tasks.items()}


def _holding(record):
    return sorted(signal.name for signal in record.signals), sorted(source.name for source in record.walk_sources())


_WHOLE_FAR = (["sig-l", "sig-n"], ["near", "spare", "tail"])


@pytest.mark.parametrize("task", ["static", "late", "near", "sig", "spare", "plain"])
def test_the_record_of_a_signal_target_comes_whole_when_it_holds_a_referenced_annotation(tmp_path, task):
    parent, ids = _annotated_elsewhere()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        (copy,) = child.import_from(reader, tasks=[ids[task]]).tasks.values()
        far = child._records_by_id[child._import_state.import_map[_ALIAS, ObjectKind.RECORD, ids["far"]]]
        child_dir = store(tmp_path, child)
    # A task that references no annotation of far brings a partial copy of it, as signals= does.
    assert _holding(far) == ((["sig-n"], ["near"]) if task == "plain" else _WHOLE_FAR)
    assert child.annotations == ()
    assert child.tasks_for(far) == ()
    (annotation,) = copy.input_annotations
    owner = copy.inputs[0] if task == "plain" else far
    assert any(annotation is held for held in owner.walk_annotations())
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(TSEditingTask, ids[task], parent=_ALIAS).input_annotations
        assert [(item.key, item.value, item.span) for item in written] == [
            (item.key, item.value, item.span) for item in copy.input_annotations
        ]


def test_a_reference_on_no_record_the_read_of_the_tasks_built_is_rejected(tmp_path, monkeypatch):
    parent, ids = _annotated_elsewhere()
    read = TimeFReader._read_tasks_with_records

    def without_far(self, task_ids, *, load):
        tasks, records, on_dataset = read(self, task_ids, load=load)
        # The import then reads far on its own, so the reference lies on no Record that it copies.
        return tasks, {identity: record for identity, record in records.items() if identity != ids["far"]}, on_dataset

    monkeypatch.setattr(TimeFReader, "_read_tasks_with_records", without_far)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        with pytest.raises(
            TimeFValidationError,
            match=r"references annotation 'static', which lies on no Record this import copies nor on the parent",
        ):
            child.import_from(reader, tasks=[ids["static"]])
    assert (child.tasks, child.annotations, _task_imports(child)) == ((), (), {})


def test_a_task_reference_on_a_source_without_signals_finds_its_copy_by_name(tmp_path):
    parent = TimeFDataset(metadata=_metadata("test/spare"))
    spare = Source(name="spare")
    record = parent.add_record(record=Record(sources=(Source(name="dev", signals=(_signal("sig-x"),)), spare)))
    note = spare.annotate(Annotation(key="spare-note", value="first"))
    task = parent.add_task(task=AnswerTask(inputs=(record,), targets=("x",), input_annotations=(note,)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        copy = child.import_from(reader, records=[record.id]).records[record.id]
        spare_copy = _named(copy, "spare")
        # The copy of spare holds no Signal, so only its name finds it for the reference.
        task_copy = child.import_from(reader, tasks=[task.id]).tasks[task.id]
    assert _named(copy, "spare") is spare_copy
    (annotation,) = task_copy.input_annotations
    assert annotation is spare_copy.annotations[0]


def test_a_partial_copy_of_the_record_of_a_signal_target_grows_whole_for_a_referenced_annotation(tmp_path):
    parent, ids = _annotated_elsewhere()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, tasks=[ids["plain"]])
        far = child._records_by_id[child._import_state.import_map[_ALIAS, ObjectKind.RECORD, ids["far"]]]
        assert _holding(far) == (["sig-n"], ["near"])
        tasks = child.import_from(reader, tasks=[ids[name] for name in ("static", "late", "spare")]).tasks
        child.import_from(reader, records=[ids["far"]])
    # The copy grew whole once, and it holds each annotation once.
    assert _holding(far) == _WHOLE_FAR
    assert _all_keys(far) == [
        ("record", ["late", "static"]),
        ("near", ["near-note"]),
        ("spare", ["spare-note"]),
        ("tail", []),
    ]
    held = {annotation.key: annotation for annotation in far.walk_annotations()}
    references = [id(task.input_annotations[0]) for task in tasks.values()]
    assert references == [id(held[key]) for key in ("static", "late", "spare-note")]


def test_task_copies_reuse_the_parent_payloads_of_packed_text(tmp_path, monkeypatch):
    text = "A long clinical question. " * 300
    parent = _tree()
    long = parent.add_task(task=AnswerTask(inputs=parent.records[:1], prompt=text, rationale=text, targets=(text,)))
    parent_dir = store(tmp_path, parent)

    def unexpected_read(*args):
        pytest.fail("a task import must not read or copy stored text")

    with TimeFReader(open_local_version(parent_dir)) as reader, monkeypatch.context() as patch:
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        child = _child(reader)
        child.import_from(reader, tasks=[long.id])
        child_dir = store(tmp_path, child)
    assert sorted(path.name for path in child_dir.iterdir()) == ["control.duckdb", "manifest.json"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        task = stored.get(AnswerTask, long.id, parent=_ALIAS)
        assert isinstance(task.prompt, StoredText)
        assert task.prompt._location.source == "test/tree@1.0.0"
        assert (task.prompt, task.rationale, task.targets) == (text, text, (text,))


def test_an_eager_task_import_loads_the_copies_that_the_tasks_need(tmp_path, monkeypatch):
    texts = {"question": "A question. " * 500, "earlier": "An earlier question. " * 300, "report": "A report. " * 600}
    parent = TimeFDataset(metadata=_metadata("test/eager-tasks"))
    record = parent.add_record(record=Record(sources=(Source(name="monitor", signals=(_signal("sig-r"),)),)))
    holder = parent.add_record(
        record=Record(sources=(Source(name="monitor", signals=(_signal("sig-h"), _signal("sig-x", offset_s=2))),))
    )
    holder.annotate(Annotation(key="report", value=texts["report"]))
    base = AnswerTask(inputs=(record,), prompt=texts["earlier"], targets=("yes",))
    edit = TSEditingTask(inputs=(record,), prompt=texts["question"], targets=(holder.signals[0],), from_tasks=(base,))
    parent.add_tasks(tasks=(base, edit))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        loads = _count_array_loads(monkeypatch)
        imported = child.import_from(reader, records=[record.id], tasks=[edit.id], load=LoadPlan.ALL_EAGER)
        # sig-r of the input and the Signal target sig-h load. sig-x of the holder stays in the parent.
        assert len(loads) == 2
    # The parent release is gone, so only values that the import loaded can still be read.
    shutil.rmtree(parent_dir)
    copy = imported.tasks[edit.id]
    assert isinstance(copy.prompt, StoredText)
    assert (copy.prompt.read(), copy.from_tasks[0].prompt.read()) == (texts["question"], texts["earlier"])
    # The read of the tasks built the Record that records= selects too, so its copy is the input.
    (input_copy,) = copy.inputs
    assert input_copy is imported.records[record.id]
    assert copy.targets is not None
    (target,) = copy.targets
    assert isinstance(target, Signal)
    assert input_copy.signals[0].to_numpy().tolist() == target.to_numpy().tolist() == list(range(16))
    # The partial copy of the Record that holds the Signal target keeps the Record's annotation.
    partial = next(copied for copied in child.records if copied is not input_copy)
    assert [signal.name for signal in partial.signals] == ["sig-h"]
    (report,) = partial.annotations
    assert isinstance(report.value, StoredText)
    assert report.value.read() == texts["report"]


@pytest.mark.parametrize("first", ["none", "records"], ids=["same-call", "after-records"])
def test_identical_occurrences_a_task_references_stay_distinct(tmp_path, first):
    parent, both = _beats_parent()
    rec_id = parent.records[0].id
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, **({"records": [rec_id]} if first == "records" else {}))
        (task,) = child.import_from(reader, tasks=[both]).tasks.values()
    (record,) = child.records
    copies = [annotation for annotation in record.annotations if annotation.key == "beat"]
    # The parent's reader may return the two in either order, so only their distinctness shows.
    assert len(copies) == 2
    first_beat, second_beat = task.input_annotations
    assert first_beat is not second_beat
    assert {id(first_beat), id(second_beat)} == {id(copy) for copy in copies}


@pytest.mark.parametrize("first", ["none", "records"], ids=["same-call", "after-records"])
def test_tasks_of_one_call_that_reference_one_annotation_share_its_copy(tmp_path, first):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, **({"records": [ids["rec"]]} if first == "records" else {}))
        imported = child.import_from(reader, tasks=[ids[name] for name in ("label", "question", "second")])
        child_dir = store(tmp_path, child)
    label, question, second = imported.tasks.values()
    (record,) = child.records
    assert second.input_annotations[0] is label.input_annotations[0] is _annotation(record, "age")
    assert second.input_annotations[2] is question.input_annotations[2]
    assert [annotation.key for annotation in record.annotations].count("age") == 1
    assert [annotation.key for annotation in child.annotations] == ["release-note"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(AnswerTask, ids["second"], parent=_ALIAS)
        assert [annotation.key for annotation in written.input_annotations] == ["age", "late", "release-note"]


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("span", (TimePoint.seconds(0.25), TimePoint.seconds(0.5))),
        ("source", ("rater-a", "rater-b")),
        ("confidence", (0.5, 0.75)),
        ("occurrence_metadata", ({"by": "a"}, {"by": "b"})),
    ],
)
def test_a_reference_to_an_earlier_copy_matches_every_field_of_the_occurrence(tmp_path, name, values):
    parent = _tree()
    rec = parent.records[0]
    flags = [rec.annotate(Annotation(key="flag", value="x", **{name: value})) for value in values]
    flagged = parent.add_task(task=AnswerTask(inputs=(rec,), targets=("x",), input_annotations=(flags[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[rec.id]).records[rec.id]
        # Put the copy of the other flag first, so that only the field tells the two apart.
        record.annotations = tuple(
            sorted(record.annotations, key=lambda annotation: getattr(annotation, name) == values[1])
        )
        wrong, right = (annotation for annotation in record.annotations if annotation.key == "flag")
        (task,) = child.import_from(reader, tasks=[flagged.id]).tasks.values()
    (flag,) = task.input_annotations
    assert getattr(wrong, name) == values[0]
    assert flag is right
    assert [annotation for annotation in record.annotations if annotation.key == "flag"] == [wrong, right]


def _annotated_record():
    """Return a Record with ``sig-a``, ``sig-b``, and the ordinal ``sig-s``, and annotations at every level."""
    signal_a = _signal("sig-a")
    ordinal = Signal.from_values(np.arange(8, dtype=np.float32), spec=_SPEC, name="sig-s", time_axis=OrdinalAxis())
    aux = Source(name="aux")
    root = Source(
        name="monitor",
        signals=(_signal("sig-b", offset_s=2), ordinal),
        sources=(Source(name="leads", signals=(signal_a,)), aux),
    )
    record = Record(sources=(root,))
    record.add_annotations(
        [Annotation(key="age", value=64, unit="years"), Annotation(key="late", span=TimePoint.seconds(2.5))]
    )
    root.annotate(Annotation(key="device", value="m1", source="tech", confidence=0.5))
    aux.annotate(Annotation(key="aux-note", value="spare", occurrence_metadata={"by": "x"}))
    signal_a.annotate(Annotation(key="artifact", value="motion", span=TimeInterval.seconds(0.0, 0.25)))
    return record


def _every_task_class():
    """Return a parent with a task of every class, with every field set, over ``_annotated_record()``.

    ``far`` holds the Signal that ``fc`` forecasts, and ``gen`` has no inputs.

    Returns:
        The parent, and the ID it gave each task, by name.
    """
    dataset = TimeFDataset(metadata=_metadata("test/tree"))
    rec = dataset.add_record(record=_annotated_record())
    sources = {source.name: source for source in rec.walk_sources()}
    signal_a, ordinal = _signal_named(rec, "sig-a"), _signal_named(rec, "sig-s")
    other, pool_1, pool_2, far = (
        dataset.add_record(record=Record(sources=(Source(name="m", signals=tuple(map(_signal, signals))),)))
        for signals in (["sig-c"], ["sig-p1"], ["sig-p2"], ["sig-f1", "sig-f2"])
    )
    note = dataset.annotate(Annotation(key="release-note", value="first release"))
    artifact = _annotation(signal_a, "artifact")
    classify = ClassificationTask(
        inputs=(rec,),
        targets=("normal",),
        target_schema="scp",
        split=Split.TRAIN,
        input_annotations=(_annotation(rec, "age"), _annotation(sources["aux"], "aux-note"), note),
        rationale="because",
        metadata={"rater": "a", "nested": {"values": [1, 2]}},
        input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
    )
    classify.annotate(Annotation(key="quality", value="good"))
    answer = AnswerTask(
        inputs=(rec,),
        prompt="Is lead I clean?",
        rationale="No artifact after 0.25 s.",
        targets=("yes",),
        scope=TimeInterval.seconds(0.0, 0.5, time_series=(signal_a,)),
        input_annotations=(artifact, _annotation(sources["monitor"], "device")),
        from_tasks=(classify,),
        split=Split.TEST,
    )
    generate = TSGenerationTask(
        prompt="make a sine",
        targets=("ok",),
        input_modalities=frozenset({InputModality.NO_INPUT, InputModality.TEXT}),
        input_annotations=(note,),
    )
    tasks = {
        "cls": classify,
        "ans": answer,
        "scalar": ScalarPredictionTask(
            inputs=(rec,), targets=(64.5, 3), unit="mV", target_name="amp", from_tasks=(answer, classify)
        ),
        "loc": TemporalLocalizationTask(
            inputs=(rec,),
            targets=(
                TimeInterval.seconds(2.0, 2.5, time_series=(_signal_named(rec, "sig-b"),)),
                StepPoint(time_series=ordinal, start=3),
            ),
            scope=StepInterval(time_series=ordinal, start=0, stop=6),
            mode=LocalizationMode.EXHAUSTIVE,
        ),
        "loc-ann": TemporalLocalizationTask(inputs=(rec,), target_annotations=(artifact, _annotation(rec, "late"))),
        "fc": ForecastingTask(inputs=(rec,), targets=(far.signals[0],)),
        "edit": TSEditingTask(inputs=(other,), targets=(signal_a, "denoised", True)),
        "gen": generate,
        "corr": TSCorrespondenceTask(
            inputs=(other,), candidate_records=(pool_1, pool_2), targets=(pool_2,), from_tasks=(generate,)
        ),
    }
    dataset.add_tasks(tasks=tasks.values())
    return dataset, {name: task.id for name, task in tasks.items()}


def _same(identity):
    return identity


def _portrait(task, *, record_id=_same, signal_id=_same, task_id=_same):
    """Return every field of a task, with each reference as its kind and mapped ID and each text read.

    An annotation shows where it lies: on the task, on the task's inputs, or on the dataset. A span
    shows the mapped IDs of the Signals it names.
    """
    owners = {}
    for record in task.inputs:
        owners.update((id(annotation), ("Record", record_id(record.id))) for annotation in record.annotations)
        for source in record.walk_sources():
            owners.update(
                (id(annotation), ("Source", record_id(record.id), source.name)) for annotation in source.annotations
            )
            for signal in source.signals:
                owners.update((id(annotation), ("Signal", signal_id(signal.id))) for annotation in signal.annotations)

    def text(value):
        return value.read() if isinstance(value, TextHandle | TextListHandle) else value

    def span(value):
        named = value.time_series
        if isinstance(value, StepSpan):
            signals = signal_id(named.id)
        else:
            signals = None if named is None else tuple(signal_id(signal.id) for signal in named)
        bounds = tuple((item.name, getattr(value, item.name)) for item in fields(value) if item.name != "time_series")
        return type(value).__name__, bounds, signals

    def portrait(value, name):
        if isinstance(value, tuple):
            return tuple(portrait(item, name) for item in value)
        if isinstance(value, Annotation):
            owner = ("Task",) if name == "annotations" else owners.get(id(value), ("Dataset",))
            fixed = ("key", "span", "unit", "description", "source", "confidence", "metadata", "occurrence_metadata")
            return owner, text(value.value), tuple(getattr(value, field) for field in fixed)
        if isinstance(value, StepSpan | TimeSpan):
            return span(value)
        for kind, mapped in ((Record, record_id), (Signal, signal_id), (Task, task_id)):
            if isinstance(value, kind):
                return kind.__name__, mapped(value.id), value.to_numpy().tolist() if isinstance(value, Signal) else None
        return text(value)

    return {
        field.name: portrait(getattr(task, field.name), field.name) for field in fields(task) if field.name != "_id"
    } | {
        "id": task_id(task.id),
        "class": type(task),
    }


def test_every_task_class_round_trips_through_a_written_child(tmp_path):
    parent, ids = _every_task_class()
    parent_ids = [ids[name] for name in ids]
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=parent_ids)
        imports = {(kind, identity): copy for (_, kind, identity), copy in child._import_state.import_map.items()}
        assert all(isinstance(copy, int) for copy in imports.values())

        def mapped(kind):
            return lambda identity: imports[kind, identity]

        expected = [
            _portrait(
                task,
                record_id=mapped(ObjectKind.RECORD),
                signal_id=mapped(ObjectKind.SIGNAL),
                task_id=mapped(ObjectKind.TASK),
            )
            for task in reader.get_many(Task, parent_ids)
        ]
        assert [_portrait(task) for task in imported.tasks.values()] == expected
        child_dir = store(tmp_path, child)
    assert {type(task) for task in imported.tasks.values()} == set(TASKS.values())
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert [_portrait(task) for task in stored.get_many(Task, parent_ids, parent=_ALIAS)] == expected


def test_a_dataset_annotation_of_the_childs_own_with_the_key_of_a_referenced_one_stays(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        own = child.annotate(Annotation(key="release-note", value="the child's own"))
        (question,) = child.import_from(reader, tasks=[ids["question"]]).tasks.values()
    own_note, note = child.annotations
    assert own_note is own
    assert question.input_annotations[2] is note
    assert note.value == "first release"


def test_an_annotation_the_child_edited_into_another_of_its_key_gets_a_copy_of_the_parents(tmp_path):
    parent = _tree()
    rec = parent.records[0]
    labels = [rec.annotate(Annotation(key="label", value=value)) for value in ("NORM", "MI")]
    mi_task = parent.add_task(task=AnswerTask(inputs=(rec,), targets=("x",), input_annotations=(labels[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[rec.id]).records[rec.id]
        # Now two copies hold NORM.
        record.annotations = tuple(
            replace(annotation, value="NORM") if annotation.value == "MI" else annotation
            for annotation in record.annotations
        )
        (task,) = child.import_from(reader, tasks=[mi_task.id]).tasks.values()
        store(tmp_path, child)
    (mi,) = task.input_annotations
    assert (mi.value, mi is record.annotations[-1]) == ("MI", True)
    assert [annotation.value for annotation in record.annotations if annotation.key == "label"] == [
        "NORM",
        "NORM",
        "MI",
    ]


def _beats_parent():
    """Return ``_tree()`` whose ``rec`` holds two identical ``beat`` occurrences, both referenced by one task.

    Returns:
        The parent, and the ID it gave the task.
    """
    parent = _tree()
    rec = parent.records[0]
    beats = (rec.annotate(Annotation(key="beat", value="N")), rec.annotate(Annotation(key="beat", value="N")))
    both = parent.add_task(task=AnswerTask(inputs=(rec,), targets=("N",), input_annotations=beats))
    return parent, both.id


@pytest.mark.parametrize("removed", [0, 1])
def test_a_copy_attached_again_beside_an_identical_one_is_an_occurrence_of_its_own(tmp_path, removed):
    parent, both = _beats_parent()
    rec_id = parent.records[0].id
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[rec_id]).records[rec_id]
        kept = [annotation for annotation in record.annotations if annotation.key == "beat"]
        # Either copy may be the one the first reference takes, so each case runs once.
        record.annotations = tuple(annotation for annotation in record.annotations if annotation is not kept[removed])
        (task,) = child.import_from(reader, tasks=[both]).tasks.values()
        child_dir = store(tmp_path, child)
    first, second = task.input_annotations
    assert first is not second
    beats = [annotation for annotation in record.annotations if annotation.key == "beat"]
    assert len(beats) == 2
    assert {id(first), id(second)} == {id(annotation) for annotation in beats}
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(AnswerTask, both, parent=_ALIAS)
        (copy,) = stored.read().records
        stored_beats = [annotation for annotation in copy.annotations if annotation.key == "beat"]
        assert len(stored_beats) == 2
        assert [annotation.value for annotation in written.input_annotations] == ["N", "N"]


def test_identical_dataset_annotations_referenced_across_calls_stay_two_occurrences(tmp_path):
    parent = _tree()
    rec = parent.records[0]
    first, second = (parent.annotate(Annotation(key="note", value="same")) for _ in range(2))
    one, both = parent.add_tasks(
        tasks=[
            AnswerTask(inputs=(rec,), targets=("x",), input_annotations=(first,)),
            AnswerTask(inputs=(rec,), targets=("x",), input_annotations=(second, first)),
        ]
    )
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, tasks=[one.id])
        (copy,) = child.import_from(reader, tasks=[both.id]).tasks.values()
        child_dir = store(tmp_path, child)
    assert len(child.annotations) == 2
    assert {id(annotation) for annotation in copy.input_annotations} == {
        id(annotation) for annotation in child.annotations
    }
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert len(stored.read().annotations) == 2


def _pairs(spread, count):
    """Return ``count`` pairs of annotations that differ in value: all in one bucket, or a bucket each."""
    if spread == "one-bucket":
        return [(Annotation(key="label", value=f"a{i}"), Annotation(key="label", value=f"b{i}")) for i in range(count)]
    return [
        (
            Annotation(key="beat", value="a", span=TimePoint.micros(i * 1000)),
            Annotation(key="beat", value="b", span=TimePoint.micros(i * 1000)),
        )
        for i in range(count)
    ]


@pytest.mark.parametrize("spread", ["one-bucket", "many-buckets"])
def test_resolving_references_takes_steps_linear_in_them(tmp_path, monkeypatch, spread):
    count = 300
    parent = _tree()
    rec = parent.records[0]
    pairs = [(rec.annotate(kept), rec.annotate(removed)) for kept, removed in _pairs(spread, count)]
    # The task references the second of each pair, last first.
    references = tuple(removed for _, removed in reversed(pairs))
    many = parent.add_task(task=AnswerTask(inputs=(rec,), targets=("x",), input_annotations=references))
    steps = {"_take": 0, "_same_occurrence": 0}

    def counted(name):
        function = getattr(composition, name)

        def counting(*args):
            steps[name] += 1
            return function(*args)

        return counting

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[rec.id]).records[rec.id]
        values = {"b", *(f"b{i}" for i in range(count))}
        # The child removes the copy of the second of each pair, which the task then attaches again.
        record.annotations = tuple(
            annotation for annotation in record.annotations if read_value(annotation.value) not in values
        )
        for name in steps:
            monkeypatch.setattr(composition, name, counted(name))
        (task,) = child.import_from(reader, tasks=[many.id]).tasks.values()
    assert len(task.input_annotations) == count
    assert all(read_value(annotation.value) in values for annotation in task.input_annotations)
    assert max(steps.values()) <= 4 * count
