from contextlib import contextmanager
from dataclasses import fields

import numpy as np
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.dataset.composition import ImportedDataset
from timenet.dataset.edit import remove_records
from timenet.errors import TimeFNotFoundError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import PackedValueReader, StoredOffsets, StoredText
from timenet.reader import TimeFReader
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
    ureg,
)
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
    names = ("_records_by_id", "_signals_by_id", "_signals_by_record_id", "_imports")
    return {name: dict(getattr(dataset, name)) for name in names} | {
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
    assert child._imports == {(_ALIAS, ObjectKind.RECORD, ids["rec"]): 1, (_ALIAS, ObjectKind.SIGNAL, ids["sig-a"]): 1}


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
    assert child._imports == {
        (_ALIAS, ObjectKind.RECORD, ids["other"]): other.id,
        (_ALIAS, ObjectKind.RECORD, ids["rec"]): rec.id,
        (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]): other.signals[0].id,
        **{(_ALIAS, ObjectKind.SIGNAL, ids[signal.name]): signal.id for signal in rec.signals},
    }
    assert all(isinstance(value, int) for value in child._imports.values())


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
        partial.import_from(reader, records=[ids["rec"]])
        whole = _child(reader)
        whole.import_from(reader, records=[ids["rec"]])
    assert "gap" in _keys(partial.records[0])
    assert "gap" in _keys(whole.records[0])


def _drop(owner, *keys):
    owner.annotations = tuple(annotation for annotation in owner.annotations if annotation.key not in keys)


def _remove_sig_b(record):
    """Remove sig-b from a copy of ``rec`` by hand, as removing a Signal does.

    It drops the annotations that only sig-b places: the ones at 2.5 s, and "gap", which no window
    places, so only a complete copy holds it.
    """
    (root,) = record.sources
    root.signals = tuple(signal for signal in root.signals if signal.name != "sig-b")
    _drop(record, "late", "gap")
    _drop(root, "src-late")


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_signal_removed_from_a_copy_comes_back_with_its_annotations_once(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        keys = _all_keys(record)
        _remove_sig_b(record)
        child.import_from(reader, signals=[ids["sig-b"]])
        sources = list(record.walk_sources())
        assert child.import_from(reader, records=[ids["rec"]]).records == {ids["rec"]: record}
        child_dir = store(tmp_path, child)
    assert _all_keys(record) == keys
    assert list(record.walk_sources()) == sources
    assert sorted(source.name for source in sources) == ["aux", "leads", "monitor"]
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert _all_keys(stored.get(Record, ids["rec"], parent=_ALIAS)) == keys


def _same_name_parent(*, lead=False):
    """Return a parent whose Record ``r`` holds two sibling Sources named ``dev``.

    The first ``dev`` holds ``sig-x`` (0 to 1 s) and the second ``sig-y`` (2 to 3 s), and each has
    an annotation in its Signal's window. Two more siblings, both named ``spare``, hold no Signal.
    With ``lead``, a first Source named ``lead`` holds ``sig-w1`` and ``sig-w2`` (4 to 5 s).
    """
    parent = TimeFDataset(metadata=_metadata("test/same"))
    z = Source(name="dev", signals=(_signal("sig-x"),))
    a = Source(name="dev", signals=(_signal("sig-y", offset_s=2),))
    spares = (Source(name="spare"), Source(name="spare"))
    leads = (Source(name="lead", signals=(_signal("sig-w1", offset_s=4), _signal("sig-w2", offset_s=4))),)
    parent.add_record(record=Record(sources=(*(leads if lead else ()), z, a, *spares)))
    z.annotate(Annotation(key="z-note", span=TimePoint.seconds(0.5)))
    a.annotate(Annotation(key="a-note", span=TimePoint.seconds(2.5)))
    return parent


def _empty(source):
    """Remove the Signals of a Source's copy by hand, with the annotations that only they place."""
    source.signals = ()
    source.annotations = ()


def _hierarchy_of(record):
    return [(source, source.signals, source.annotations) for source in record.walk_sources()]


def test_sibling_sources_of_one_name_are_told_apart_by_their_signals(tmp_path):
    parent = _same_name_parent()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["r"]]).records[ids["r"]]
        (z_copy,) = (source for source in record.sources if source.signals and source.signals[0].name == "sig-x")
        (a_copy,) = (source for source in record.sources if source.name == "dev" and source is not z_copy)
        _empty(a_copy)
        child.import_from(reader, signals=[ids["sig-y"]])
        # sig-x identifies the copy of the first dev, so the name finds the other one.
        assert [(_keys(copy), [signal.name for signal in copy.signals]) for copy in (z_copy, a_copy)] == [
            (["z-note"], ["sig-x"]),
            (["a-note"], ["sig-y"]),
        ]
        # A repeated import brings nothing below them, nor below the two spare Sources, which only
        # their name could identify, so it leaves every copy as it is.
        hierarchy = _hierarchy_of(record)
        child.import_from(reader, records=[ids["r"]])
        assert _hierarchy_of(record) == hierarchy
        assert sorted(source.name for source in record.sources) == ["dev", "dev", "spare", "spare"]


@pytest.mark.parametrize("first", ["records", "signals"], ids=["two-copies", "one-copy"])
def test_sibling_sources_of_one_name_that_nothing_tells_apart_are_rejected(tmp_path, first):
    parent = _same_name_parent()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        selection = {"records": [ids["r"]]} if first == "records" else {"signals": [ids["sig-x"]]}
        child.import_from(reader, **selection)
        (record,) = child.records
        # The child empties every copy, so no imported Signal identifies one: with two copies the
        # name matches both, and with one it could copy either parent Source.
        for source in record.sources:
            if source.name == "dev":
                _empty(source)
        hierarchy = _hierarchy_of(record)
        indexes = _indexes(child)
        with pytest.raises(TimeFValidationError, match="cannot tell apart the Sources named 'dev' of Record"):
            child.import_from(reader, signals=[ids["sig-y"]])
    assert _hierarchy_of(record) == hierarchy
    assert _indexes(child) == indexes


def test_an_extension_that_fails_leaves_the_copy_and_the_indexes_as_they_were(tmp_path):
    parent = _same_name_parent(lead=True)
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-w1"], ids["sig-x"]])
        (record,) = child.records
        lead, dev = record.sources
        _empty(dev)
        hierarchy = [(record.sources, record.annotations), *_hierarchy_of(record)]
        indexes = _indexes(child)
        # The call adds sig-w2 to the copy of lead in place, then finds no way to tell the dev copies
        # apart for sig-y: the copy goes back to what it was.
        with pytest.raises(TimeFValidationError, match="cannot tell apart the Sources named 'dev'"):
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
    assert (child.metadata.access, child.metadata.access_url) == (Access.RESTRICTED, "https://r.example")
    assert child.records == ()
    assert child._imports == {}


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


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_repeated_imports_do_not_repeat_an_annotation_no_window_places(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=[ids["sig-a"]])
        for _ in range(2):
            child.import_from(reader, records=[ids["rec"]], signals=[ids["sig-b"]])
    assert _keys(child.records[0]) == ["age", "early", "gap", "late"]


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
    expected = sorted((kind.value, parent_id, child_id) for (_, kind, parent_id), child_id in child._imports.items())
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
    assert (child.records, child.parent_cards, child.lineage, child._imports) == ((), {}, {}, {})


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
        carded.parent_cards = dict(child.parent_cards)
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
        child.parent_cards[_ALIAS] = card
        with pytest.raises(TimeFValidationError, match=r"already imports test/tree@1\.0\.0, not test/tree@1\.0\.1"):
            child.import_from(second, records=[ids["rec"]])
    assert child.records == ()


def test_an_import_takes_the_strictest_access_of_its_parents(tmp_path):
    parents = {
        ParentAlias("credentialed"): _tree("test/c", access=Access.CREDENTIALED, access_url="https://c.example"),
        ParentAlias("restricted"): _tree("test/r", access=Access.RESTRICTED, access_url="https://r.example"),
        ParentAlias("also"): _tree("test/also", access=Access.RESTRICTED, access_url="https://also.example"),
    }
    readers = {alias: TimeFReader(open_local_version(store(tmp_path, parent))) for alias, parent in parents.items()}
    try:
        child = _child(*readers.values(), aliases=tuple(readers))
        steps = []
        for reader in readers.values():
            child.import_from(reader)
            steps.append((child.metadata.access, child.metadata.access_url))
    finally:
        for reader in readers.values():
            reader.close()
    assert steps == [
        (Access.CREDENTIALED, "https://c.example"),
        (Access.RESTRICTED, "https://r.example"),
        (Access.RESTRICTED, "https://r.example"),
    ]
    # An equally strict parent keeps its own access page in its card.
    assert child.parent_cards[ParentAlias("also")].metadata.access_url == "https://also.example"


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
        (record,) = reader.read().records
        assert _keys(record) == ["age", "early", "late"]
        assert [signal.to_numpy().tolist() for signal in record.signals] == [list(range(16))] * 2


def test_the_writer_rejects_copies_imported_under_an_alias_without_a_parent_card(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, records=[ids["rec"]])
        del child.parent_cards[_ALIAS]
        with pytest.raises(TimeFValidationError, match=r"parent aliases \['base'\] without a parent card"):
            store(tmp_path, child)


def _without_rec(child, imported, rec_id):
    """Remove the copy of ``rec``, carrying the import map and the parent cards through the edit by hand."""
    edited = remove_records(child, [imported.records[rec_id].id])
    edited._imports = dict(child._imports)
    edited.parent_cards = dict(child.parent_cards)
    return edited


def test_an_import_copies_again_what_the_child_no_longer_holds(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    selected = [ids["rec"], ids["other"]]
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=selected)
        edited = _without_rec(child, imported, ids["rec"])
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


def test_the_writer_stores_import_rows_only_for_copies_the_child_holds(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"], ids["other"]])
        edited_dir = store(tmp_path, _without_rec(child, imported, ids["rec"]))
    kept = imported.records[ids["other"]]
    assert _rows(edited_dir) == [("Record", ids["other"], kept.id), ("Signal", ids["sig-c"], kept.signals[0].id)]
    with TimeFReader(open_local_version(edited_dir)) as stored, pytest.raises(TimeFNotFoundError):
        stored.get(Record, ids["rec"], parent=_ALIAS)


def test_the_writer_ignores_stale_entries_of_an_alias_the_child_dropped(tmp_path):
    parent = _tree()
    ids = _ids(parent)
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=[ids["rec"]])
        edited = remove_records(child, [imported.records[ids["rec"]].id])
        # The map still names the removed copies, but the edit dropped the parent's card.
        edited._imports = dict(child._imports)
        edited_dir = store(tmp_path, edited)
    assert _rows(edited_dir) == []


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

    ``label`` reads ``rec``. ``question`` derives from ``label`` and scopes ``sig-a``, and ``locate``
    derives from ``question``. ``match`` picks ``pool-1`` from two candidates, ``edit`` targets
    ``sig-e1`` of ``edited``, and ``pooled`` has no candidates, which ``from-pooled`` derives from.

    Returns:
        The dataset, and the ID it gave each of its Records, Signals, and tasks, by name.
    """
    dataset = _tree()
    rec, other = dataset.records
    signal_a = _signal_named(rec, "sig-a")
    pools = [
        dataset.add_record(record=Record(sources=(Source(name="monitor", signals=(_signal(f"sig-p{index}"),)),)))
        for index in (1, 2)
    ]
    edited = dataset.add_record(
        record=Record(sources=(Source(name="monitor", signals=(_signal("sig-e1"), _signal("sig-e2"))),))
    )
    label = ClassificationTask(
        inputs=(rec,),
        targets=("normal",),
        target_schema="scp",
        split=Split.TRAIN,
        metadata={"rater": "a"},
    )
    label.annotate(Annotation(key="quality", value="good"))
    question = AnswerTask(
        inputs=(rec,),
        prompt="Is lead I clean?",
        rationale="No artifact after 0.25 s.",
        targets=("yes",),
        scope=TimeInterval.seconds(0.0, 0.5, time_series=(signal_a,)),
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
    }
    dataset.add_tasks(tasks=tasks.values())
    records = {"rec": rec, "other": other, "pool-1": pools[0], "pool-2": pools[1], "edited": edited}
    ids = {name: record.id for name, record in records.items()}
    ids |= {signal.name: signal.id for record in dataset.records for signal in record.signals}
    return dataset, ids | {name: task.id for name, task in tasks.items()}


def _task_imports(dataset):
    return {key: value for key, value in dataset._imports.items() if key[1] is ObjectKind.TASK}


def test_a_task_import_brings_its_inputs_whole(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=[ids["question"]])
        parent_keys = _all_keys(reader.get(Record, ids["rec"]))
    (question,) = imported.tasks.values()
    (record,) = child.records
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
    assert child.annotations == ()
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
    # The task's own annotation is a copy of the parent's.
    (quality,) = label.annotations
    assert quality.value == "good"
    assert quality is not parent_quality


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
        question.scope = None
        question.metadata["checked"] = True
        tasks, annotations = child.tasks, child.annotations
        again = child.import_from(reader, tasks=[ids["question"]]).tasks[ids["question"]]
    assert again is question
    assert (again.scope, again.metadata) == (None, {"checked": True})
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
    assert question.inputs == question.from_tasks[0].inputs == (record,)


def test_task_copies_that_fail_to_register_leave_no_task_and_no_map_entry(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    validate = TimeFDataset._validate_task_batch

    def refuse(self, batch):
        validate(self, batch)
        raise TimeFValidationError("refused for the test")

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=[ids["rec"]]).records[ids["rec"]]
        imports, last_ids = dict(child._imports), child.last_ids
        with monkeypatch.context() as patch:
            patch.setattr(TimeFDataset, "_validate_task_batch", refuse)
            with pytest.raises(TimeFValidationError, match="refused for the test"):
                child.import_from(reader, tasks=[ids["question"]], records=[ids["other"]])
        assert child.tasks == ()
        assert child.tasks_for(record) == ()
        # The Records stay imported, and no task is in the map or took an ID.
        other = child._imports[_ALIAS, ObjectKind.RECORD, ids["other"]]
        assert [copy.id for copy in child.records] == [record.id, other]
        assert child._imports.keys() - imports.keys() == {
            (_ALIAS, ObjectKind.RECORD, ids["other"]),
            (_ALIAS, ObjectKind.SIGNAL, ids["sig-c"]),
        }
        assert _task_imports(child) == {}
        assert child.last_ids[ObjectKind.TASK] == last_ids[ObjectKind.TASK]
        # Once the cause is gone, the same import completes.
        (question,) = child.import_from(reader, tasks=[ids["question"]]).tasks.values()
    assert question.id == 1
    assert len(child.tasks) == 2
    assert set(_task_imports(child)) == {(_ALIAS, ObjectKind.TASK, ids[name]) for name in ("label", "question")}


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
    assert (child.records, child.tasks, child.parent_cards, child._imports) == ((), (), {}, {})


@pytest.mark.parametrize("field", ["input_annotations", "target_annotations"])
def test_a_task_that_references_annotations_is_rejected_before_the_child_changes(tmp_path, field):
    parent = _tree()
    rec = parent.records[0]
    context = field == "input_annotations"
    age = (_annotation(rec, "age"),)
    referencing = AnswerTask(
        inputs=(rec,),
        targets=("x",) if context else None,
        input_annotations=age if context else (),
        target_annotations=() if context else age,
    )
    derived = AnswerTask(inputs=(rec,), targets=("y",), from_tasks=(referencing,))
    parent.add_tasks(tasks=[referencing, derived])
    message = rf"AnswerTask {referencing.id} of test/tree@1\.0\.0 references annotations in {field}"
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        # A task that derives from it brings it along, so the import refuses that task too.
        for task in (referencing, derived):
            with pytest.raises(TimeFValidationError, match=message):
                child.import_from(reader, records=[rec.id], tasks=[task.id])
    assert (child.records, child.tasks, child.parent_cards, child._imports) == ((), (), {}, {})


def test_a_child_that_streams_its_tasks_cannot_import_tasks(tmp_path):
    parent, ids = _tasks_tree()
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.set_task_stream([AnswerTask], lambda: iter(()))
        with pytest.raises(TimeFValidationError, match="streams its tasks"):
            child.import_from(reader, records=[ids["rec"]], tasks=[ids["question"]])
        assert (child.records, child.parent_cards, child._imports) == ((), {}, {})
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
        assert question.from_tasks[0].id == imported.tasks[ids["locate"]].from_tasks[0].from_tasks[0].id
        targets = stored.get(Task, ids["edit"], parent=_ALIAS).targets
        assert targets is not None
        signal, _ = targets
        assert isinstance(signal, Signal)
        assert signal.to_numpy().tolist() == list(range(16))
        with pytest.raises(TimeFNotFoundError, match="the stored object is a TemporalLocalizationTask"):
            stored.get(AnswerTask, ids["locate"], parent=_ALIAS)


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

    ``far`` holds the Signal that ``fc`` forecasts, and ``gen`` has no inputs. No task references an
    annotation, so ``input_annotations`` and ``target_annotations`` stay empty.

    Returns:
        The parent, and the ID it gave each task, by name.
    """
    dataset = TimeFDataset(metadata=_metadata("test/tree"))
    rec = dataset.add_record(record=_annotated_record())
    signal_a, ordinal = _signal_named(rec, "sig-a"), _signal_named(rec, "sig-s")
    other, pool_1, pool_2, far = (
        dataset.add_record(record=Record(sources=(Source(name="m", signals=tuple(map(_signal, signals))),)))
        for signals in (["sig-c"], ["sig-p1"], ["sig-p2"], ["sig-f1", "sig-f2"])
    )
    classify = ClassificationTask(
        inputs=(rec,),
        targets=("normal",),
        target_schema="scp",
        split=Split.TRAIN,
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
        from_tasks=(classify,),
        split=Split.TEST,
    )
    generate = TSGenerationTask(
        prompt="make a sine",
        targets=("ok",),
        input_modalities=frozenset({InputModality.NO_INPUT, InputModality.TEXT}),
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

    A span shows the mapped IDs of the Signals it names.
    """

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

    def portrait(value):
        if isinstance(value, tuple):
            return tuple(portrait(item) for item in value)
        if isinstance(value, Annotation):
            fixed = ("key", "span", "unit", "description", "source", "confidence", "metadata", "occurrence_metadata")
            return text(value.value), tuple(getattr(value, field) for field in fixed)
        if isinstance(value, StepSpan | TimeSpan):
            return span(value)
        for kind, mapped in ((Record, record_id), (Signal, signal_id), (Task, task_id)):
            if isinstance(value, kind):
                return kind.__name__, mapped(value.id), value.to_numpy().tolist() if isinstance(value, Signal) else None
        return text(value)

    return {field.name: portrait(getattr(task, field.name)) for field in fields(task) if field.name != "_id"} | {
        "id": task_id(task.id),
        "class": type(task),
    }


def test_every_task_class_round_trips_through_a_written_child(tmp_path):
    parent, ids = _every_task_class()
    parent_ids = [ids[name] for name in ids]
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=parent_ids)
        imports = {(kind, identity): copy for (_, kind, identity), copy in child._imports.items()}
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


def test_a_task_field_that_holds_references_outside_a_tuple_is_rejected(tmp_path, monkeypatch):
    parent, ids = _tasks_tree()
    read = TimeFReader._read_tasks_with_records

    def with_a_list(self, task_ids):
        tasks, records = read(self, task_ids)
        for task in tasks.values():
            # No JSON field holds Records, so set the value past the field's validation.
            object.__setattr__(task, "metadata", {"inputs": list(task.inputs)})
        return tasks, records

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "_read_tasks_with_records", with_a_list)
        with pytest.raises(
            TimeFValidationError, match=r"ClassificationTask\.metadata holds a dict that references objects"
        ):
            child.import_from(reader, tasks=[ids["label"]])
    assert (child.tasks, _task_imports(child)) == ((), {})
