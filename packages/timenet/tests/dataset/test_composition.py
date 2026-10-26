from contextlib import contextmanager

import numpy as np
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
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
    Access,
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


_ALIAS = InputAlias("base")
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
    inputs = {alias: _ref(reader) for alias, reader in zip(aliases, readers, strict=True)}
    return TimeFDataset(metadata=_metadata(dataset_id, inputs=inputs))


def _signal(signal_id, *, offset_s=0):
    """Return 16 values at 16 Hz, covering one second from ``offset_s``."""
    return Signal.from_values(
        np.arange(16, dtype=np.float32),
        spec=_SPEC,
        name=signal_id,
        time_axis=RegularAxis.from_rate_hz(16, offset_us=offset_s * 1_000_000),
        id=signal_id,
    )


def _tree(dataset_id="test/tree", *, time_span=None, **update):
    """Return a parent whose Record ``rec`` nests Sources and has annotations at every level.

    ``sig-a`` covers 0 to 1 s below ``monitor/leads``, and ``sig-b`` covers 2 to 3 s directly in
    ``monitor``. The annotations named ``late`` lie at 2.5 s, so only ``sig-b`` places them. The
    Source ``aux`` holds no Signal.
    """
    dataset = TimeFDataset(metadata=_metadata(dataset_id, **update))
    signal_a, signal_b = _signal("sig-a"), _signal("sig-b", offset_s=2)
    aux = Source(id="src-aux", name="aux")
    root = Source(
        id="src-root",
        name="monitor",
        metadata={"model": "m1"},
        signals=(signal_b,),
        sources=(Source(id="src-leads", name="leads", signals=(signal_a,)), aux),
    )
    record = dataset.add_record(
        record=Record(record_id="rec", metadata={"site": "north"}, sources=(root,), time_span=time_span)
    )
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
    dataset.add_record(
        record=Record(record_id="other", sources=(Source(id="src-other", name="monitor", signals=(_signal("sig-c"),)),))
    )
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
    names = ("_records_by_id", "_sources_by_id", "_signals_by_id", "_sources_by_record_id", "_signals_by_record_id")
    return {name: dict(getattr(dataset, name)) for name in names} | {"_imports": dict(dataset._imports)}


def _rows(release_dir):
    with connect_control(release_dir / "control.duckdb", read_only=True) as connection:
        return connection.execute(
            "SELECT object_kind, parent_id FROM imported_objects ORDER BY object_kind, parent_id"
        ).fetchall()


def test_a_signal_import_copies_its_source_chain_and_the_annotations_its_window_places(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, signals=["sig-a"])
        (record,) = child.records
        record.metadata["site"] = "edited"
        assert reader.get(Record, "rec").metadata == {"site": "north"}
    (root,) = record.sources
    (leads,) = root.sources
    (signal,) = leads.signals
    assert imported.signals == {"sig-a": signal}
    assert imported.records == {}
    assert (root.name, leads.name, root.signals) == ("monitor", "leads", ())
    assert root.metadata == {"model": "m1"}
    # The annotations at 2.5 s lie only on sig-b, which was not imported.
    assert _keys(record) == ["age", "early"]
    assert _keys(root) == ["device", "src-early"]
    assert _keys(leads) == []
    assert _keys(signal) == ["artifact"]
    assert child.annotations == ()
    assert {record.id, root.id, leads.id, signal.id}.isdisjoint({"rec", "src-root", "src-leads", "sig-a"})
    assert all(annotation.occurrence_id is not None for annotation in record.walk_annotations())


def test_a_record_import_extends_a_partial_copy_in_place(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
        (record,) = child.records
        (root,) = record.sources
        (leads,) = root.sources
        (signal_a,) = leads.signals
        record.metadata["site"] = "edited"
        root.metadata["model"] = "edited"
        imported = child.import_from(reader, records=["rec"])
        again = child.import_from(reader, signals=["sig-b"])
    assert imported.records == {"rec": record}
    assert (child.records, record.sources, leads.signals) == ((record,), (root,), (signal_a,))
    assert (record.metadata, root.metadata) == ({"site": "edited"}, {"model": "edited"})
    (signal_b,) = root.signals
    assert again.signals == {"sig-b": signal_b}
    # The dataset indexed the Signal the copy gained, so a task can scope a span to it.
    scope = TimeInterval.seconds(2.0, 2.5, time_series_ids=(signal_b.id,))
    child.add_task(task=ClassificationTask(inputs=(record,), targets=("late",), scope=scope))
    aux = next(source for source in root.sources if source.name == "aux")
    assert {source.name for source in root.sources} == {"leads", "aux"}
    assert _keys(record) == ["age", "early", "late"]
    assert _keys(root) == ["device", "src-early", "src-late"]
    assert _keys(aux) == ["aux-note"]
    assert _keys(signal_a) == ["artifact"]


def test_growing_a_partial_copy_adds_only_what_the_new_signal_places(tmp_path, monkeypatch):
    parent = _tree()
    parent.records[0].annotate(Annotation(key="report", value="A long clinical report. " * 300))
    parent_dir = store(tmp_path, parent)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
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
        child.import_from(reader, signals=["sig-b"])
    assert [(annotation.key, annotation.value) for annotation in record.annotations] == [("age", 65), ("late", None)]
    assert _keys(root) == ["src-late"]


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_complete_copy_keeps_annotations_its_windows_do_not_place(tmp_path):
    parent = _tree()
    # The parent keeps this annotation with a warning: 1.5 s lies between sig-a and sig-b.
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        partial = _child(reader)
        partial.import_from(reader, signals=["sig-a"])
        assert "gap" not in _keys(partial.records[0])
        partial.import_from(reader, records=["rec"])
        whole = _child(reader)
        whole.import_from(reader, records=["rec"])
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
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        keys = _all_keys(record)
        _remove_sig_b(record)
        child.import_from(reader, signals=["sig-b"])
        sources = list(record.walk_sources())
        assert child.import_from(reader, records=["rec"]).records == {"rec": record}
        child_dir = store(tmp_path, child)
    assert _all_keys(record) == keys
    assert list(record.walk_sources()) == sources
    assert sorted(source.name for source in sources) == ["aux", "leads", "monitor"]
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert _all_keys(stored.get(Record, "rec", alias=_ALIAS)) == keys


def _same_name_parent():
    """Return a parent whose Record ``r`` holds two sibling Sources named ``dev``.

    ``src-z`` holds ``sig-x`` (0 to 1 s) and ``src-a`` holds ``sig-y`` (2 to 3 s), and each has an
    annotation in its Signal's window. Two more siblings, both named ``spare``, hold no Signal.
    """
    parent = TimeFDataset(metadata=_metadata("test/same"))
    z = Source(id="src-z", name="dev", signals=(_signal("sig-x"),))
    a = Source(id="src-a", name="dev", signals=(_signal("sig-y", offset_s=2),))
    spares = (Source(id="src-s1", name="spare"), Source(id="src-s2", name="spare"))
    parent.add_record(record=Record(record_id="r", sources=(z, a, *spares)))
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
    with _reader(tmp_path, _same_name_parent()) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["r"]).records["r"]
        (z_copy,) = (source for source in record.sources if source.signals and source.signals[0].name == "sig-x")
        (a_copy,) = (source for source in record.sources if source.name == "dev" and source is not z_copy)
        _empty(a_copy)
        child.import_from(reader, signals=["sig-y"])
        # sig-x identifies the copy of src-z, so the name finds the other one.
        assert [(_keys(copy), [signal.name for signal in copy.signals]) for copy in (z_copy, a_copy)] == [
            (["z-note"], ["sig-x"]),
            (["a-note"], ["sig-y"]),
        ]
        # A repeated import brings nothing below them, nor below the two spare Sources, which only
        # their name could identify, so it leaves every copy as it is.
        hierarchy = _hierarchy_of(record)
        child.import_from(reader, records=["r"])
        assert _hierarchy_of(record) == hierarchy
        assert sorted(source.name for source in record.sources) == ["dev", "dev", "spare", "spare"]


@pytest.mark.parametrize("first", [{"records": ["r"]}, {"signals": ["sig-x"]}], ids=["two-copies", "one-copy"])
def test_sibling_sources_of_one_name_that_nothing_tells_apart_are_rejected(tmp_path, first):
    with _reader(tmp_path, _same_name_parent()) as reader:
        child = _child(reader)
        record = child.import_from(reader, **first).records.get("r") or child.records[0]
        # The child empties every copy, so no imported Signal identifies one: with two copies the
        # name matches both, and with one it could copy either parent Source.
        for source in record.sources:
            if source.name == "dev":
                _empty(source)
        hierarchy = _hierarchy_of(record)
        indexes = _indexes(child)
        with pytest.raises(TimeFValidationError, match="cannot tell apart the Sources named 'dev' of Record 'r'"):
            child.import_from(reader, signals=["sig-y"])
    assert _hierarchy_of(record) == hierarchy
    assert _indexes(child) == indexes


def test_an_extension_that_fails_leaves_the_copy_and_the_indexes_as_they_were(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        clash = _child(reader).import_from(reader, signals=["sig-b"]).signals["sig-b"].id
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
        (record,) = child.records
        # Another Record holds the ID that the copy of sig-b gets.
        child.add_record(record=Record(record_id="mine", sources=(Source(name="mine", signals=(_signal(clash),)),)))
        hierarchy = [(record.sources, record.annotations)] + [
            (source.sources, source.signals, source.annotations) for source in record.walk_sources()
        ]
        indexes = _indexes(child)
        with pytest.raises(TimeFValidationError, match="already belong to another Source"):
            child.import_from(reader, records=["rec"])
    assert [(record.sources, record.annotations)] + [
        (source.sources, source.signals, source.annotations) for source in record.walk_sources()
    ] == hierarchy
    assert _indexes(child) == indexes


def test_a_record_that_fails_keeps_the_parent_recorded(tmp_path):
    parent = _tree(access=Access.RESTRICTED, access_url="https://r.example")
    with _reader(tmp_path, parent) as reader:
        clash = _child(reader).import_from(reader, records=["rec"]).records["rec"].id
        child = _child(reader)
        child.add_record(record=Record(record_id=clash))
        with pytest.raises(TimeFValidationError, match="already registered"):
            child.import_from(reader, records=["rec", "other"])
    assert list(child.parent_cards) == [_ALIAS]
    assert set(child.lineage) == {"test/tree@1.0.0"}
    assert (child.metadata.access, child.metadata.access_url) == (Access.RESTRICTED, "https://r.example")
    assert [record.id for record in child.records] == [clash]
    assert child._imports == {}


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_a_partial_copy_of_a_record_with_a_session_span_keeps_every_annotation_in_it(tmp_path):
    parent = _tree(time_span=TimeInterval.seconds(0.0, 3.0))
    parent.records[0].annotate(Annotation(key="after", span=TimePoint.seconds(5.0)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
        (record,) = child.records
        # The session span places "late", which only sig-b's window covers; "after" lies past it.
        assert _keys(record) == ["age", "early", "late"]
        child.import_from(reader, records=["rec"])
    assert _keys(record) == ["after", "age", "early", "late"]


def test_reimporting_a_whole_record_changes_nothing_and_keeps_child_edits(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        first = child.import_from(reader, records=["rec"])
        record = first.records["rec"]
        # A connector may replace the copied annotations; a later import leaves them so.
        record.annotations = ()
        hierarchy = [(source, source.signals, source.annotations) for source in record.walk_sources()]
        second = child.import_from(reader, records=["rec"], signals=["sig-a"])
    assert second.records == {"rec": record}
    assert second.signals["sig-a"] is next(signal for signal in record.signals if signal.name == "sig-a")
    assert child.records == (record,)
    assert record.annotations == ()
    assert [(source, source.signals, source.annotations) for source in record.walk_sources()] == hierarchy
    assert len(hierarchy) == 3


@pytest.mark.filterwarnings("ignore::timenet.errors.SpanOutsideWindowWarning")
def test_repeated_imports_do_not_repeat_an_annotation_no_window_places(tmp_path):
    parent = _tree()
    parent.records[0].annotate(Annotation(key="gap", span=TimePoint.seconds(1.5)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
        for _ in range(2):
            child.import_from(reader, records=["rec"], signals=["sig-b"])
    assert _keys(child.records[0]) == ["age", "early", "gap", "late"]


def test_copies_reuse_the_parent_payloads_of_irregular_offsets_and_packed_text(tmp_path, monkeypatch):
    text = "A long clinical report. " * 300
    offsets = [0, 1_000, 5_000, 9_000]
    parent = TimeFDataset(metadata=_metadata("test/irregular"))
    signal = Signal.from_irregular(np.arange(4, dtype=np.float32), time_offsets_us=offsets, spec=_SPEC, name="env")
    record = parent.add_record(record=Record(record_id="rec", sources=(Source(name="env", signals=(signal,)),)))
    record.annotate(Annotation(key="report", value=text))
    parent_dir = store(tmp_path, parent)

    def unexpected_read(*args):
        pytest.fail("an import must not read or copy stored values")

    with TimeFReader(open_local_version(parent_dir)) as reader, monkeypatch.context() as patch:
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        patch.setattr(TimeFReader, "_read_array", unexpected_read)
        child = _child(reader)
        child.import_from(reader, records=["rec"])
        child_dir = store(tmp_path, child)
    assert sorted(path.name for path in child_dir.iterdir()) == ["control.duckdb", "manifest.json"]
    with TimeFReader(open_local_version(child_dir)) as reader:
        (copy,) = reader.read().records
        (report,) = copy.annotations
        assert isinstance(report.value, StoredText)
        assert isinstance(copy.signals[0]._time_offsets, StoredOffsets)
        assert report.value._location.owner == "test/irregular@1.0.0"
        assert copy.signals[0].time_offsets_us().tolist() == offsets
        assert copy.signals[0].to_numpy().tolist() == [0, 1, 2, 3]
        assert report.value.read() == text


def test_a_written_child_reads_its_copies_by_parent_id(tmp_path, monkeypatch):
    with _reader(tmp_path, _tree()) as reader:
        ref = _ref(reader)
        child = _child(reader)
        imported = child.import_from(reader, records=["rec"], signals=["sig-c"])
        imported.records["rec"].metadata["site"] = "edited"
        child_dir = store(tmp_path, child)
    # Sources and annotations are reached through their owners, so only Records and Signals get rows.
    assert _rows(child_dir) == [
        ("Record", "other"),
        ("Record", "rec"),
        ("Signal", "sig-a"),
        ("Signal", "sig-b"),
        ("Signal", "sig-c"),
    ]
    with TimeFReader(open_local_version(child_dir)) as stored:

        def unexpected_read(**_):
            raise AssertionError("a get by ID must not read the whole dataset")

        monkeypatch.setattr(stored, "read", unexpected_read)
        assert stored.get(Record, "rec", alias=_ALIAS).metadata["site"] == "edited"
        assert stored.get(Signal, "sig-c", parent=ref).id == imported.signals["sig-c"].id
        signals = stored.get_many(Signal, ["sig-a", "sig-b"], alias=_ALIAS)
        assert {signal.name for signal in signals} == {"sig-a", "sig-b"}
        with pytest.raises(TimeFValidationError, match="not both"):
            stored.get(Record, "rec", alias=_ALIAS, parent=ref)


def test_an_import_reads_only_the_selected_records(tmp_path, monkeypatch):
    batches = []
    read_records = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        requested = None if record_ids is None else tuple(record_ids)
        batches.append(requested)
        return read_records(self, requested, **kwargs)

    def unexpected_read(*args, **kwargs):
        pytest.fail("an import must not read the whole parent")

    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "read", unexpected_read)
        monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
        child.import_from(reader, records=["rec"], signals=["sig-c"])
    assert batches == [("rec", "other")]


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
    other = TimeFDataset(metadata=_metadata("test/other"))
    cosine = TimeSeriesSpec(spec_type="cosine", name="Cosine", unit_value=ureg.dimensionless)
    signal = Signal.from_values(
        np.zeros(4, dtype=np.float32), spec=cosine, name="c", time_axis=RegularAxis.from_rate_hz(4)
    )
    other.add_record(record=Record(record_id="rec-other", sources=(Source(name="monitor", signals=(signal,)),)))
    with _reader(tmp_path, _tree()) as tree, _reader(tmp_path, other) as second:
        aliases = (_ALIAS, InputAlias("other"))
        child = _child(tree, second, aliases=aliases)
        child.import_from(tree, records=["rec"])
        child_dir = store(tmp_path, child)
        with TimeFReader(open_local_version(child_dir)) as stored:
            reread = stored.read()
            assert reread.schema is not None
            assert [spec.spec_type for spec in reread.schema.time_series_specs] == ["sine"]
            reread._metadata = reread.metadata.model_copy(update={"dataset_version": Version(1, 0, 1)})
            reread.import_from(second, records=["rec-other"])
            assert reread.schema is None
            reread_dir = reread.write(path=tmp_path)
    specs = open_local_version(reread_dir).manifest.dataset_schema.time_series_specs
    assert sorted(spec.spec_type for spec in specs) == ["cosine", "sine"]


def test_an_unknown_parent_id_raises_before_the_child_changes(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        with pytest.raises(TimeFNotFoundError, match="missing"):
            child.import_from(reader, records=["rec", "missing"])
        with pytest.raises(TimeFNotFoundError, match="missing"):
            child.import_from(reader, signals=["sig-a", "missing"])
    assert (child.records, child.parent_cards, child.lineage, child._imports) == ((), {}, {}, {})


@pytest.mark.parametrize("selection", ["records", "signals"])
def test_a_selection_must_be_a_collection_of_parent_ids(tmp_path, selection):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        with pytest.raises(TimeFValidationError, match="not the string 'rec'"):
            child.import_from(reader, **{selection: "rec"})
        with pytest.raises(TimeFValidationError, match="parent IDs are strings"):
            child.import_from(reader, **{selection: [1]})
    assert child.records == ()


@pytest.mark.parametrize("alias", [None, _ALIAS])
def test_a_parent_the_card_does_not_declare_is_rejected(tmp_path, alias):
    with _reader(tmp_path, _tree()) as reader:
        child = TimeFDataset(metadata=_metadata("test/child"))
        with pytest.raises(TimeFValidationError, match=r"declare it in metadata\.inputs"):
            child.import_from(reader, alias=alias, records=["rec"])
    assert (child.records, child.parent_cards) == ((), {})


def test_an_alias_must_declare_the_parent(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        with pytest.raises(
            TimeFValidationError, match=r"input alias 'other' of test/child does not declare .*\['base'\]"
        ):
            child.import_from(reader, alias=InputAlias("other"), records=["rec"])
    assert (child.records, child.parent_cards) == ((), {})


def test_a_parent_declared_under_two_aliases_is_imported_under_one(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader, reader, aliases=(InputAlias("a"), InputAlias("b")))
        with pytest.raises(TimeFValidationError, match=r"under the input aliases \['a', 'b'\]; pass alias="):
            child.import_from(reader, records=["rec"])
        child.import_from(reader, alias=InputAlias("a"), records=["rec"])
        with pytest.raises(TimeFValidationError, match=r"already imports test/tree@1\.0\.0 under the input alias 'a'"):
            child.import_from(reader, alias=InputAlias("b"), records=["other"])
        # The check reads the parent cards, not the import map: a dataset with the cards and no map
        # is refused too.
        carded = _child(reader, reader, aliases=(InputAlias("a"), InputAlias("b")))
        carded.parent_cards = dict(child.parent_cards)
        with pytest.raises(TimeFValidationError, match="under the input alias 'a'"):
            carded.import_from(reader, alias=InputAlias("b"), records=["other"])
    assert len(child.records) == 1
    assert carded.records == ()


def test_an_alias_that_imports_another_release_is_rejected(tmp_path):
    with _reader(tmp_path, _tree()) as first:
        card = ParentDatasetCard(metadata=first.metadata)
    newer = _tree(dataset_version=Version(1, 0, 1))
    with _reader(tmp_path, newer) as second:
        # The card declares the newer release, while the parent cards it kept name the older one.
        child = _child(second)
        child.parent_cards[_ALIAS] = card
        with pytest.raises(TimeFValidationError, match=r"already imports test/tree@1\.0\.0, not test/tree@1\.0\.1"):
            child.import_from(second, records=["rec"])
    assert child.records == ()


def test_an_import_takes_the_strictest_access_of_its_parents(tmp_path):
    parents = {
        InputAlias("credentialed"): _tree("test/c", access=Access.CREDENTIALED, access_url="https://c.example"),
        InputAlias("restricted"): _tree("test/r", access=Access.RESTRICTED, access_url="https://r.example"),
        InputAlias("also"): _tree("test/also", access=Access.RESTRICTED, access_url="https://also.example"),
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
    assert child.parent_cards[InputAlias("also")].metadata.access_url == "https://also.example"


def test_a_grandchild_records_every_ancestor_in_its_lineage(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        child.import_from(reader, records=["rec"])
        child_dir = store(tmp_path, child)
    with TimeFReader(open_local_version(child_dir)) as reader:
        grandchild = _child(reader, dataset_id="test/grandchild", aliases=(InputAlias("child"),))
        grandchild.import_from(reader, records=[reader.get(Record, "rec", alias=_ALIAS).id])
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
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        child.import_from(reader, records=["rec"])
        del child.parent_cards[_ALIAS]
        with pytest.raises(TimeFValidationError, match=r"input aliases \['base'\] without a parent card"):
            store(tmp_path, child)


def _without_rec(child, imported):
    """Remove the copy of ``rec``, carrying the import map and the parent cards through the edit by hand."""
    edited = remove_records(child, [imported.records["rec"].id])
    edited._imports = dict(child._imports)
    edited.parent_cards = dict(child.parent_cards)
    return edited


def test_an_import_copies_again_what_the_child_no_longer_holds(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=["rec", "other"])
        edited = _without_rec(child, imported)
        (kept,) = edited.records
        again = edited.import_from(reader, records=["rec", "other"])
        edited_dir = store(tmp_path, edited)
    assert again.records["other"] is kept
    assert edited.records == (kept, again.records["rec"])
    assert again.records["rec"] is not imported.records["rec"]
    assert {signal.name for signal in again.records["rec"].signals} == {"sig-a", "sig-b"}
    with TimeFReader(open_local_version(edited_dir)) as stored:
        assert stored.get(Record, "rec", alias=_ALIAS).metadata == {"site": "north"}


def test_the_writer_stores_import_rows_only_for_copies_the_child_holds(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=["rec", "other"])
        edited_dir = store(tmp_path, _without_rec(child, imported))
    assert _rows(edited_dir) == [("Record", "other"), ("Signal", "sig-c")]
    with TimeFReader(open_local_version(edited_dir)) as stored, pytest.raises(TimeFNotFoundError):
        stored.get(Record, "rec", alias=_ALIAS)


def test_the_writer_ignores_stale_entries_of_an_alias_the_child_dropped(tmp_path):
    with _reader(tmp_path, _tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, records=["rec"])
        edited = remove_records(child, [imported.records["rec"].id])
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
        merged = _child(*readers, dataset_id="test/merged", aliases=(InputAlias("left"), InputAlias("right")))
        for reader, (_, identity) in zip(readers, children, strict=True):
            merged.import_from(reader, records=[identity])
        assert len({record.id for record in merged.records}) == 2
        merged.records[0].metadata["branch"] = "edited"
        assert merged.records[1].metadata["branch"] == "right"
        merged_path = store(tmp_path, merged)
    finally:
        for reader in readers:
            reader.close()
    version = open_local_version(merged_path)
    assert not version.manifest.files.time_series
    assert {source.dataset_id for source in version.manifest.payload_sources.values()} == {original.metadata.dataset_id}
    for path in [base_path, *(path for path, _ in children)]:
        (path / "control.duckdb").rename(path / "unavailable.duckdb")
    with TimeFReader(version) as reader:
        for record in reader.read().records:
            assert record.signals[0].to_arrow().equals(original.records[0].signals[0].to_arrow())
