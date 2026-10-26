from contextlib import contextmanager
from dataclasses import fields, replace

import numpy as np
import pytest

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset, composition
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
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    ForecastingTask,
    InputModality,
    License,
    LocalizationMode,
    ScalarPredictionTask,
    Split,
    StepInterval,
    StepPoint,
    StepSpan,
    Task,
    TemporalLocalizationTask,
    TextHandle,
    TimeInterval,
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
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.types.tasks import TASKS
from timenet.types.text import TextListHandle
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
        with pytest.raises(TimeFNotFoundError, match="missing"):
            child.import_from(reader, records=["rec"], tasks=["missing"])
    assert (child.records, child.parent_cards, child.lineage, child._imports) == ((), {}, {}, {})


@pytest.mark.parametrize("selection", ["records", "signals", "tasks"])
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


def _annotation(annotated, key):
    return next(annotation for annotation in annotated.annotations if annotation.key == key)


def _tasks_tree():
    """Return ``_tree()`` with tasks over it, two candidate Records, and a Record of edited Signals.

    ``label`` reads ``rec`` and its ``age``. ``question`` derives from ``label``, scopes ``sig-a``,
    and references ``sig-a``'s ``artifact``, ``monitor``'s ``device``, and the dataset's
    ``release-note``; ``locate`` derives from ``question``. ``match`` picks ``pool-1`` from two
    candidates, ``edit`` targets ``sig-e1`` of ``edited``, and ``pooled`` has no candidates, which
    ``from-pooled`` derives from. ``second`` references ``age``, ``late``, and ``release-note``. The
    dataset's ``unused-note`` is referenced by no task.
    """
    dataset = _tree()
    rec, other = dataset.records
    (root,) = rec.sources
    signal_a = next(signal for signal in rec.signals if signal.name == "sig-a")
    pools = [
        dataset.add_record(
            record=Record(
                record_id=f"pool-{index}", sources=(Source(name="monitor", signals=(_signal(f"sig-p{index}"),)),)
            )
        )
        for index in (1, 2)
    ]
    edited = dataset.add_record(
        record=Record(
            record_id="edited", sources=(Source(name="monitor", signals=(_signal("sig-e1"), _signal("sig-e2"))),)
        )
    )
    (release_note,) = dataset.annotations
    dataset.annotate(Annotation(key="unused-note", value="not referenced"))
    label = ClassificationTask(
        id="label",
        inputs=(rec,),
        targets=("normal",),
        target_schema="scp",
        split=Split.TRAIN,
        input_annotations=(_annotation(rec, "age"),),
        metadata={"rater": "a"},
    )
    label.annotate(Annotation(key="quality", value="good"))
    question = AnswerTask(
        id="question",
        inputs=(rec,),
        prompt="Is lead I clean?",
        rationale="No artifact after 0.25 s.",
        targets=("yes",),
        scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=("sig-a",)),
        input_annotations=(_annotation(signal_a, "artifact"), _annotation(root, "device"), release_note),
        from_tasks=(label,),
    )
    locate = TemporalLocalizationTask(
        id="locate",
        inputs=(rec,),
        targets=(TimeInterval.seconds(2.0, 2.5, time_series_ids=("sig-b",)),),
        mode=LocalizationMode.EXHAUSTIVE,
        from_tasks=(question,),
    )
    pooled = TSCorrespondenceTask(id="pooled", inputs=(other,), targets=("none",))
    dataset.add_tasks(
        tasks=[
            label,
            question,
            locate,
            TSCorrespondenceTask(id="match", inputs=(other,), candidate_records=tuple(pools), targets=(pools[0],)),
            TSEditingTask(id="edit", inputs=(other,), targets=(edited.signals[0], "denoised")),
            pooled,
            AnswerTask(id="from-pooled", inputs=(other,), targets=("x",), from_tasks=(pooled,)),
            AnswerTask(
                id="second",
                inputs=(rec,),
                targets=("no",),
                input_annotations=(_annotation(rec, "age"), _annotation(rec, "late"), release_note),
            ),
        ]
    )
    return dataset


def _task_imports(dataset):
    return {key: value for key, value in dataset._imports.items() if key[1] == "Task"}


def test_a_task_import_brings_its_inputs_whole_and_the_annotations_it_references(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=["question"])
        parent_keys = _all_keys(reader.get(Record, "rec"))
        (parent_note, _) = reader.read().annotations
    (question,) = imported.tasks.values()
    (record,) = child.records
    (root,) = record.sources
    signal_a = next(signal for signal in record.signals if signal.name == "sig-a")
    assert list(imported.tasks) == ["question"]
    assert imported.records == imported.signals == {}
    assert isinstance(question, AnswerTask)
    assert question.id != "question"
    assert question.inputs == (record,)
    assert sorted(signal.name for signal in record.signals) == ["sig-a", "sig-b"]
    assert _all_keys(record) == parent_keys
    # Each reference is the copy on its owner's copy, and the dataset's own annotation is the child's.
    artifact, device, note = question.input_annotations
    assert artifact is _annotation(signal_a, "artifact")
    assert device is _annotation(root, "device")
    assert child.annotations == (note,)
    assert note.value == "first release"
    assert note.occurrence_id != parent_note.occurrence_id
    assert question.scope == TimeInterval.seconds(0.0, 0.5, time_series_ids=(signal_a.id,))
    assert (question.prompt, question.rationale, question.targets) == (
        "Is lead I clean?",
        "No artifact after 0.25 s.",
        ("yes",),
    )
    assert record.task_ids == tuple(task.id for task in child.tasks)


def test_a_task_import_brings_the_tasks_it_derives_from_transitively(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=["locate"])
        (parent_quality,) = reader.get(Task, "label").annotations
    (locate,) = imported.tasks.values()
    (question,) = locate.from_tasks
    (label,) = question.from_tasks
    assert child.tasks == (locate, question, label)
    assert (type(locate), type(question), type(label)) == (TemporalLocalizationTask, AnswerTask, ClassificationTask)
    (record,) = child.records
    assert locate.inputs == question.inputs == label.inputs == (record,)
    signal_b = next(signal for signal in record.signals if signal.name == "sig-b")
    assert locate.targets == (TimeInterval.seconds(2.0, 2.5, time_series_ids=(signal_b.id,)),)
    # Every field comes, including those of a task class's own.
    assert locate.mode is LocalizationMode.EXHAUSTIVE
    assert (label.target_schema, label.split, label.metadata, label.targets) == (
        "scp",
        Split.TRAIN,
        {"rater": "a"},
        ("normal",),
    )
    assert label.input_annotations == (_annotation(record, "age"),)
    (quality,) = label.annotations
    assert quality.value == "good"
    assert quality.occurrence_id is not None
    assert quality.occurrence_id != parent_quality.occurrence_id


def test_a_task_import_brings_candidate_records_and_record_targets_whole(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        (match,) = child.import_from(reader, tasks=["match"]).tasks.values()
        copies = {
            name: child.import_from(reader, records=[name]).records[name] for name in ("other", "pool-1", "pool-2")
        }
    assert child.records == tuple(copies.values())
    assert match.inputs == (copies["other"],)
    assert match.candidate_records == (copies["pool-1"], copies["pool-2"])
    assert match.targets == (copies["pool-1"],)


def test_a_signal_target_brings_a_partial_record(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        (edit,) = child.import_from(reader, tasks=["edit"]).tasks.values()
        other = child.import_from(reader, records=["other"]).records["other"]
        signal = child.import_from(reader, signals=["sig-e1"]).signals["sig-e1"]
    (edited,) = (record for record in child.records if record is not other)
    assert edit.inputs == (other,)
    assert edit.targets == (signal, "denoised")
    assert edited.signals == (signal,)


def test_only_the_dataset_annotations_a_task_references_come_along(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        child.import_from(reader, records=["rec"], tasks=["label"])
        assert child.annotations == ()
        child.import_from(reader, tasks=["question"])
        (note,) = child.annotations
        child.import_from(reader, tasks=["locate"], records=["other"])
    assert child.annotations == (note,)
    assert note.key == "release-note"


def test_a_repeated_task_import_reuses_the_copies(tmp_path, monkeypatch):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        first = child.import_from(reader, tasks=["question"])
        (record,) = child.records
        tasks, annotations, hierarchy = child.tasks, child.annotations, _all_keys(record)

        def unexpected_read(*args):
            pytest.fail("an import must not read a task the child holds a copy of")

        with monkeypatch.context() as patch:
            patch.setattr(DuckDBControlReader, "read_tasks_by_id", unexpected_read)
            again = child.import_from(reader, tasks=["question", "label"])
        assert again.tasks["question"] is first.tasks["question"]
        assert again.tasks["label"] is first.tasks["question"].from_tasks[0]
        assert (child.tasks, child.annotations, _all_keys(record)) == (tasks, annotations, hierarchy)
        (locate,) = child.import_from(reader, tasks=["locate"]).tasks.values()
    assert locate.from_tasks == (first.tasks["question"],)
    assert child.tasks == (*tasks, locate)
    assert child.records == (record,)
    assert set(_task_imports(child)) == {(_ALIAS, "Task", identity) for identity in ("label", "question", "locate")}


def test_a_task_import_after_a_record_import_reuses_the_copies(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        child.import_from(reader, signals=["sig-a"])
        record = child.import_from(reader, records=["rec"]).records["rec"]
        record.metadata["site"] = "edited"
        hierarchy = [
            (container, container.annotations) for container in (record, *record.walk_sources(), *record.signals)
        ]
        (question,) = child.import_from(reader, tasks=["question"]).tasks.values()
    assert child.records == (record,)
    assert record.metadata == {"site": "edited"}
    assert [
        (container, container.annotations) for container in (record, *record.walk_sources(), *record.signals)
    ] == hierarchy
    (root,) = record.sources
    signal_a = next(signal for signal in record.signals if signal.name == "sig-a")
    assert question.inputs == question.from_tasks[0].inputs == (record,)
    assert question.input_annotations[:2] == (_annotation(signal_a, "artifact"), _annotation(root, "device"))
    assert question.from_tasks[0].input_annotations[0] is _annotation(record, "age")


def test_an_annotation_the_child_removed_comes_back_for_a_task_that_references_it(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        # A connector may replace the copied annotations with its own.
        record.annotations = ()
        record.annotate(Annotation(key="note", value="mine"))
        (label,) = child.import_from(reader, tasks=["label"]).tasks.values()
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
    parent.add_task(task=AnswerTask(id="mi", inputs=(rec,), targets=("x",), input_annotations=(labels[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        record.annotations = tuple(annotation for annotation in record.annotations if annotation.value != "MI")
        (task,) = child.import_from(reader, tasks=["mi"]).tasks.values()
        store(tmp_path, child)
    (mi,) = task.input_annotations
    assert mi is record.annotations[-1]
    assert [annotation.value for annotation in record.annotations if annotation.key == "label"] == ["NORM", "MI"]


def test_an_annotation_without_an_occurrence_id_is_no_copy_for_a_task(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        # Holds the content of the parent's age, attached without annotate(), so no task can reference it.
        record.annotations = (Annotation(key="age", value=64),)
        (label,) = child.import_from(reader, tasks=["label"]).tasks.values()
    (age,) = label.input_annotations
    assert age.occurrence_id is not None
    assert age is record.annotations[-1]


def _edit_age(record):
    record.annotations = tuple(
        replace(annotation, value=65) if annotation.key == "age" else annotation for annotation in record.annotations
    )


def _replace_age(record):
    record.annotations = tuple(annotation for annotation in record.annotations if annotation.key != "age")
    record.annotate(Annotation(key="age", value=65))


@pytest.mark.parametrize("change", [_edit_age, _replace_age], ids=["edited", "replaced"])
def test_a_task_that_references_an_annotation_the_child_changed_gets_a_copy_of_the_parents(tmp_path, change):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        # The child changes age from 64 to 65.
        change(record)
        (label,) = child.import_from(reader, tasks=["label"]).tasks.values()
        child_dir = store(tmp_path, child)
    # The Record holds both, and the task references the parent's 64.
    assert sorted(annotation.value for annotation in record.annotations if annotation.key == "age") == [64, 65]
    (age,) = label.input_annotations
    assert (age.value, age is record.annotations[-1]) == (64, True)
    occurrences = [annotation.occurrence_id for annotation in record.annotations]
    assert len(occurrences) == len(set(occurrences))
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert [annotation.value for annotation in stored.get(Task, "label", alias=_ALIAS).input_annotations] == [64]


def test_task_copies_that_fail_to_register_leave_no_task_and_no_map_entry(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        clash = _child(reader).import_from(reader, tasks=["question"]).tasks["question"].id
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        (root,) = record.sources
        signal_a = next(signal for signal in record.signals if signal.name == "sig-a")
        # Every reference of question, label, and second attaches a copy again: to a Signal, a
        # Source, the dataset, and two to the Record.
        record.annotations = root.annotations = ()
        object.__setattr__(signal_a, "annotations", ())
        mine = child.add_record(record=Record(record_id="mine"))
        # The child's own task holds the ID the copy of question gets.
        own = child.add_task(task=AnswerTask(id=clash, inputs=(mine,), targets=("own",)))
        imports = dict(child._imports)
        with pytest.raises(TimeFValidationError, match="already registered"):
            child.import_from(reader, tasks=["question", "second"], records=["other"])
        assert child.tasks == (own,)
        assert record.task_ids == ()
        assert (record.annotations, root.annotations, signal_a.annotations, child.annotations) == ((), (), (), ())
        # The Records stay imported, and no task is in the map.
        assert [copy.id for copy in child.records] == [record.id, "mine", child._imports[_ALIAS, "Record", "other"]]
        assert child._imports.keys() - imports.keys() == {(_ALIAS, "Record", "other"), (_ALIAS, "Signal", "sig-c")}
        assert _task_imports(child) == {}
        # Once the clash is gone, the same import completes.
        edited = remove_records(child, ["mine"], cascade=True)
        edited._imports, edited.parent_cards = dict(child._imports), dict(child.parent_cards)
        question, _ = edited.import_from(reader, tasks=["question", "second"]).tasks.values()
    assert question.id == clash
    assert len(edited.tasks) == 3
    assert set(_task_imports(edited)) == {(_ALIAS, "Task", identity) for identity in ("label", "question", "second")}


def test_a_correspondence_task_drawing_on_the_whole_parent_is_rejected(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        with pytest.raises(TimeFValidationError, match=r"'pooled'.*no candidate_records.*explicit candidate_records"):
            child.import_from(reader, records=["rec"], tasks=["match", "pooled"])
        # It holds for a task that a selected task derives from, too.
        with pytest.raises(TimeFValidationError, match=r"'pooled'.*no candidate_records"):
            child.import_from(reader, tasks=["from-pooled"])
    assert (child.records, child.tasks, child.parent_cards, child._imports) == ((), (), {}, {})


def test_a_child_that_streams_its_tasks_cannot_import_tasks(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        child.set_task_stream([AnswerTask], lambda: iter(()))
        with pytest.raises(TimeFValidationError, match="streams its tasks"):
            child.import_from(reader, records=["rec"], tasks=["question"])
        assert (child.records, child.parent_cards, child._imports) == ((), {}, {})
        child.import_from(reader, records=["rec"])
    assert len(child.records) == 1


def test_a_task_import_reads_only_the_tasks_and_the_records_it_needs(tmp_path, monkeypatch):
    batches = []
    read_records = DuckDBControlReader.read_records

    def counting(self, record_ids=None, **kwargs):
        requested = None if record_ids is None else tuple(record_ids)
        batches.append(requested)
        return read_records(self, requested, **kwargs)

    def unexpected_read(*args, **kwargs):
        pytest.fail("an import must not read the whole parent")

    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "read", unexpected_read)
        monkeypatch.setattr(DuckDBControlReader, "read_records", counting)
        child.import_from(reader, records=["rec", "pool-2"], tasks=["locate", "edit"])
    # The task read builds the Records the tasks reference, which the import uses, and the Record
    # of the Signal target, which the import reads again with the other Records: a Signal names
    # no Record.
    assert batches == [("rec",), ("other",), ("edited",), ("pool-2", "edited")]


def test_a_written_child_reads_its_task_copies_by_parent_id(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=["locate", "edit"])
        child_dir = store(tmp_path, child)
    assert [row for row in _rows(child_dir) if row[0] == "Task"] == [
        ("Task", "edit"),
        ("Task", "label"),
        ("Task", "locate"),
        ("Task", "question"),
    ]
    with TimeFReader(open_local_version(child_dir)) as stored:
        question = stored.get(AnswerTask, "question", alias=_ALIAS)
        assert question.prompt == "Is lead I clean?"
        assert [annotation.key for annotation in question.input_annotations] == ["artifact", "device", "release-note"]
        assert question.from_tasks[0].id == imported.tasks["locate"].from_tasks[0].from_tasks[0].id
        targets = stored.get(Task, "edit", alias=_ALIAS).targets
        assert targets is not None
        signal, _ = targets
        assert isinstance(signal, Signal)
        assert signal.to_numpy().tolist() == list(range(16))
        with pytest.raises(TimeFNotFoundError, match="the stored object is a TemporalLocalizationTask"):
            stored.get(AnswerTask, "locate", alias=_ALIAS)


def _annotated_elsewhere():
    """Return a parent whose tasks read Record ``rec`` and reference annotations on Record ``far``.

    ``far`` holds the Source ``near`` with ``sig-n`` (0 to 1 s), the Source ``tail`` with ``sig-l``
    (2 to 3 s), and the Signal-less Source ``spare``. Each task references one annotation: one on
    ``far`` itself without a span (``static``) and one at 2.5 s (``late``), one on ``near`` without a
    span (``near-note``), one on ``sig-l`` (``sig-note``), and one on ``spare`` (``spare-note``).
    """
    parent = TimeFDataset(metadata=_metadata("test/far"))
    rec = parent.add_record(record=Record(record_id="rec", sources=(Source(name="m", signals=(_signal("sig-r"),)),)))
    near = Source(id="src-near", name="near", signals=(_signal("sig-n"),))
    tail = Source(id="src-tail", name="tail", signals=(_signal("sig-l", offset_s=2),))
    spare = Source(id="src-spare", name="spare")
    far = parent.add_record(record=Record(record_id="far", sources=(near, tail, spare)))
    annotations = {
        "static": far.annotate(Annotation(key="static", value="far")),
        "late": far.annotate(Annotation(key="late", span=TimePoint.seconds(2.5))),
        "near": near.annotate(Annotation(key="near-note", value="near")),
        "sig": tail.signals[0].annotate(Annotation(key="sig-note", value="tail")),
        "spare": spare.annotate(Annotation(key="spare-note", value="spare")),
    }
    parent.add_tasks(
        tasks=[
            AnswerTask(id=name, inputs=(rec,), targets=("x",), input_annotations=(annotation,))
            for name, annotation in annotations.items()
        ]
    )
    return parent


def _holding(record):
    return sorted(signal.name for signal in record.signals), sorted(source.name for source in record.walk_sources())


@pytest.mark.parametrize(
    ("task", "held"),
    [
        ("static", (["sig-n"], ["near"])),
        ("late", (["sig-l"], ["tail"])),
        ("near", (["sig-n"], ["near"])),
        ("sig", (["sig-l"], ["tail"])),
        ("spare", (["sig-l", "sig-n"], ["near", "spare", "tail"])),
    ],
)
def test_an_annotation_a_task_references_on_another_record_brings_a_partial_copy_of_it(tmp_path, task, held):
    with _reader(tmp_path, _annotated_elsewhere()) as reader:
        child = _child(reader)
        (copy,) = child.import_from(reader, tasks=[task]).tasks.values()
        far = child._records_by_id[child._imports[_ALIAS, "Record", "far"]]
        child_dir = store(tmp_path, child)
    # The copy of far holds what places the annotation, and only that, unless only the whole
    # Record holds the annotation's owner.
    assert _holding(far) == held
    (annotation,) = copy.input_annotations
    assert any(annotation is held_annotation for held_annotation in far.walk_annotations())
    assert [record.id for record in copy.inputs] != [far.id]
    with TimeFReader(open_local_version(child_dir)) as stored:
        (written,) = stored.get(AnswerTask, task, alias=_ALIAS).input_annotations
        assert (written.key, written.occurrence_id) == (annotation.key, annotation.occurrence_id)


def test_annotations_on_another_record_share_one_copy_that_a_later_import_completes(tmp_path):
    with _reader(tmp_path, _annotated_elsewhere()) as reader:
        child = _child(reader)
        tasks = child.import_from(reader, tasks=["static", "late", "near"]).tasks
        far = child._records_by_id[child._imports[_ALIAS, "Record", "far"]]
        assert _holding(far) == (["sig-l", "sig-n"], ["near", "tail"])
        child.import_from(reader, records=["far"])
    # The whole Record adds spare and its annotation, and attaches nothing twice.
    assert _holding(far) == (["sig-l", "sig-n"], ["near", "spare", "tail"])
    assert _all_keys(far) == [
        ("record", ["late", "static"]),
        ("near", ["near-note"]),
        ("spare", ["spare-note"]),
        ("tail", []),
    ]
    assert [task.input_annotations[0].key for task in tasks.values()] == ["static", "late", "near-note"]


def test_task_copies_reuse_the_parent_payloads_of_packed_text(tmp_path, monkeypatch):
    text = "A long clinical question. " * 300
    parent = _tree()
    parent.add_task(task=AnswerTask(id="long", inputs=parent.records[:1], prompt=text, rationale=text, targets=(text,)))
    parent_dir = store(tmp_path, parent)

    def unexpected_read(*args):
        pytest.fail("a task import must not read or copy stored text")

    with TimeFReader(open_local_version(parent_dir)) as reader, monkeypatch.context() as patch:
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        child = _child(reader)
        child.import_from(reader, tasks=["long"])
        child_dir = store(tmp_path, child)
    assert sorted(path.name for path in child_dir.iterdir()) == ["control.duckdb", "manifest.json"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        task = stored.get(AnswerTask, "long", alias=_ALIAS)
        assert isinstance(task.prompt, StoredText)
        assert task.prompt._location.owner == "test/tree@1.0.0"
        assert (task.prompt, task.rationale, task.targets) == (text, text, (text,))


@pytest.mark.parametrize("first", [{}, {"records": ["rec"]}], ids=["same-call", "after-records"])
def test_identical_occurrences_a_task_references_stay_distinct(tmp_path, first):
    parent = _tree()
    rec = parent.records[0]
    # Two raters gave the same label: one content, two occurrences.
    beats = (rec.annotate(Annotation(key="beat", value="N")), rec.annotate(Annotation(key="beat", value="N")))
    parent.add_task(task=AnswerTask(id="both", inputs=(rec,), targets=("N",), input_annotations=beats))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, **first)
        (task,) = child.import_from(reader, tasks=["both"]).tasks.values()
    (record,) = child.records
    copies = [annotation for annotation in record.annotations if annotation.key == "beat"]
    # The parent's reader may return the two in either order, so only their distinctness shows.
    assert len(copies) == 2
    first_beat, second_beat = task.input_annotations
    assert first_beat is not second_beat
    assert {id(first_beat), id(second_beat)} == {id(copy) for copy in copies}


@pytest.mark.parametrize("first", [{}, {"records": ["rec"]}], ids=["same-call", "after-records"])
def test_tasks_of_one_call_that_reference_one_annotation_share_its_copy(tmp_path, first):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        child.import_from(reader, **first)
        imported = child.import_from(reader, tasks=["label", "question", "second"])
        child_dir = store(tmp_path, child)
    label, question, second = imported.tasks.values()
    (record,) = child.records
    assert second.input_annotations[0] is label.input_annotations[0] is _annotation(record, "age")
    assert second.input_annotations[2] is question.input_annotations[2]
    assert [annotation.key for annotation in record.annotations].count("age") == 1
    assert [annotation.key for annotation in child.annotations] == ["release-note"]
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(AnswerTask, "second", alias=_ALIAS)
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
    parent.add_task(task=AnswerTask(id="flagged", inputs=(rec,), targets=("x",), input_annotations=(flags[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        # Put the copy of the other flag first, so that only the field tells the two apart.
        record.annotations = tuple(
            sorted(record.annotations, key=lambda annotation: getattr(annotation, name) == values[1])
        )
        wrong, right = (annotation for annotation in record.annotations if annotation.key == "flag")
        (task,) = child.import_from(reader, tasks=["flagged"]).tasks.values()
    (flag,) = task.input_annotations
    assert getattr(wrong, name) == values[0]
    assert flag is right
    assert [annotation for annotation in record.annotations if annotation.key == "flag"] == [wrong, right]


def _annotated_record():
    """Return Record ``rec`` with ``sig-a``, ``sig-b``, and the ordinal ``sig-s``, and annotations at every level."""
    signal_a = _signal("sig-a")
    ordinal = Signal.from_values(
        np.arange(8, dtype=np.float32), spec=_SPEC, name="sig-s", time_axis=OrdinalAxis(), id="sig-s"
    )
    aux = Source(name="aux")
    root = Source(
        name="monitor",
        signals=(_signal("sig-b", offset_s=2), ordinal),
        sources=(Source(name="leads", signals=(signal_a,)), aux),
    )
    record = Record(record_id="rec", sources=(root,))
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
    """
    dataset = TimeFDataset(metadata=_metadata("test/tree"))
    rec = dataset.add_record(record=_annotated_record())
    sources = {source.name: source for source in rec.walk_sources()}
    signal_a = next(signal for signal in rec.signals if signal.name == "sig-a")
    other, pool_1, pool_2, far = (
        dataset.add_record(
            record=Record(record_id=name, sources=(Source(name="m", signals=tuple(map(_signal, signals))),))
        )
        for name, signals in (
            ("other", ["sig-c"]),
            ("pool-1", ["sig-p1"]),
            ("pool-2", ["sig-p2"]),
            ("far", ["sig-f1", "sig-f2"]),
        )
    )
    note = dataset.annotate(Annotation(key="release-note", value="first release"))
    artifact = _annotation(signal_a, "artifact")
    classify = ClassificationTask(
        id="cls",
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
        id="ans",
        inputs=(rec,),
        prompt="Is lead I clean?",
        rationale="No artifact after 0.25 s.",
        targets=("yes",),
        scope=TimeInterval.seconds(0.0, 0.5, time_series_ids=("sig-a",)),
        input_annotations=(artifact, _annotation(sources["monitor"], "device")),
        from_tasks=(classify,),
        split=Split.TEST,
    )
    generate = TSGenerationTask(
        id="gen",
        prompt="make a sine",
        targets=("ok",),
        input_modalities=frozenset({InputModality.NO_INPUT, InputModality.TEXT}),
        input_annotations=(note,),
    )
    dataset.add_tasks(
        tasks=[
            classify,
            answer,
            ScalarPredictionTask(
                id="scalar",
                inputs=(rec,),
                targets=(64.5, 3),
                unit="mV",
                target_name="amp",
                from_tasks=(answer, classify),
            ),
            TemporalLocalizationTask(
                id="loc",
                inputs=(rec,),
                targets=(
                    TimeInterval.seconds(2.0, 2.5, time_series_ids=("sig-b",)),
                    StepPoint(time_series_id="sig-s", start=3),
                ),
                scope=StepInterval(time_series_id="sig-s", start=0, stop=6),
                mode=LocalizationMode.EXHAUSTIVE,
            ),
            TemporalLocalizationTask(
                id="loc-ann", inputs=(rec,), target_annotations=(artifact, _annotation(rec, "late"))
            ),
            ForecastingTask(id="fc", inputs=(rec,), targets=(far.signals[0],)),
            TSEditingTask(id="edit", inputs=(other,), targets=(signal_a, "denoised", True)),
            generate,
            TSCorrespondenceTask(
                id="corr",
                inputs=(other,),
                candidate_records=(pool_1, pool_2),
                targets=(pool_2,),
                from_tasks=(generate,),
            ),
        ]
    )
    return dataset


_EVERY_TASK = ("cls", "ans", "scalar", "loc", "loc-ann", "fc", "edit", "gen", "corr")


def _mapped_span(span, signal_id):
    if isinstance(span, StepSpan):
        return replace(span, time_series_id=signal_id(span.time_series_id))
    if isinstance(span, TimeSpan) and span.time_series_ids is not None:
        return replace(span, time_series_ids=tuple(map(signal_id, span.time_series_ids)))
    return span


def _portrait(task, *, record_id=str, signal_id=str, task_id=str):
    """Return every field of a task, with each reference as its kind and mapped ID and each text read.

    An annotation shows where it lies: on the task, on the task's inputs, or on the dataset.
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

    def portrait(value, name):
        if isinstance(value, tuple):
            return tuple(portrait(item, name) for item in value)
        if isinstance(value, Annotation):
            owner = ("Task",) if name == "annotations" else owners.get(id(value), ("Dataset",))
            fixed = ("key", "span", "unit", "description", "source", "confidence", "metadata", "occurrence_metadata")
            return owner, text(value.value), tuple(getattr(value, field) for field in fixed)
        if isinstance(value, StepSpan | TimeSpan):
            return _mapped_span(value, signal_id)
        for kind, mapped in ((Record, record_id), (Signal, signal_id), (Task, task_id)):
            if isinstance(value, kind):
                return kind.__name__, mapped(value.id), value.to_numpy().tolist() if isinstance(value, Signal) else None
        return text(value)

    return {
        field.name: portrait(getattr(task, field.name), field.name) for field in fields(task) if field.name != "id"
    } | {
        "id": task_id(task.id),
        "class": type(task),
    }


def test_every_task_class_round_trips_through_a_written_child(tmp_path):
    with _reader(tmp_path, _every_task_class()) as reader:
        child = _child(reader)
        imported = child.import_from(reader, tasks=_EVERY_TASK)
        imports = {(kind, identity): copy for (_, kind, identity), copy in child._imports.items()}

        def mapped(kind):
            return lambda identity: imports[kind, identity]

        expected = [
            _portrait(task, record_id=mapped("Record"), signal_id=mapped("Signal"), task_id=mapped("Task"))
            for task in reader.get_many(Task, _EVERY_TASK)
        ]
        assert [_portrait(task) for task in imported.tasks.values()] == expected
        child_dir = store(tmp_path, child)
    assert {type(task) for task in imported.tasks.values()} == set(TASKS.values())
    with TimeFReader(open_local_version(child_dir)) as stored:
        assert [_portrait(task) for task in stored.get_many(Task, _EVERY_TASK, alias=_ALIAS)] == expected


def test_a_task_field_that_holds_references_outside_a_tuple_is_rejected(tmp_path, monkeypatch):
    get_many = TimeFReader.get_many

    def with_a_list(self, kind, ids, **kwargs):
        found = get_many(self, kind, ids, **kwargs)
        for task in found if kind is Task else ():
            task.metadata = {"inputs": list(task.inputs)}
        return found

    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        monkeypatch.setattr(TimeFReader, "get_many", with_a_list)
        with pytest.raises(
            TimeFValidationError, match=r"ClassificationTask\.metadata holds a dict that references objects"
        ):
            child.import_from(reader, tasks=["label"])
    assert (child.tasks, _task_imports(child)) == ((), {})


def test_a_dataset_annotation_of_the_childs_own_with_the_key_of_a_referenced_one_stays(tmp_path):
    with _reader(tmp_path, _tasks_tree()) as reader:
        child = _child(reader)
        own = child.annotate(Annotation(key="release-note", value="the child's own"))
        (question,) = child.import_from(reader, tasks=["question"]).tasks.values()
    own_note, note = child.annotations
    assert own_note is own
    assert question.input_annotations[2] is note
    assert note.value == "first release"


def test_an_annotation_the_child_edited_into_another_of_its_key_gets_a_copy_of_the_parents(tmp_path):
    parent = _tree()
    rec = parent.records[0]
    labels = [rec.annotate(Annotation(key="label", value=value)) for value in ("NORM", "MI")]
    parent.add_task(task=AnswerTask(id="mi", inputs=(rec,), targets=("x",), input_annotations=(labels[1],)))
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        # Now two copies hold NORM, and the edited one keeps the occurrence ID of the MI copy.
        record.annotations = tuple(
            replace(annotation, value="NORM") if annotation.value == "MI" else annotation
            for annotation in record.annotations
        )
        (task,) = child.import_from(reader, tasks=["mi"]).tasks.values()
        store(tmp_path, child)
    (mi,) = task.input_annotations
    assert (mi.value, mi is record.annotations[-1]) == ("MI", True)
    assert [annotation.value for annotation in record.annotations if annotation.key == "label"] == [
        "NORM",
        "NORM",
        "MI",
    ]


def _beats_parent():
    """Return ``_tree()`` whose ``rec`` holds two identical ``beat`` occurrences, both referenced by ``both``."""
    parent = _tree()
    rec = parent.records[0]
    beats = (rec.annotate(Annotation(key="beat", value="N")), rec.annotate(Annotation(key="beat", value="N")))
    parent.add_task(task=AnswerTask(id="both", inputs=(rec,), targets=("N",), input_annotations=beats))
    return parent


@pytest.mark.parametrize("removed", [0, 1])
def test_a_copy_attached_again_beside_an_identical_one_gets_an_occurrence_id_of_its_own(tmp_path, removed):
    with _reader(tmp_path, _beats_parent()) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        kept = [annotation for annotation in record.annotations if annotation.key == "beat"]
        # Either copy may be the one the first reference takes, so each case runs once.
        record.annotations = tuple(annotation for annotation in record.annotations if annotation is not kept[removed])
        (task,) = child.import_from(reader, tasks=["both"]).tasks.values()
        child_dir = store(tmp_path, child)
    first, second = task.input_annotations
    assert first is not second
    assert {first.occurrence_id, second.occurrence_id} <= {
        annotation.occurrence_id for annotation in record.annotations
    }
    occurrence_ids = [annotation.occurrence_id for annotation in record.walk_annotations()]
    assert len(occurrence_ids) == len(set(occurrence_ids))
    with TimeFReader(open_local_version(child_dir)) as stored:
        written = stored.get(AnswerTask, "both", alias=_ALIAS)
        assert len({annotation.occurrence_id for annotation in written.input_annotations}) == 2


def test_identical_dataset_annotations_referenced_across_calls_keep_distinct_occurrence_ids(tmp_path):
    parent = _tree()
    rec = parent.records[0]
    first, second = (parent.annotate(Annotation(key="note", value="same")) for _ in range(2))
    parent.add_tasks(
        tasks=[
            AnswerTask(id="one", inputs=(rec,), targets=("x",), input_annotations=(first,)),
            AnswerTask(id="both", inputs=(rec,), targets=("x",), input_annotations=(second, first)),
        ]
    )
    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        child.import_from(reader, tasks=["one"])
        (both,) = child.import_from(reader, tasks=["both"]).tasks.values()
        store(tmp_path, child)
    assert len(child.annotations) == 2
    assert {id(annotation) for annotation in both.input_annotations} == {
        id(annotation) for annotation in child.annotations
    }
    assert len({annotation.occurrence_id for annotation in child.annotations}) == 2


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
    parent.add_task(task=AnswerTask(id="many", inputs=(rec,), targets=("x",), input_annotations=references))
    steps = {"_take": 0, "_same_occurrence": 0}

    def counted(name):
        function = getattr(composition, name)

        def counting(*args):
            steps[name] += 1
            return function(*args)

        return counting

    with _reader(tmp_path, parent) as reader:
        child = _child(reader)
        record = child.import_from(reader, records=["rec"]).records["rec"]
        values = {"b", *(f"b{i}" for i in range(count))}
        # The child removes the copy of the second of each pair, which the task then attaches again.
        record.annotations = tuple(
            annotation for annotation in record.annotations if read_value(annotation.value) not in values
        )
        for name in steps:
            monkeypatch.setattr(composition, name, counted(name))
        (task,) = child.import_from(reader, tasks=["many"]).tasks.values()
    assert len(task.input_annotations) == count
    assert all(read_value(annotation.value) in values for annotation in task.input_annotations)
    assert max(steps.values()) <= 4 * count
