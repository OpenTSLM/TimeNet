"""Write a hierarchy, read it back, and check every relationship survived the trip.

Two ids run through every one of these tests. The caller's own name is ``external_id`` and is what a
reader is addressed by; the dense ``INTEGER`` surrogate is ``<entity>_id`` and is what the tables
join on and what :meth:`TimeFReader.values` takes. A test that mixes them up passes for the wrong
reason, so each assertion names which one it means.
"""

from fractions import Fraction
import pickle

import numpy as np
import pytest

from timenet.control_plane import (
    Annotation,
    DeclarativeDataset,
    Record,
    RecordRef,
    Signal,
    Source,
    Task,
    TimeFReader,
    TimeFWriter,
    schema as ddl,
)
from timenet.control_plane.reader import render_task
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.testing import make_dataset, make_metadata, make_record
from timenet.types import TimeSeriesSpec, ureg


@pytest.fixture
def version(tmp_path):
    """Write the fixture dataset and return its committed version directory."""
    dataset = make_dataset(n_records=3, n_values=256)
    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def reader(version):
    """Open the written version for reading."""
    with TimeFReader(version) as opened:
        yield opened


def _signal(record, external_id):
    """Return the signal of one record that was built under this id."""
    found = [signal for signal in record.signals() if signal.external_id == external_id]
    assert found, f"no signal {external_id!r} in record {record.external_id!r}"
    return found[0]


def _values(reader, record_external_id, signal_external_id):
    """Read one signal's values, reaching its surrogate id the way a caller does."""
    return reader.values(_signal(reader.record(record_external_id), signal_external_id).signal_id)


def test_every_record_round_trips(reader):
    assert reader.record_ids() == ["record-000", "record-001", "record-002"]


def test_a_record_carries_both_its_ids(reader):
    record = reader.record("record-000")
    assert record.external_id == "record-000"
    assert isinstance(record.record_id, int)
    assert reader.record("record-002").record_id != record.record_id


def test_every_level_keeps_the_name_it_was_built_under(reader):
    record = reader.record("record-001")
    monitor = record.sources[0]
    assert monitor.external_id == "record-001-monitor"
    assert [child.external_id for child in monitor.sources] == ["record-001-ecg", "record-001-temperature"]
    assert [signal.external_id for signal in record.signals()] == [
        "record-001-lead-i",
        "record-001-lead-ii",
        "record-001-chest",
    ]
    assert reader.task("diagnosis-001").external_id == "diagnosis-001"


def test_surrogate_ids_are_dense_from_zero(reader):
    """The worker partition is ``record_id % n``, which only covers every row while ids are dense."""
    for table, column in ddl.KEYED_TABLES.items():
        row = reader.connection.execute(
            f"SELECT count(*), count(DISTINCT {column}), min({column}), max({column}) FROM {table}"  # noqa: S608
        ).fetchone()
        total, distinct, lowest, highest = row
        assert total > 0, f"{table} is empty, so it proves nothing about {column}"
        assert (distinct, lowest, highest) == (total, 0, total - 1), f"{table}.{column} is not dense"


def test_the_database_holds_one_dataset_named_in_meta(reader):
    """There is no datasets table any more, so the id it used to hold lives in meta."""
    tables = {
        row[0] for row in reader.connection.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    assert tables == set(ddl.TABLES)
    assert dict(reader.connection.execute("SELECT key, value FROM meta").fetchall()) == {
        "schema_version": str(ddl.SCHEMA_VERSION),
        "dataset_id": "test/bedside",
    }


def test_nothing_carries_a_metadata_blob(reader):
    """A blob repeats on every row and cannot be searched. Anything a builder has is an annotation."""
    columns = reader.connection.execute(
        "SELECT table_name FROM information_schema.columns WHERE column_name = 'metadata'"
    ).fetchall()
    assert columns == []
    record = reader.record("record-000")
    for view in (record, *record.walk_sources(), *record.signals(), reader.task("diagnosis-000")):
        assert not hasattr(view, "metadata")


def test_source_tree_keeps_its_shape(reader):
    record = reader.record("record-000")
    assert [source.name for source in record.sources] == ["Bedside monitor"]
    monitor = record.sources[0]
    assert [child.name for child in monitor.sources] == ["ECG", "Temperature sensor"]
    assert monitor.depth == 0
    assert all(child.depth == 1 for child in monitor.sources)


def test_signals_hang_off_the_right_source(reader):
    record = reader.record("record-000")
    ecg, temperature = record.sources[0].sources
    assert [signal.name for signal in ecg.signals] == ["I", "II"]
    assert [signal.name for signal in temperature.signals] == ["Chest temperature"]


def test_signal_resolves_its_spec_and_axis(reader):
    lead_i = _signal(reader.record("record-000"), "record-000-lead-i")
    assert lead_i.spec_type == "ecg-voltage"
    assert lead_i.unit == "millivolt"
    assert lead_i.dtype == "float32"
    assert lead_i.axis_type == "regular"
    assert lead_i.n_values == 256


def test_annotations_land_at_every_level(reader):
    record = reader.record("record-000")
    assert [a.name for a in record.annotations] == ["patient_sex", "patient_age"]
    ecg = record.sources[0].sources[0]
    assert [a.name for a in ecg.annotations] == ["device_model"]
    assert [a.name for a in _signal(record, "record-000-lead-i").annotations] == ["lead_status"]


def test_annotations_for_dispatches_through_the_table_map(reader):
    """One call per target kind, each landing in the table ``ANNOTATION_TABLES`` names."""
    record = reader.record("record-000")
    ecg = record.sources[0].sources[0]
    assert [a.name for a in reader.annotations_for("record", record.record_id)] == ["patient_sex", "patient_age"]
    assert [a.name for a in reader.annotations_for("source", ecg.source_id)] == ["device_model"]
    assert [a.name for a in reader.annotations_for("signal", _signal(record, "record-000-lead-i").signal_id)] == [
        "lead_status"
    ]
    assert [a.name for a in reader.annotations_for("task", reader.task("diagnosis-000").task_id)] == ["task_type"]


def test_point_annotation_keeps_its_placement(reader):
    (status,) = _signal(reader.record("record-000"), "record-000-lead-i").annotations
    assert status.span_type == "point"
    assert status.start_us == 6_000_000
    assert status.to_text() == "lead_status=Lead fell off @6s"


def test_an_annotation_on_three_records_is_one_payload_and_three_attachments(reader):
    """The whole point of storing a payload once: 100,000 records cost one row plus short links."""
    payloads = reader.connection.execute("SELECT count(*) FROM annotations WHERE name = 'patient_sex'").fetchone()[0]
    attachments = reader.connection.execute(
        "SELECT count(*) FROM record_annotations a JOIN annotations n USING (annotation_id) "
        "WHERE n.name = 'patient_sex'"
    ).fetchone()[0]
    assert payloads == 1
    assert attachments == 3
    assert reader.records_with("patient_sex", "male") == ["record-000", "record-001", "record-002"]


def test_values_round_trip_with_their_dtype(reader):
    values = _values(reader, "record-000", "record-000-lead-i")
    assert values.shape == (256,)
    assert values.dtype == np.float32


def test_irregular_signal_round_trips(reader):
    chest = _signal(reader.record("record-000"), "record-000-chest")
    assert chest.axis_type == "irregular"
    assert reader.values(chest.signal_id).shape == (10,)


def test_task_resolves_its_inputs_and_targets(reader):
    task = reader.task("diagnosis-000")
    assert task.prompt == "Diagnose this patient."
    assert len(task.inputs) == 1
    assert task.inputs[0].external_id == "record-000"
    assert task.target == ["The patient is stable."]
    assert [a.name for a in task.annotations] == ["task_type"]


def test_a_task_item_stores_the_record_s_surrogate_id(reader):
    """The caller's name is resolved once, at write time, so the item row joins on an integer."""
    record = reader.record("record-001")
    stored = reader.connection.execute(
        "SELECT ti.record_id FROM task_items ti JOIN tasks t USING (task_id) "
        "WHERE t.external_id = 'diagnosis-001' AND ti.item_type = 'record'"
    ).fetchall()
    assert [row[0] for row in stored] == [record.record_id]
    assert reader.task("diagnosis-001").inputs[0].record_id == record.record_id


def test_a_task_naming_a_record_by_id_resolves_to_that_record(tmp_path):
    """A streaming build has dropped the record object, so it names the record by id instead."""
    dataset = DeclarativeDataset(metadata=make_metadata())
    for index in range(3):
        dataset.add_record(make_record(f"record-{index:03d}", seed=index, n_values=32))
    dataset.add_task(Task(id="by-ref", prompt="Diagnose.", inputs=[RecordRef("record-002")]))

    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with TimeFReader(tmp_path / dataset.metadata.dataset_id / "1.0.0") as reader:
        task = reader.task("by-ref")
        # A task input is either a record or a plain text item; this task's is a record.
        inputs = [item for item in task.inputs if not isinstance(item, str)]
        assert [item.external_id for item in inputs] == ["record-002"]
        assert inputs[0].record_id == reader.record("record-002").record_id
        assert reader.tasks_for_record("record-002") == ["by-ref"]


def test_render_task_walks_the_whole_hierarchy(reader):
    rendered = render_task(reader, "diagnosis-000")
    assert "PROMPT: Diagnose this patient." in rendered
    assert "INPUT record record-000" in rendered
    assert "source: Bedside monitor" in rendered
    assert "signal: I (ecg-voltage" in rendered
    assert "[signal] lead_status=Lead fell off @6s" in rendered
    assert "[record] patient_sex=male" in rendered
    assert "TARGET: The patient is stable." in rendered


def test_reverse_lookup_from_annotation_to_records(reader):
    assert reader.records_with("patient_sex", "male") == ["record-000", "record-001", "record-002"]


def test_reverse_lookup_reaches_records_through_signals(reader):
    # lead_status is attached to a signal, not to the record, and a signal attachment carries no
    # record, so this only resolves if the lookup walks source_signals.
    assert reader.records_with("lead_status") == ["record-000", "record-001", "record-002"]


def test_reverse_lookup_from_annotation_to_objects(reader):
    """One arm per attachment table, each naming the kind its own table makes it.

    A dataset attachment has no target column, because the database holds one dataset, so that arm
    reports 0.
    """
    records = reader.records(reader.record_ids())
    assert reader.objects_with("collection") == [("dataset", 0)]
    assert reader.objects_with("patient_sex", "male") == sorted(("record", r.record_id) for r in records)
    assert reader.objects_with("device_model") == sorted(("source", r.sources[0].sources[0].source_id) for r in records)
    assert reader.objects_with("lead_status") == sorted(
        ("signal", _signal(r, f"{r.external_id}-lead-i").signal_id) for r in records
    )
    assert reader.objects_with("task_type") == sorted(("task", t.task_id) for t in reader.tasks(reader.task_ids()))


def test_tasks_for_record_without_a_link_table(reader):
    assert reader.tasks_for_record("record-001") == ["diagnosis-001"]


def test_subtree_returns_the_branch(reader):
    record = reader.record("record-000")
    monitor = record.sources[0]
    ecg = monitor.sources[0]
    assert [name for _, name, _ in reader.subtree(record.record_id, ecg.source_id)] == ["ECG"]
    assert [name for _, name, _ in reader.subtree(record.record_id, monitor.source_id)] == [
        "Bedside monitor",
        "ECG",
        "Temperature sensor",
    ]


def test_dataset_level_annotation_lands_in_dataset_annotations(reader):
    rows = reader.connection.execute(
        "SELECT a.attachment_id, n.name, n.value FROM dataset_annotations a JOIN annotations n USING (annotation_id)"
    ).fetchall()
    assert rows == [(0, "collection", "test fixtures")]
    assert [a.value for a in reader.annotations_for("dataset")] == ["test fixtures"]


def test_unknown_record_raises(reader):
    with pytest.raises(TimeFFormatError, match="no record"):
        reader.record("nope")


def test_unknown_object_type_raises(reader):
    with pytest.raises(TimeFFormatError, match="not an annotatable object type"):
        reader.annotations_for("banana", 0)


def test_reader_survives_pickling(version):
    """A torch DataLoader ships its dataset to every worker, so the reader must cross a process."""
    with TimeFReader(version) as reader:
        reader.record_ids()
        revived = pickle.loads(pickle.dumps(reader))
    assert revived.record_ids() == ["record-000", "record-001", "record-002"]
    assert _values(revived, "record-000", "record-000-lead-i").shape == (256,)
    revived.close()


def test_one_signal_shared_by_two_records(tmp_path):
    """Real corpora reuse a series across records, so a signal is stored once and linked twice.

    In ARFBench 7,013 of 9,187 series are referenced by more than one record, because several
    questions ask about the same underlying metric. The series belongs to no single record, so its
    annotation is stored with no record scope and has to come back on both.
    """
    shared = Signal(
        id="shared-series",
        name="cpu",
        values=np.arange(64, dtype=np.float32),
        time_axis=RegularAxis(period_us=Fraction(1_000_000, 1)),
        spec=TimeSeriesSpec(spec_type="metric", name="metric", unit_value=ureg.Unit(""), dtype="float32"),
    )
    shared.annotate(Annotation.static(name="unit_kind", value="ratio"))
    dataset = DeclarativeDataset(metadata=make_metadata())
    for index in (0, 1):
        dataset.add_record(Record(id=f"r-{index}", sources=[Source(id=f"src-{index}", name="host", signals=[shared])]))

    with TimeFWriter(tmp_path, dataset.metadata) as writer:
        writer.write(dataset)
    with TimeFReader(tmp_path / dataset.metadata.dataset_id / "1.0.0") as reader:
        counts = reader.counts()
        assert counts["signals"] == 1
        assert counts["source_signals"] == 2
        assert counts["signal_annotations"] == 1
        surrogates = set()
        for index in (0, 1):
            signals = reader.record(f"r-{index}").signals()
            assert [signal.external_id for signal in signals] == ["shared-series"]
            assert [a.name for a in signals[0].annotations] == ["unit_kind"]
            surrogates.add(signals[0].signal_id)
        # Both records see the same series, so both views carry the same surrogate id.
        assert len(surrogates) == 1
        assert reader.values(surrogates.pop()).tolist() == list(range(64))
        assert reader.records_with("unit_kind") == ["r-0", "r-1"]
        # One batch holds two views of the one series, and the statement is about the series, so
        # every view of it has to come back carrying the annotation.
        batched = reader.records(["r-0", "r-1"])
        assert [[a.name for a in record.signals()[0].annotations] for record in batched] == [
            ["unit_kind"],
            ["unit_kind"],
        ]


def test_write_stream_never_holds_the_whole_corpus(tmp_path):
    """The streaming path must consume records lazily, not materialize them.

    A corpus larger than memory is the reason this path exists, so counting the records a finished
    write consumed proves nothing: a writer that drains the generator into a list first consumes the
    same 50. The generator therefore looks at the values plane every time it is pulled. The budgets
    are small enough that row groups flush while the walk is still running, so a lazy writer has
    shards on disk before the last record is asked for, and a writer that materializes first has
    none, because nothing is written until the generator is exhausted.
    """
    metadata = make_metadata(dataset_id="test/streamed")
    produced = []
    shards_on_disk = []

    def records():
        for index in range(50):
            shards_on_disk.append(len(list(tmp_path.rglob("*.parquet"))))
            record = make_record(f"streamed-{index:03d}", seed=index, n_values=64)
            produced.append(record.id)
            yield record

    with TimeFWriter(tmp_path, metadata, chunk_max_bytes=1024, row_group_target_bytes=4096) as writer:
        writer.write_stream(records())
    assert len(produced) == 50
    assert shards_on_disk[0] == 0, "a shard existed before the first record was produced"
    assert shards_on_disk[-1] > 0, "nothing reached the values plane until the generator ran dry"

    with TimeFReader(tmp_path / "test/streamed" / "1.0.0") as reader:
        assert len(reader.record_ids()) == 50
        counts = reader.counts()
        assert counts["signals"] == 150
        assert counts["signal_chunks"] == 150
        record = reader.record("streamed-007")
        assert [s.name for s in record.sources[0].sources[0].signals] == ["I", "II"]
        assert _values(reader, "streamed-007", "streamed-007-lead-i").shape == (64,)


def test_write_stream_matches_the_in_memory_path(tmp_path):
    """Streaming and materializing the same dataset must produce the same tables."""
    dataset = make_dataset(n_records=4, n_values=64)
    with TimeFWriter(tmp_path / "whole", dataset.metadata) as writer:
        writer.write(dataset)
    with TimeFWriter(tmp_path / "streamed", dataset.metadata) as writer:
        writer.write_stream(iter(dataset.records), iter(dataset.tasks), annotations=dataset.annotations)

    version = f"{dataset.metadata.dataset_id}/1.0.0"
    with TimeFReader(tmp_path / "whole" / version) as a, TimeFReader(tmp_path / "streamed" / version) as b:
        assert a.counts() == b.counts()
        assert a.record_ids() == b.record_ids()
        assert [s.external_id for s in a.record("record-000").signals()] == [
            s.external_id for s in b.record("record-000").signals()
        ]
        assert (
            _values(a, "record-000", "record-000-lead-i").tolist()
            == _values(b, "record-000", "record-000-lead-i").tolist()
        )


def test_records_batch_matches_one_at_a_time(reader):
    """The batched path must rebuild exactly what the single-record path does."""
    ids = reader.record_ids()
    batched = reader.records(ids)
    assert [r.external_id for r in batched] == ids
    for one, many in zip((reader.record(rid) for rid in ids), batched, strict=True):
        assert one.record_id == many.record_id
        assert [s.name for s in one.sources] == [s.name for s in many.sources]
        assert [s.signal_id for s in one.signals()] == [s.signal_id for s in many.signals()]
        assert [a.to_text() for a in one.annotations] == [a.to_text() for a in many.annotations]
        assert [a.to_text() for s in one.signals() for a in s.annotations] == [
            a.to_text() for s in many.signals() for a in s.annotations
        ]


def test_an_id_naming_no_record_is_skipped(reader):
    assert [r.external_id for r in reader.records(["record-002", "nope", "record-000"])] == [
        "record-002",
        "record-000",
    ]


def test_tasks_batch_matches_one_at_a_time(reader):
    ids = reader.task_ids()
    batched = reader.tasks(ids)
    assert [t.external_id for t in batched] == ids
    for one, many in zip((reader.task(tid) for tid in ids), batched, strict=True):
        assert one.task_id == many.task_id
        assert one.prompt == many.prompt
        assert [i if isinstance(i, str) else i.external_id for i in one.inputs] == [
            i if isinstance(i, str) else i.external_id for i in many.inputs
        ]
        assert [a.name for a in one.annotations] == [a.name for a in many.annotations]


def test_iter_records_covers_everything_once(reader):
    walked = list(reader.iter_records(batch_size=2))
    assert sorted(r.record_id for r in walked) == list(range(len(reader.record_ids())))
    assert sorted(r.external_id for r in walked) == reader.record_ids()


def test_iter_records_never_resolves_an_external_id(reader, monkeypatch):
    """The walk draws surrogate ids straight from the table, so it has no names to translate."""

    def refuse(*args, **kwargs):
        raise AssertionError("iter_records paid a resolution hop")

    monkeypatch.setattr(TimeFReader, "_resolve_ids", refuse)
    assert len(list(reader.iter_records(batch_size=2))) == 3
    assert len(list(reader.iter_tasks(batch_size=2))) == 3


def test_workers_split_the_corpus_into_disjoint_slices(reader):
    """Four DataLoader workers must together see every record exactly once."""
    slices = [[r.record_id for r in reader.iter_records(batch_size=2, worker_index=i, num_workers=4)] for i in range(4)]
    flat = [record_id for part in slices for record_id in part]
    assert sorted(flat) == list(range(len(reader.record_ids())))
    assert len(flat) == len(set(flat))


def test_iter_tasks_hydrates_its_records(reader):
    tasks = list(reader.iter_tasks(batch_size=2))
    assert sorted(t.external_id for t in tasks) == reader.task_ids()
    assert all(t.inputs and not isinstance(t.inputs[0], str) for t in tasks)


def test_a_bad_worker_slice_is_refused(reader):
    with pytest.raises(ValueError, match="not a slice"):
        list(reader.iter_records(worker_index=4, num_workers=4))
    with pytest.raises(ValueError, match="batch_size"):
        list(reader.iter_records(batch_size=0))
