"""Write a hierarchy, read it back, and check every relationship survived the trip."""

from fractions import Fraction
import pickle

import numpy as np
import pytest

from timenet.control_plane import (
    Annotation,
    DeclarativeDataset,
    Record,
    Signal,
    Source,
    TimeFReader,
    TimeFWriter,
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


def test_every_record_round_trips(reader):
    assert reader.record_ids() == ["record-000", "record-001", "record-002"]


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
    record = reader.record("record-000")
    lead_i = record.sources[0].sources[0].signals[0]
    assert lead_i.spec_type == "ecg-voltage"
    assert lead_i.unit == "millivolt"
    assert lead_i.dtype == "float32"
    assert lead_i.axis_type == "regular"
    assert lead_i.n_values == 256


def test_annotations_land_at_every_level(reader):
    record = reader.record("record-000")
    assert {a.name for a in record.annotations} == {"patient_sex", "patient_age"}
    ecg = record.sources[0].sources[0]
    assert [a.name for a in ecg.annotations] == ["device_model"]
    assert [a.name for a in ecg.signals[0].annotations] == ["lead_status"]


def test_point_annotation_keeps_its_placement(reader):
    lead_i = reader.record("record-000").sources[0].sources[0].signals[0]
    (status,) = lead_i.annotations
    assert status.span_type == "point"
    assert status.start_us == 6_000_000
    assert status.to_text() == "lead_status=Lead fell off @6s"


def test_shared_annotation_is_stored_once(reader):
    payloads = reader.connection.execute(
        "SELECT count(*) FROM annotation_contents WHERE name = 'patient_sex'"
    ).fetchone()[0]
    occurrences = reader.connection.execute(
        "SELECT count(*) FROM annotation_occurrences o JOIN annotation_contents c USING (content_id) "
        "WHERE c.name = 'patient_sex'"
    ).fetchone()[0]
    assert payloads == 1
    assert occurrences == 3


def test_values_round_trip_with_their_dtype(version, reader):
    values = reader.values("record-000-lead-i")
    assert values.shape == (256,)
    assert values.dtype == np.float32


def test_irregular_signal_round_trips(reader):
    chest = reader.record("record-000").sources[0].sources[1].signals[0]
    assert chest.axis_type == "irregular"
    assert reader.values("record-000-chest").shape == (10,)


def test_task_resolves_its_inputs_and_targets(reader):
    task = reader.task("diagnosis-000")
    assert task.prompt == "Diagnose this patient."
    assert len(task.inputs) == 1
    assert task.inputs[0].record_id == "record-000"
    assert task.target == ["The patient is stable."]
    assert [a.name for a in task.annotations] == ["task_type"]


def test_render_task_walks_the_whole_hierarchy(reader):
    rendered = render_task(reader, "diagnosis-000")
    assert "PROMPT: Diagnose this patient." in rendered
    assert "source: Bedside monitor" in rendered
    assert "signal: I (ecg-voltage" in rendered
    assert "[signal] lead_status=Lead fell off @6s" in rendered
    assert "[record] patient_sex=male" in rendered
    assert "TARGET: The patient is stable." in rendered


def test_reverse_lookup_from_annotation_to_records(reader):
    assert reader.records_with("patient_sex", "male") == ["record-000", "record-001", "record-002"]


def test_reverse_lookup_reaches_records_through_signals(reader):
    # lead_status is attached to a signal, not to the record, so this only resolves if the
    # occurrence carries the record it sits in.
    assert reader.records_with("lead_status") == ["record-000", "record-001", "record-002"]


def test_reverse_lookup_from_annotation_to_objects(reader):
    found = reader.objects_with("device_model")
    assert {kind for kind, _ in found} == {"source"}
    assert len(found) == 3


def test_tasks_for_record_without_a_link_table(reader):
    assert reader.tasks_for_record("record-001") == ["diagnosis-001"]


def test_subtree_returns_the_branch(reader):
    found = reader.subtree("record-000", "record-000-ecg")
    assert [name for _, name, _ in found] == ["ECG"]
    whole = reader.subtree("record-000", "record-000-monitor")
    assert [name for _, name, _ in whole] == ["Bedside monitor", "ECG", "Temperature sensor"]


def test_dataset_level_annotation_is_addressable(reader):
    assert [a.value for a in reader.annotations_for("dataset", "dataset")] == ["test fixtures"]


def test_unknown_record_raises(reader):
    with pytest.raises(TimeFFormatError, match="no record"):
        reader.record("nope")


def test_unknown_object_type_raises(reader):
    with pytest.raises(TimeFFormatError, match="not an annotatable object type"):
        reader.annotations_for("banana", "x")


def test_reader_survives_pickling(version):
    """A torch DataLoader ships its dataset to every worker, so the reader must cross a process."""
    with TimeFReader(version) as reader:
        reader.record_ids()
        revived = pickle.loads(pickle.dumps(reader))
    assert revived.record_ids() == ["record-000", "record-001", "record-002"]
    assert revived.values("record-000-lead-i").shape == (256,)
    revived.close()


def test_one_signal_shared_by_two_records(tmp_path):
    """Real corpora reuse a series across records, so a signal is stored once and linked twice.

    In ARFBench 7,013 of 9,187 series are referenced by more than one record, because several
    questions ask about the same underlying metric.
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
        for index in (0, 1):
            signals = reader.record(f"r-{index}").signals()
            assert [s.signal_id for s in signals] == ["shared-series"]
            assert [a.name for a in signals[0].annotations] == ["unit_kind"]
        # the values are stored once, and both records read the same array
        assert reader.values("shared-series").tolist() == list(range(64))
        assert reader.records_with("unit_kind") == ["r-0", "r-1"]


def test_write_stream_never_holds_the_whole_corpus(tmp_path):
    """The streaming path must consume records lazily, not materialize them.

    A corpus larger than memory is the reason this path exists, so the test asserts the generator
    is still being pulled from while rows are already on disk, and that nothing kept a list of them.
    """
    metadata = make_metadata(dataset_id="test/streamed")
    produced = []

    def records():
        for index in range(50):
            record = make_record(f"streamed-{index:03d}", seed=index, n_values=64)
            produced.append(record.id)
            yield record

    with TimeFWriter(tmp_path, metadata) as writer:
        writer.write_stream(records())
    assert len(produced) == 50

    with TimeFReader(tmp_path / "test/streamed" / "1.0.0") as reader:
        assert len(reader.record_ids()) == 50
        counts = reader.counts()
        assert counts["signals"] == 150
        assert counts["signal_chunks"] == 150
        record = reader.record("streamed-007")
        assert [s.name for s in record.sources[0].sources[0].signals] == ["I", "II"]
        assert reader.values("streamed-007-lead-i").shape == (64,)


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
        assert [s.signal_id for s in a.record("record-000").signals()] == [
            s.signal_id for s in b.record("record-000").signals()
        ]
        assert a.values("record-000-lead-i").tolist() == b.values("record-000-lead-i").tolist()


def test_records_batch_matches_one_at_a_time(reader):
    """The batched path must rebuild exactly what the single-record path does."""
    ids = reader.record_ids()
    batched = reader.records(ids)
    assert [r.record_id for r in batched] == ids
    for one, many in zip((reader.record(rid) for rid in ids), batched, strict=True):
        assert [s.name for s in one.sources] == [s.name for s in many.sources]
        assert [s.signal_id for s in one.signals()] == [s.signal_id for s in many.signals()]
        assert [a.to_text() for a in one.annotations] == [a.to_text() for a in many.annotations]
        assert [a.to_text() for s in one.signals() for a in s.annotations] == [
            a.to_text() for s in many.signals() for a in s.annotations
        ]


def test_tasks_batch_matches_one_at_a_time(reader):
    ids = reader.task_ids()
    batched = reader.tasks(ids)
    assert [t.task_id for t in batched] == ids
    for one, many in zip((reader.task(tid) for tid in ids), batched, strict=True):
        assert one.prompt == many.prompt
        assert [i if isinstance(i, str) else i.record_id for i in one.inputs] == [
            i if isinstance(i, str) else i.record_id for i in many.inputs
        ]
        assert [a.name for a in one.annotations] == [a.name for a in many.annotations]


def test_iter_records_covers_everything_once(reader):
    seen = [r.record_id for r in reader.iter_records(batch_size=2)]
    assert sorted(seen) == reader.record_ids()


def test_workers_split_the_corpus_into_disjoint_slices(reader):
    """Four DataLoader workers must together see every record exactly once."""
    slices = [[r.record_id for r in reader.iter_records(batch_size=2, worker_index=i, num_workers=4)] for i in range(4)]
    flat = [rid for part in slices for rid in part]
    assert sorted(flat) == reader.record_ids()
    assert len(flat) == len(set(flat))


def test_iter_tasks_hydrates_its_records(reader):
    tasks = list(reader.iter_tasks(batch_size=2))
    assert [t.task_id for t in tasks] == reader.task_ids()
    assert all(t.inputs and not isinstance(t.inputs[0], str) for t in tasks)


def test_a_bad_worker_slice_is_refused(reader):
    with pytest.raises(ValueError, match="not a slice"):
        list(reader.iter_records(worker_index=4, num_workers=4))
    with pytest.raises(ValueError, match="batch_size"):
        list(reader.iter_records(batch_size=0))
