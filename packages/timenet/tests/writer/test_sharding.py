"""Sharded values-plane round trips, targeted-read verification, and record read order."""

import duckdb
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.dataset.edit import edit_version
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.storage.parquet import SignalRowGroupCache
from timenet.testing import assert_datasets_equal
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


# Small enough that a modest dataset shards every artifact type into several parts, tiny on disk.
_SMALL_TARGETS = {
    "shard_target_bytes": 256,
    "row_group_target_bytes": 128,
    "chunk_max_bytes": 64,
}


def _signal_id(i: int) -> int:
    """Return the id the dataset gives the signal of record ``i``. Signals count from 1."""
    return 1 + i


def _sharded_dataset(n_records: int, series_len: int) -> TimeFDataset:
    """Build a synthetic dataset large enough to shard every artifact type under small byte targets."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/sharding",
            dataset_version=Version(1, 0, 0),
            name="Sharding",
            description="synthetic",
            license=License.MIT,
        )
    )
    spec = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
    axis = RegularAxis.from_rate_hz(16)
    for i in range(n_records):
        values = [float((i + j) % 11) for j in range(series_len)]
        ts = Signal.from_values(values, spec=spec, name="a", time_axis=axis)
        record = dataset.add_record(
            record=Record(
                sources=(Source(name="Source", signals=(ts,)),),
            )
        )
        record.annotate(Annotation(key="subject_ids", value=[f"subj-{i}"]))
        record.annotate(Annotation(key="label", value=f"cls-{i % 3}"))
        dataset.add_task(task=ClassificationTask(inputs=(record,), targets=(f"c{i % 2}",)))
    return dataset


def _write(tmp_path, dataset, **targets):
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **targets) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _index_rows(version_dir):
    with duckdb.connect(str(version_dir / "control.duckdb"), read_only=True) as connection:
        rows = connection.execute(
            """SELECT signal_id, chunk_index, value_path, chunk_major_index, chunk_minor_index, n_values
                   FROM signal_chunks
                   ORDER BY signal_id, chunk_index"""
        ).fetchall()
    return [
        {
            "signal_id": row[0],
            "chunk_idx": row[1],
            "chunk_file": row[2],
            "chunk_major_idx": row[3],
            "chunk_minor_idx": row[4],
            "n_values": row[5],
        }
        for row in rows
    ]


# ---- round trip ------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_records", "series_len"), [(12, 128), (5, 300), (20, 40)])
def test_values_shard_and_the_dataset_round_trips(tmp_path, n_records, series_len):
    original = _sharded_dataset(n_records, series_len)
    version_dir = _write(tmp_path, _sharded_dataset(n_records, series_len), **_SMALL_TARGETS)
    files = Manifest.model_validate_json((version_dir / "manifest.json").read_text()).files
    assert len(files.control) == 1
    assert len(files.time_series) >= 3  # values-plane shards
    for rel in files.all_parts():
        assert (version_dir / rel).exists()
    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


# ---- metadata --------------------------------------------------------------------------------


def test_index_metadata_locates_every_series_in_the_shards(tmp_path):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    shards = {part.path for part in manifest.files.time_series}
    row_group_counts = {rel: pq.ParquetFile(version_dir / rel).metadata.num_row_groups for rel in shards}
    rows = _index_rows(version_dir)
    for row in rows:
        assert row["chunk_file"] in shards  # locator points at a listed shard
        assert 0 <= row["chunk_major_idx"] < row_group_counts[row["chunk_file"]]  # a real row group in it
        assert row["n_values"] > 0


# ---- targeted reads: only the shards / row groups we need ------------------------------------


def test_open_and_build_records_reads_no_value_shard(tmp_path, monkeypatch):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    opened: list[str] = []
    original = pq.ParquetFile
    monkeypatch.setattr(pq, "ParquetFile", lambda p, *a, **k: opened.append(str(p)) or original(p, *a, **k))
    with TimeFReader(open_local_version(version_dir)) as reader:
        dataset = reader.read()
        _ = [s.signals for s in dataset.records]  # touch every record's series metadata
    # values are lazy; building records may open control-plane tables but no value shard (under time_series/)
    assert not [p for p in opened if "/time_series/" in p]


@pytest.mark.parametrize("series_id", [_signal_id(0), _signal_id(6), _signal_id(11)])
def test_reading_a_series_opens_only_its_value_shards(tmp_path, monkeypatch, series_id):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
    expected_shards = {row["chunk_file"] for row in _index_rows(version_dir) if row["signal_id"] == series_id}
    assert expected_shards <= {part.path for part in manifest.files.time_series}
    assert len(expected_shards) < len(manifest.files.time_series)  # the series lives in only some shards

    opened: list[str] = []
    original = pq.ParquetFile
    monkeypatch.setattr(pq, "ParquetFile", lambda p, *a, **k: opened.append(str(p)) or original(p, *a, **k))
    with TimeFReader(open_local_version(version_dir)) as reader:
        series = reader.get(Signal, series_id)
        # Getting a Signal builds its Record from control tables, without opening a value shard.
        assert not [p for p in opened if "/time_series/" in p]
        series.to_arrow()
    opened_rel = {p.removeprefix(f"{version_dir}/") for p in opened if "/time_series/" in p}
    assert opened_rel == expected_shards  # exactly the series' value shards, nothing else


def test_reading_a_series_reads_only_its_row_groups(tmp_path, monkeypatch):
    version_dir = _write(tmp_path, _sharded_dataset(12, 128), **_SMALL_TARGETS)
    expected = {
        (r["chunk_file"], r["chunk_major_idx"]) for r in _index_rows(version_dir) if r["signal_id"] == _signal_id(6)
    }

    read: list[tuple[str, int]] = []
    original = SignalRowGroupCache.read_values

    def spy(self, version, rel_path, row_group):
        read.append((rel_path, row_group))
        return original(self, version, rel_path, row_group)

    monkeypatch.setattr(SignalRowGroupCache, "read_values", spy)
    with TimeFReader(open_local_version(version_dir)) as reader:
        series = reader.get(Signal, _signal_id(6))
        series.to_arrow()
    assert set(read) == expected  # the reader decoded only the row groups of signal 6's chunks, not whole shards


# ---- record order ----------------------------------------------------------------------------


def _metadata(dataset_id, parents=None):
    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version(1, 0, 0),
        name="Record order",
        description="synthetic",
        license=License.MIT,
        parents=parents or {},
    )


_AXIS = RegularAxis.from_rate_hz(16)


def _signal(spec_type):
    spec = TimeSeriesSpec(spec_type=spec_type, name=spec_type.upper(), unit_value=ureg.dimensionless)
    return Signal.from_values([float(i) for i in range(128)], spec=spec, name=spec_type, time_axis=_AXIS)


def _record(*spec_types):
    return Record(sources=(Source(name="Source", signals=tuple(map(_signal, spec_types))),))


def _stored_signal_ids(version_dir):
    rows = _index_rows(version_dir)
    return list(
        dict.fromkeys(
            row["signal_id"]
            for row in sorted(rows, key=lambda row: (row["chunk_file"], row["chunk_major_idx"], row["chunk_minor_idx"]))
        )
    )


def _child_of(tmp_path, parents, *local_records, scattered=True):
    """Write ``parents``, then a child that imports all of their records and adds ``local_records``.

    The child declares each parent and imports each parent's records in one call, in
    reverse unless ``scattered`` is false. Copies take the child's IDs in the order it imports
    them, so in reverse their IDs sort against where the parent stores their values. Returns the
    child's version directory and, per parent, the copies' IDs in the parent's record order. Each
    parent below stores its records' earliest values in that order.
    """
    readers = [TimeFReader(open_local_version(_write(tmp_path, parent, **_SMALL_TARGETS))) for parent in parents]
    try:
        declared = {
            ParentAlias(f"parent-{index}"): DatasetRef(
                dataset_id=reader.metadata.dataset_id, version=reader.metadata.dataset_version
            )
            for index, reader in enumerate(readers)
        }
        child = TimeFDataset(metadata=_metadata("test/child", parents=declared))
        imports = []
        for reader, parent in zip(readers, parents, strict=True):
            selected = [record.id for record in parent.records]
            imports.append(child.import_from(reader, records=selected[::-1] if scattered else selected))
        for record in local_records:
            child.add_record(record=record)
        child_path = _write(tmp_path, child)
    finally:
        for reader in readers:
            reader.close()
    copies = [
        tuple(imported.records[record.id].id for record in parent.records)
        for imported, parent in zip(imports, parents, strict=True)
    ]
    return child_path, copies


def test_a_release_without_imported_records_and_its_edits_order_records_by_id(tmp_path):
    dataset = TimeFDataset(metadata=_metadata("timenet/own-values"))
    a, b, c = (dataset.add_record(record=_record(spec_type)) for spec_type in ("z", "a", "a"))
    base = _write(tmp_path, dataset, **_SMALL_TARGETS)
    # The writer groups values by spec type first, so a's values are stored last.
    assert _stored_signal_ids(base) == [b.signals[0].id, c.signals[0].id, a.signals[0].id]
    # An edit stores no values; it reads the base's, and keeps the base's IDs and their order.
    edited = edit_version(base, dataset_version=Version(1, 0, 1), remove_record_ids=[c.id])
    for path, expected in ((base, (a.id, b.id, c.id)), (edited, (a.id, b.id))):
        with TimeFReader(open_local_version(path)) as reader:
            assert reader.record_ids() == expected


@pytest.mark.parametrize("scattered", [True, False], ids=["scattered-import", "import-in-parent-order"])
def test_lazy_reads_of_imported_records_follow_storage_order(tmp_path, monkeypatch, scattered):
    child_path, (copies,) = _child_of(tmp_path, [_sharded_dataset(12, 128)], scattered=scattered)
    # Imported in reverse, the copies' IDs sort against where their values are stored; imported in
    # the parent's order, they sort the same way.
    assert (sorted(copies) != list(copies)) is scattered
    read = []
    original = SignalRowGroupCache.read_values

    def track(self, version, relpath, group):
        read.append((relpath, group))
        return original(self, version, relpath, group)

    monkeypatch.setattr(SignalRowGroupCache, "read_values", track)
    with TimeFReader(open_local_version(child_path)) as reader:
        assert reader.record_ids() == copies
        # Each Signal loads on its own, so only the record order keeps these reads sequential.
        for record in reader.iter_records():
            for signal in record.signals:
                signal.to_arrow()
    assert len(read) > len(copies)
    assert read == sorted(read)


def test_an_edit_of_an_imported_child_keeps_storage_order(tmp_path):
    child_path, (copies,) = _child_of(tmp_path, [_sharded_dataset(6, 128)])
    assert sorted(copies) != list(copies)
    edited = edit_version(child_path, dataset_version=Version(1, 0, 1), remove_record_ids=[copies[2]])
    with TimeFReader(open_local_version(edited)) as reader:
        assert reader.record_ids() == (*copies[:2], *copies[3:])


def test_an_edit_that_drops_every_imported_copy_orders_records_by_id(tmp_path):
    local = [_record(spec_type) for spec_type in ("z", "a", "a")]
    child_path, (copies,) = _child_of(tmp_path, [_sharded_dataset(3, 128)], *local)
    edited = edit_version(child_path, dataset_version=Version(1, 0, 1), remove_record_ids=list(copies))
    with TimeFReader(open_local_version(edited)) as reader:
        # Ranked by storage, the first local record would come last: its spec type stores its values
        # after the others'.
        assert reader.record_ids() == tuple(record.id for record in local)


def test_every_record_of_a_mixed_child_has_its_own_place_in_the_record_order(tmp_path):
    # This parent's source key sorts first, and both parents spread their records over several
    # values files with the same names, so ranking by file before source would interleave them.
    first = TimeFDataset(metadata=_metadata("aaa/first"))
    # The writer groups values by spec type, so the first record's earliest values belong to its
    # second Signal, and its first Signal's values are stored after the second record's.
    first.add_record(record=_record("z", "s"))
    first.add_record(record=_record("s"))
    local_values = _record("s")
    empty = [Record(sources=(Source(name="Empty"),)) for _ in range(2)]
    child_path, (first_copies, second_copies) = _child_of(
        tmp_path, [first, _sharded_dataset(4, 128)], local_values, *empty
    )
    # The child's own values come first, then each parent's in turn; records without values come
    # last, by ID.
    expected = (local_values.id, *first_copies, *second_copies, *(record.id for record in empty))
    assert sorted(expected) != list(expected)
    with duckdb.connect(str(child_path / "control.duckdb"), read_only=True) as connection:
        ranks = connection.execute("SELECT read_order FROM records ORDER BY read_order").fetchall()
    assert [rank for (rank,) in ranks] == list(range(len(expected)))
    for _ in range(2):
        with TimeFReader(open_local_version(child_path)) as reader:
            assert reader.record_ids() == expected
            assert tuple(record.id for record in reader.iter_records()) == expected
