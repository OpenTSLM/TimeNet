"""Sharded values-plane round trips and targeted-read verification."""

import threading

import duckdb
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
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
from timenet.writer import TimeFWriter, sharded


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


# ---- parallel shard writes -------------------------------------------------------------------


def _two_modality_dataset() -> TimeFDataset:
    """Build a dataset whose two modalities each fill several shards.

    With a few workers, shards of ``s`` are still in flight when ``t`` opens its writer. Both modalities
    get dictionary encoding, so this does not exercise an encoding change between them.
    """
    dataset = _sharded_dataset(12, 128)
    spec = TimeSeriesSpec(spec_type="t", name="T", unit_value=ureg.dimensionless)
    for i in range(12):
        values = [float(i * j) / 7 for j in range(96)]
        ts = Signal.from_values(values, spec=spec, name="b", time_axis=RegularAxis.from_rate_hz(16))
        dataset.add_record(record=Record(sources=(Source(name="Source", signals=(ts,)),)))
    return dataset


def _value_shards(version_dir) -> dict[str, bytes]:
    return {
        path.relative_to(version_dir).as_posix(): path.read_bytes()
        for path in sorted((version_dir / "time_series").iterdir())
    }


def test_overlapping_shard_writes_produce_the_serial_bytes_and_report_shards_in_order(tmp_path, monkeypatch):
    original = sharded._write_part
    part_1_written = threading.Event()

    def write_part(path, *args):
        if path.name == "part-00000000.parquet":
            # Part 1 must finish while part 0 waits, so the parts overlap and complete out of order.
            assert part_1_written.wait(timeout=10), "part 1 never ran while part 0 was in flight"
        original(path, *args)
        if path.name == "part-00000001.parquet":
            part_1_written.set()

    finalized: list[int] = []

    def on_progress(event):
        if event.stage == "shard_finalized":
            finalized.append(event.completed)

    serial = _write(tmp_path / "serial", _two_modality_dataset(), **_SMALL_TARGETS)
    monkeypatch.setattr(sharded, "_write_part", write_part)
    parallel = _write(
        tmp_path / "parallel", _two_modality_dataset(), shard_workers=3, progress_cb=on_progress, **_SMALL_TARGETS
    )
    assert part_1_written.is_set()  # the pool wrote the parts, not the serial path
    shards = _value_shards(serial)
    assert len(shards) > 3  # enough shards that the pool waits for a free worker
    assert _value_shards(parallel) == shards
    assert finalized == list(range(1, len(shards) + 1))  # submission order, though part 1 finished first


def test_a_failed_parallel_shard_write_aborts_the_release(tmp_path, monkeypatch):
    original = sharded._write_part
    written: list[object] = []

    def fail_third(*args):
        written.append(args[0])
        if len(written) == 3:
            raise OSError("disk full")
        original(*args)

    monkeypatch.setattr(sharded, "_write_part", fail_third)
    dataset = _two_modality_dataset()
    with pytest.raises(OSError, match="disk full"):
        _write(tmp_path, dataset, shard_workers=2, **_SMALL_TARGETS)
    dataset_dir = tmp_path / dataset.metadata.dataset_id
    assert not (dataset_dir / str(dataset.metadata.dataset_version)).exists()  # nothing committed
    assert not list(dataset_dir.glob("*.tmp-*"))  # and no staging directory left behind
