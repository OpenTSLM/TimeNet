"""Tests for reader-side values backends."""

import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.storage.parquet import PayloadTableCache, SignalRowGroupCache
from timenet.testing import make_dataset
from timenet.values_backends.parquet import reader as parquet_reader
from timenet.values_backends.parquet.reader import ParquetValuesReader
from timenet.values_backends.types import ChunkDataIndex, StoredChunk
from timenet.writer import TimeFWriter


def _write(tmp_path, dataset=None, **kwargs):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_row_group_cache_includes_dataset_root(tmp_path):
    sample = make_dataset()
    spec = sample.records[0].signals[0].spec
    rows = [StoredChunk(0, "time_series/part-00000000.parquet", ChunkDataIndex(0, 0), 1)]
    reader = ParquetValuesReader()
    try:
        for name, value in (("a", 1.0), ("b", 2.0)):
            dataset = TimeFDataset(metadata=sample.metadata)
            signal = Signal.from_values(
                [value], spec=spec, name="sensor", time_axis=sample.records[0].signals[0].time_axis
            )
            dataset.add_record(record=Record(sources=(Source(name="sensor", signals=(signal,)),)))
            version = open_local_version(_write(tmp_path / name, dataset))
            assert reader.load(version, rows, spec).to_pylist() == [value]
    finally:
        reader.close()


def test_a_loaded_series_does_not_pin_its_row_group(tmp_path):
    # Chunks of several series share a row group. Handing back a slice of the decoded group would
    # keep the whole group alive for as long as the caller keeps one series' values.
    version_dir = _write(tmp_path, chunk_max_bytes=64)
    widest_group = 0
    for shard in sorted((version_dir / "time_series").glob("*.parquet")):
        handle = pq.ParquetFile(shard)
        metadata = handle.metadata
        widest_group = max(
            widest_group, *(metadata.row_group(group).num_rows for group in range(metadata.num_row_groups))
        )
        handle.close()
    assert widest_group > 1, "the fixture needs one row group to hold several chunks"

    with TimeFReader(open_local_version(version_dir)) as reader:
        for record in reader.iter_records(with_annotations=False):
            for series in record.signals:
                values = series.to_arrow()
                assert values.offset == 0
                assert values.get_total_buffer_size() == values.nbytes


def test_a_regular_row_group_reads_only_the_values_column(tmp_path, monkeypatch):
    # Time offsets belong to the shared control-plane axis, so every values read projects only values.
    version_dir = _write(tmp_path)
    with TimeFReader(open_local_version(version_dir)) as reader:
        record = next(iter(reader.iter_records(with_annotations=False)))
        series = record.signals[0]
        control = reader._control_reader()
        row = control.connection.execute("SELECT signal_key FROM signals WHERE signal_id = ?", [series.id]).fetchone()
        assert row is not None
        chunks = control.chunk_location_by_key(row[0], series.id).chunks

    projections: list[list[str]] = []
    values_reader = ParquetValuesReader()
    original_read = pq.ParquetFile.read_row_group

    def read_row_group(handle, row_group, *, columns, **kwargs):
        projections.append(columns)
        return original_read(handle, row_group, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_row_group)
    version = open_local_version(version_dir)
    expected = values_reader.load(version, chunks, series.spec)
    assert values_reader.load(version, chunks, series.spec).equals(expected)
    values_reader.close()

    assert projections == [["values"]]


def test_signal_cache_separates_files_and_groups(tmp_path, monkeypatch):
    version = open_local_version(_write(tmp_path))
    opened = []
    original_open = pq.ParquetFile

    def open_file(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(pq, "ParquetFile", open_file)
    cache = SignalRowGroupCache(2)
    try:
        for name, start in (("first.parquet", 0), ("second.parquet", 10), ("third.parquet", 20)):
            pq.write_table(
                pa.table({"values": [[start], [start + 1]]}), version.resolve_file_path(name), row_group_size=1
            )
        assert cache.read_values(version, "first.parquet", 0).to_pylist() == [[0]]
        assert cache.read_values(version, "first.parquet", 1).to_pylist() == [[1]]
        assert cache.read_values(version, "second.parquet", 0).to_pylist() == [[10]]
        # Revisit an evicted group while its file handle is still retained.
        assert cache.read_values(version, "first.parquet", 0).to_pylist() == [[0]]
        assert cache.read_values(version, "third.parquet", 0).to_pylist() == [[20]]
        assert len(opened) == 3
        assert opened[1].closed
        assert not opened[0].closed
        # Re-open an evicted file and check that decoding still returns its own values.
        assert cache.read_values(version, "second.parquet", 0).to_pylist() == [[10]]
        assert len(opened) == 4
        assert opened[0].closed
    finally:
        cache.close()
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("cache_type", [PayloadTableCache, SignalRowGroupCache])
def test_parquet_cache_reports_missing_files(tmp_path, cache_type):
    version = open_local_version(_write(tmp_path))
    cache = cache_type(8)
    with pytest.raises(TimeFFormatError, match=r"cannot read Parquet file .*missing\.parquet"):
        if isinstance(cache, PayloadTableCache):
            cache.read_table(version, "missing.parquet")
        else:
            cache.read_values(version, "missing.parquet", 0)
    cache.close()


@pytest.mark.parametrize("row_group", [0, 1])
def test_signal_cache_rejects_missing_values_or_row_group(tmp_path, row_group):
    version = open_local_version(_write(tmp_path))
    pq.write_table(pa.table({"wrong_column": [1]}), version.resolve_file_path("broken.parquet"))
    cache = SignalRowGroupCache(8)
    # Group 0 lacks values; group 1 is outside the file. Both must fail with file context.
    with pytest.raises(TimeFFormatError, match=rf"broken\.parquet.*row group {row_group}"):
        cache.read_values(version, "broken.parquet", row_group)
    cache.close()


def test_a_signal_whose_chunks_name_two_owner_releases_fails_the_read(tmp_path):
    version_dir = _write(tmp_path, chunk_max_bytes=64)
    # Opening checks the release's files, so open it before changing its catalog.
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        row = connection.execute(
            "SELECT signal_key, signal_id FROM signals JOIN signal_chunks USING (signal_key) "
            "GROUP BY ALL HAVING count(*) > 1 ORDER BY signal_key LIMIT 1"
        ).fetchone()
        assert row is not None, "the fixture needs a Signal with several chunks"
        signal_key, signal_id = row
        connection.execute(
            "UPDATE signal_chunks SET owner_key = 'org/parent@1.0.0' WHERE signal_key = ? AND chunk_index = 0",
            [signal_key],
        )

    with TimeFReader(version) as reader:
        (signal,) = [signal for record in reader.iter_records() for signal in record.signals if signal.id == signal_id]
        with pytest.raises(TimeFFormatError, match=f"signal '{signal_id}' has chunks stored by more than one release"):
            signal.to_arrow()


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("cache_entries", [2, 64])
def test_eager_scan_releases_workers_on_success_and_failure(tmp_path, monkeypatch, corrupt, cache_entries):
    monkeypatch.setattr(parquet_reader, "_ROW_GROUP_CACHE_SIZE", cache_entries)
    path = _write(tmp_path, shard_target_bytes=64, row_group_target_bytes=32, chunk_max_bytes=32)
    version = open_local_version(path)
    decoding_threads = []
    original_read = pq.ParquetFile.read_row_group

    def read_group(*args, **kwargs):
        decoding_threads.append(threading.current_thread().name)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_group)
    if corrupt:
        # Simulate a file disappearing after preparation, while the reader is about to use it.
        shards = sorted((path / "time_series").glob("*.parquet"))
        shards[1].unlink()
    with TimeFReader(version) as reader:
        if corrupt:
            with pytest.raises(TimeFFormatError):
                reader.read().read()
        else:
            dataset = reader.read().read()
            assert dataset.records
            assert any(name.startswith("timef-scan") for name in decoding_threads)
            decoded_groups = len(decoding_threads)
            assert reader.read().read() is dataset
            assert len(decoding_threads) == decoded_groups
        assert not [thread for thread in threading.enumerate() if thread.name.startswith("timef-scan")]
