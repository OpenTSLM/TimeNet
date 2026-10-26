"""Tests for reader-side values backends."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.storage.parquet import SignalRowGroupCache
from timenet.testing import make_dataset
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
            version = DatasetVersion.open_local(_write(tmp_path / name, dataset))
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

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        for record in reader.iter_records(with_annotations=False):
            for series in record.signals:
                values = series.to_arrow()
                assert values.offset == 0
                assert values.get_total_buffer_size() == values.nbytes


def test_a_regular_row_group_reads_only_the_values_column(tmp_path, monkeypatch):
    # Time offsets belong to the shared control-plane axis, so every values read projects only values.
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(iter(reader.iter_records(with_annotations=False)))
        series = record.signals[0]
        control = reader._control_reader()
        rows = control.chunk_rows(series.id)

    projections: list[list[str]] = []
    values_reader = ParquetValuesReader()
    original_read = pq.ParquetFile.read_row_group

    def read_row_group(handle, row_group, *, columns):
        projections.append(columns)
        return original_read(handle, row_group, columns=columns)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_row_group)
    version = DatasetVersion.open_local(version_dir)
    expected = values_reader.load(version, rows, series.spec)
    assert values_reader.load(version, rows, series.spec).equals(expected)
    values_reader.close()

    assert projections == [["values"]]


def test_signal_cache_separates_files_and_groups(tmp_path, monkeypatch):
    version = DatasetVersion.open_local(_write(tmp_path))
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
            pq.write_table(pa.table({"values": [[start], [start + 1]]}), version.path(name), row_group_size=1)
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


def test_parquet_cache_reports_missing_files(tmp_path):
    version = DatasetVersion.open_local(_write(tmp_path))
    cache = SignalRowGroupCache(8)
    with pytest.raises(TimeFFormatError, match=r"cannot read Parquet file .*missing\.parquet"):
        cache.read_values(version, "missing.parquet", 0)
    cache.close()


@pytest.mark.parametrize("row_group", [0, 1])
def test_signal_cache_rejects_missing_values_or_row_group(tmp_path, row_group):
    version = DatasetVersion.open_local(_write(tmp_path))
    pq.write_table(pa.table({"wrong_column": [1]}), version.path("broken.parquet"))
    cache = SignalRowGroupCache(8)
    # Group 0 lacks values; group 1 is outside the file. Both must fail with file context.
    with pytest.raises(TimeFFormatError, match=rf"broken\.parquet.*row group {row_group}"):
        cache.read_values(version, "broken.parquet", row_group)
    cache.close()
