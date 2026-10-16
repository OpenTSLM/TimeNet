"""Tests for reader-side values backends."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.storage.parquet import ParquetReadCache
from timenet.testing import make_dataset
from timenet.values_backends.parquet.reader import ParquetValuesReader
from timenet.writer import TimeFWriter


def _write(tmp_path, dataset=None, **kwargs):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


class _ProjectionSpy:
    """Wraps a real shard handle and records the columns each row-group read asks for."""

    def __init__(self, handle, projections):
        self._handle = handle
        self._projections = projections

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def read_row_group(self, row_group: int, *, columns: list[str]) -> pa.Table:
        self._projections.append(columns)
        return self._handle.read_row_group(row_group, columns=columns)


def test_row_group_cache_includes_dataset_root(tmp_path):
    sample = make_dataset()
    spec = sample.records[0].signals[0].spec
    rows = [{"chunk_file": "time_series/part-00000000.parquet", "chunk_major_idx": 0, "chunk_minor_idx": 0}]
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
        row = control.connection.execute("SELECT signal_key FROM signals WHERE signal_id = ?", [series.id]).fetchone()
        assert row is not None
        rows = control.chunk_rows_by_key(row[0], series.id)

    projections: list[list[str]] = []
    values_reader = ParquetValuesReader()
    opener = values_reader._parquet._open_file
    monkeypatch.setattr(
        values_reader._parquet,
        "_open_file",
        lambda version, rel_path: _ProjectionSpy(opener(version, rel_path), projections),
    )
    version = DatasetVersion.open_local(version_dir)
    expected = values_reader.load(version, rows, series.spec)
    assert values_reader.load(version, rows, series.spec).equals(expected)
    values_reader.close()

    assert projections == [["values"]]


def test_parquet_cache_separates_files_groups_and_projections(tmp_path):
    version = DatasetVersion.open_local(_write(tmp_path, chunk_max_bytes=64, row_group_target_bytes=64))
    path = max(version.manifest.files.time_series, key=lambda part: part.size).path
    cache = ParquetReadCache(8)
    try:
        values = cache.read_table(version, path, row_group=0, columns=("values",))
        ids = cache.read_table(version, path, row_group=0, columns=("time_series_id",))
        group = cache.read_table(version, path, row_group=0)
        complete = cache.read_table(version, path)
        assert values.column_names == ["values"]
        assert ids.column_names == ["time_series_id"]
        assert group.select(["values"]).equals(values)
        assert group.select(["time_series_id"]).equals(ids)
        assert cache.read_column(version, path, 0, "values").equals(values.column("values").chunk(0))
        assert len(complete) > len(group)
    finally:
        cache.close()


@pytest.mark.parametrize("row_group", [None, 0])
def test_parquet_cache_reports_missing_files(tmp_path, row_group):
    version = DatasetVersion.open_local(_write(tmp_path))
    cache = ParquetReadCache(8)
    with pytest.raises(TimeFFormatError, match=r"cannot read Parquet file .*missing\.parquet"):
        cache.read_table(version, "missing.parquet", row_group=row_group)
    cache.close()
