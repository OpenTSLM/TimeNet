"""Tests for reader-side values backends."""

from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.values_backends.parquet.reader import ParquetValuesReader
from timenet.writer import TimeFWriter


def _write(tmp_path, **kwargs):
    dataset = make_dataset()
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


class _Shard:
    def __init__(self, value: float) -> None:
        self._value = value

    def read_row_group(self, row_group: int, *, columns: list[str]) -> pa.Table:
        assert row_group == 0
        # Projected, never the whole row group: a shard also carries ids, spec_type and signal, and
        # decoding those on every value read is what the projection exists to avoid.
        assert columns == ["values", "time_offsets_us"]
        return pa.table(
            {
                "values": pa.array([[self._value]], type=pa.list_(pa.float32())),
                "time_offsets_us": pa.nulls(1, type=pa.list_(pa.int64())),
            }
        )


def test_row_group_cache_includes_dataset_root(monkeypatch):
    reader = ParquetValuesReader()
    monkeypatch.setattr(reader, "_shard", lambda version, rel_path: _Shard(1.0 if version.root == "a" else 2.0))
    rows = [{"chunk_file": "time_series/part-00000.parquet", "chunk_major_idx": 0, "chunk_minor_idx": 0}]
    spec = make_dataset().records[0].time_series[0].spec

    # The reader keys its row-group cache on the handle's root, so the same relative path under two
    # different roots must not collide; only version.root is touched here (_shard is stubbed).
    version_a = cast("DatasetVersion", SimpleNamespace(root="a"))
    version_b = cast("DatasetVersion", SimpleNamespace(root="b"))
    assert reader.load(version_a, rows, spec).to_pylist() == [1.0]
    assert reader.load(version_b, rows, spec).to_pylist() == [2.0]


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
            for series in record.time_series:
                values = series.to_arrow()
                assert values.offset == 0
                assert values.get_total_buffer_size() == values.nbytes


def test_a_regular_row_group_reads_only_the_values_column(tmp_path, monkeypatch):
    # Every series in the fixture is regular, so time_offsets_us is null in every row of the shard.
    # The footer says so, so the read can leave that column alone.
    version_dir = _write(tmp_path)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        record = next(iter(reader.iter_records(with_annotations=False)))
        series = record.time_series[0]
        rows = reader._index_rows(record.record_id, series.time_series_id)

    projections: list[list[str]] = []
    values_reader = ParquetValuesReader()
    opener = values_reader._shard
    monkeypatch.setattr(
        values_reader, "_shard", lambda version, rel_path: _ProjectionSpy(opener(version, rel_path), projections)
    )
    values_reader.load(DatasetVersion.open_local(version_dir), rows, series.spec)
    values_reader.close()

    assert projections == [["values"]]
