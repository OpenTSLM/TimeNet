"""Tests for reader-side values backends."""

import threading
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeFFormatError
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

    def read_row_group(self, row_group: int, *, columns: list[str], use_threads: bool = True) -> pa.Table:
        self._projections.append(columns)
        return self._handle.read_row_group(row_group, columns=columns, use_threads=use_threads)


class _Shard:
    def __init__(self, value: float) -> None:
        self._value = value

    def read_row_group(self, row_group: int, *, columns: list[str], use_threads: bool = True) -> pa.Table:
        assert row_group == 0
        # Projected, never the whole row group: a shard also carries ids, spec_type and signal, and
        # decoding those on every value read is what the projection exists to avoid.
        assert columns == ["values"]
        return pa.table({"values": pa.array([[self._value]], type=pa.list_(pa.float32()))})


def test_row_group_cache_includes_dataset_root(monkeypatch):
    reader = ParquetValuesReader()
    monkeypatch.setattr(reader, "_shard", lambda version, rel_path: _Shard(1.0 if version.root == "a" else 2.0))
    rows = [{"chunk_file": "time_series/part-00000.parquet", "chunk_major_idx": 0, "chunk_minor_idx": 0}]
    spec = next(signal for record in make_dataset().records for signal in record.signals).spec

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
    opener = values_reader._shard
    monkeypatch.setattr(
        values_reader, "_shard", lambda version, rel_path: _ProjectionSpy(opener(version, rel_path), projections)
    )
    values_reader.load(DatasetVersion.open_local(version_dir), rows, series.spec)
    values_reader.close()

    assert projections == [["values"]]


@pytest.mark.parametrize("corrupt", [False, True])
def test_eager_scan_releases_workers_on_success_and_failure(tmp_path, corrupt):
    path = _write(tmp_path, shard_target_bytes=64, row_group_target_bytes=32, chunk_max_bytes=32)
    version = DatasetVersion.open_local(path)
    if corrupt:
        # Simulate a file disappearing after preparation, while the reader is about to use it.
        shards = sorted((path / "time_series").glob("*.parquet"))
        shards[1].unlink()
    with TimeFReader(version) as reader:
        if corrupt:
            with pytest.raises(TimeFFormatError):
                reader.read()
        else:
            assert reader.read().records
        assert not [thread for thread in threading.enumerate() if thread.name.startswith("timef-scan")]
