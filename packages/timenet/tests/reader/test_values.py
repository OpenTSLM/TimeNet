"""Tests for reader-side values backends."""

from pathlib import Path

import pyarrow as pa

from timenet.reader.values import ParquetValuesReader


class _Shard:
    def __init__(self, value: float) -> None:
        self._value = value

    def read_row_group(self, row_group: int, *, columns: list[str]) -> pa.Table:
        assert row_group == 0
        assert columns == ["values"]
        return pa.table({"values": pa.array([[self._value]], type=pa.list_(pa.float32()))})


def test_row_group_cache_includes_dataset_root(monkeypatch):
    reader = ParquetValuesReader()
    monkeypatch.setattr(reader, "_shard", lambda root, rel_path: _Shard(1.0 if root == Path("a") else 2.0))
    rows = [{"chunk_file": "time_series/shard-00000.parquet", "chunk_offset0": 0, "chunk_offset1": 0}]

    assert reader.load(Path("a"), rows).to_pylist() == [1.0]
    assert reader.load(Path("b"), rows).to_pylist() == [2.0]
