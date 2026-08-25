"""Tests for reader-side values backends."""

from types import SimpleNamespace
from typing import cast

import pyarrow as pa

from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.values_backends.parquet.reader import ParquetValuesReader


class _Shard:
    def __init__(self, value: float) -> None:
        self._value = value

    def read_row_group(self, row_group: int, *, columns: list[str]) -> pa.Table:
        assert row_group == 0
        # Projected, never the whole row group: a shard also carries ids, spec_type and channel, and
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
    spec = make_dataset().samples[0].time_series[0].spec

    # The reader keys its row-group cache on the handle's root, so the same relative path under two
    # different roots must not collide; only version.root is touched here (_shard is stubbed).
    version_a = cast("DatasetVersion", SimpleNamespace(root="a"))
    version_b = cast("DatasetVersion", SimpleNamespace(root="b"))
    assert reader.load(version_a, rows, spec).to_pylist() == [1.0]
    assert reader.load(version_b, rows, spec).to_pylist() == [2.0]
