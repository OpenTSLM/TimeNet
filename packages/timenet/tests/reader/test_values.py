"""Tests for reader-side values backends."""

import dataclasses

import pyarrow as pa

from timenet.registry import LocalRegistry
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


def test_row_group_cache_includes_dataset_root(tmp_path, monkeypatch):
    # The decoded-row-group cache keys on the version root, so the same shard/row-group under two
    # different roots does not collide (it did when the key ignored the root).
    reader = ParquetValuesReader()
    monkeypatch.setattr(reader, "_shard", lambda version, rel_path: _Shard(1.0 if version.root == "a" else 2.0))
    rows = [{"chunk_file": "time_series/shard-00000.parquet", "chunk_major_idx": 0, "chunk_minor_idx": 0}]
    registry = LocalRegistry(tmp_path)
    registry.store(make_dataset())
    base = registry.open_version("timenet/hello-world")  # a real handle; only its root varies below
    spec = make_dataset().samples[0].time_series[0].spec

    assert reader.load(dataclasses.replace(base, root="a"), rows, spec).to_pylist() == [1.0]
    assert reader.load(dataclasses.replace(base, root="b"), rows, spec).to_pylist() == [2.0]
