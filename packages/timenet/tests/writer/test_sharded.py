import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer.encodings import ParquetEncoding
from timenet.writer.sharded import write_sharded_table


_SCHEMA = pa.schema([("id", pa.string()), ("n", pa.int64())])


def _rows(count):
    return [{"id": f"id-{i:04d}", "n": i} for i in range(count)]


def _write(tmp_path, rows, control_target_bytes, row_group_target_bytes=64):
    return write_sharded_table(
        rows,
        _SCHEMA,
        lambda i: f"t/part-{i:08d}.parquet",
        staging_dir=tmp_path,
        control_target_bytes=control_target_bytes,
        row_group_target_bytes=row_group_target_bytes,
        encoding=ParquetEncoding(dictionary_columns=[], compression="zstd", compression_level=3),
    )


def test_small_target_shards_per_row(tmp_path):
    parts = _write(tmp_path, _rows(3), control_target_bytes=1)
    assert parts == [f"t/part-{i:08d}.parquet" for i in range(3)]
    assert all(pq.read_table(tmp_path / rel).num_rows == 1 for rel in parts)
    seen = [row["id"] for rel in parts for row in pq.read_table(tmp_path / rel).to_pylist()]
    assert seen == [f"id-{i:04d}" for i in range(3)]


def test_large_target_writes_one_part(tmp_path):
    parts = _write(tmp_path, _rows(50), control_target_bytes=128 * 2**20)
    assert len(parts) == 1
    table = pq.read_table(tmp_path / parts[0])
    assert table.num_rows == 50
    ids = table.column("id").to_pylist()
    assert ids[0] == "id-0000" and ids[-1] == "id-0049"


def test_empty_input_writes_one_empty_part(tmp_path):
    parts = _write(tmp_path, [], control_target_bytes=1)
    assert len(parts) == 1
    assert pq.read_table(tmp_path / parts[0]).num_rows == 0


def test_row_groups_are_sized_within_a_part(tmp_path):
    # A tiny row-group target forces multiple row groups inside a single part.
    parts = _write(tmp_path, _rows(50), control_target_bytes=128 * 2**20, row_group_target_bytes=1)
    assert len(parts) == 1
    assert pq.ParquetFile(tmp_path / parts[0]).num_row_groups > 1
