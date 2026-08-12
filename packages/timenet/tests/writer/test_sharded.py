import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer.sharded import write_sharded_table


_SCHEMA = pa.schema([("id", pa.string()), ("n", pa.int64())])


def _rows(count):
    return [{"id": f"id-{i:04d}", "n": i} for i in range(count)]


def _write(tmp_path, rows, control_target_bytes):
    return write_sharded_table(
        rows,
        _SCHEMA,
        lambda i: f"t/part-{i:08d}.parquet",
        key_of=lambda r: (r["id"],),
        staging_dir=tmp_path,
        control_target_bytes=control_target_bytes,
        row_group_target_bytes=64,
        dictionary_columns=[],
        column_encoding=None,
        compression="zstd",
        compression_level=3,
    )


def test_small_target_shards_per_row_with_keys(tmp_path):
    parts = _write(tmp_path, _rows(3), control_target_bytes=1)
    assert [p.path for p in parts] == [f"t/part-{i:08d}.parquet" for i in range(3)]
    assert parts[0].first_key == ("id-0000",) and parts[0].last_key == ("id-0000",)
    assert all(p.n_rows == 1 for p in parts)
    seen = [row["id"] for part in parts for row in pq.read_table(tmp_path / part.path).to_pylist()]
    assert seen == [f"id-{i:04d}" for i in range(3)]


def test_large_target_writes_one_part(tmp_path):
    parts = _write(tmp_path, _rows(50), control_target_bytes=128 * 2**20)
    assert len(parts) == 1
    assert parts[0].n_rows == 50
    assert parts[0].first_key == ("id-0000",) and parts[0].last_key == ("id-0049",)


def test_empty_input_writes_one_empty_part_with_null_keys(tmp_path):
    parts = _write(tmp_path, [], control_target_bytes=1)
    assert len(parts) == 1
    assert parts[0].n_rows == 0
    assert parts[0].first_key is None and parts[0].last_key is None
    assert pq.read_table(tmp_path / parts[0].path).num_rows == 0


def test_key_of_none_records_null_keys(tmp_path):
    parts = write_sharded_table(
        _rows(2),
        _SCHEMA,
        lambda i: f"t/part-{i:08d}.parquet",
        key_of=None,
        staging_dir=tmp_path,
        control_target_bytes=128 * 2**20,
        row_group_target_bytes=64,
        dictionary_columns=[],
        column_encoding=None,
        compression="zstd",
        compression_level=3,
    )
    assert parts[0].first_key is None and parts[0].last_key is None
