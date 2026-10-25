import pyarrow as pa
import pyarrow.parquet as pq

from timenet_connectors.sources.parquet import iter_parquet_rows


def test_iter_parquet_rows_yields_plain_dicts_in_file_order(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", None, "z"]}), path, row_group_size=2)
    assert list(iter_parquet_rows(path)) == [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}]
