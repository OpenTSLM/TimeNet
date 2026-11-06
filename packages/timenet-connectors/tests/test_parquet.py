import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.parquet import iter_parquet_rows


def test_iter_parquet_rows_yields_plain_dicts_in_file_order(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", None, "z"]}), path, row_group_size=2)
    assert list(iter_parquet_rows(path)) == [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}]


def test_iter_parquet_rows_reads_only_the_named_columns(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"], "c": [True, False]}), path)
    assert list(iter_parquet_rows(path, columns=["c", "a"])) == [{"c": True, "a": 1}, {"c": False, "a": 2}]


def test_iter_parquet_rows_rejects_a_column_the_file_does_not_have(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"a": [1]}), path)
    with pytest.raises(TimeNetConnectorError, match=r"no column \['nope'\]"):
        list(iter_parquet_rows(path, columns=["a", "nope"]))


def test_a_file_that_is_not_parquet_raises_a_connector_error(tmp_path):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(TimeNetConnectorError, match=r"broken\.parquet: cannot read the parquet file"):
        list(iter_parquet_rows(path))


def test_a_corrupt_page_raises_a_connector_error(tmp_path):
    path = tmp_path / "corrupt.parquet"
    pq.write_table(pa.table({"a": list(range(1000))}), path)
    data = bytearray(path.read_bytes())
    for position in range(10, 200):
        data[position] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(TimeNetConnectorError, match=r"corrupt\.parquet: cannot read the parquet file"):
        list(iter_parquet_rows(path))


def test_a_missing_file_is_not_a_source_data_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_parquet_rows(tmp_path / "missing.parquet"))
