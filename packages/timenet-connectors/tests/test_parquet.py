from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import parquet
from timenet_connectors.sources.readers.base import DataRow
from timenet_connectors.sources.readers.parquet import ParquetLocation, ParquetRowSource


class Row(DataRow):
    identifier: int = Field(alias="id")
    text: str | None
    note: str = "not published"


def test_rows_keep_values_and_global_positions_across_batches_and_groups(tmp_path, monkeypatch):
    path = tmp_path / "rows.parquet"
    identifiers = [2**60 + 1, 2**60 + 2, 2**60 + 3]
    pq.write_table(pa.table({"id": identifiers, "text": [" x ", None, "z"]}), path, row_group_size=2)
    monkeypatch.setattr(parquet, "_BATCH_ROWS", 1)
    with ParquetRowSource(path, Row).open_rows() as rows:
        entries = list(rows)
    assert [entry.row.identifier for entry in entries] == identifiers
    assert [entry.row.text for entry in entries] == [" x ", None, "z"]
    assert [entry.location for entry in entries] == [ParquetLocation(path, i) for i in range(3)]
    assert all(entry.row.note == "not published" for entry in entries)


@pytest.fixture
def table_with_unused_column(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1], "text": ["x"], "unused": [b"large payload"]}), path)
    return path


def test_unused_columns_require_explicit_projection(table_with_unused_column):
    path = table_with_unused_column
    with (
        ParquetRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"unexpected.*unused"),
    ):
        list(rows)


def test_projection_reads_only_model_columns(table_with_unused_column, monkeypatch):
    selected = []
    original = pq.ParquetFile.iter_batches

    def batches(self, **kwargs):
        selected.append(kwargs["columns"])
        return original(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    with ParquetRowSource(table_with_unused_column, Row, project_columns=True).open_rows() as rows:
        assert next(rows).row.identifier == 1
    assert selected == [["id", "text"]]


@pytest.mark.parametrize("project", [False, True])
def test_empty_files_still_validate_required_columns(tmp_path, project):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": pa.array([], type=pa.int64())}), path)
    with (
        ParquetRowSource(path, Row, project_columns=project).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"missing columns.*text"),
    ):
        list(rows)


def test_repeated_columns_fail(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["id", "id"]), path)
    with (
        ParquetRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match="repeated columns"),
    ):
        list(rows)


def test_parquet_preserves_nested_values_and_decimal_precision(tmp_path):
    class RichRow(DataRow):
        samples: list[int | None]
        amount: Decimal

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"samples": [[2**60 + 1, None]], "amount": [Decimal("123456789.123456789")]}), path)
    with ParquetRowSource(path, RichRow).open_rows() as rows:
        row = next(rows).row
    assert row.samples == [2**60 + 1, None]
    assert row.amount == Decimal("123456789.123456789")


def test_decoded_strings_are_not_coerced_to_numbers(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": ["1"], "text": ["x"]}), path)
    with (
        ParquetRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"rows.parquet row 0: id:") as error,
    ):
        next(rows)
    assert error.value.__cause__ is not None
