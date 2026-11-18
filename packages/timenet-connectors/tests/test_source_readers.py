from contextlib import closing
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ConfigDict, ValidationError
import pytest

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.sources.readers import (
    CsvDataReader,
    DataRow,
    JsonArrayDataReader,
    JsonlDataReader,
    ParquetDataReader,
)


class Row(DataRow):
    name: str
    count: int


class StrictRow(Row):
    model_config = ConfigDict(strict=True)


def test_data_row_rejects_unexpected_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Row.model_validate({"name": "sensor", "count": 1, "extra": "value"})


def test_csv_reads_rows_in_order(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nfirst,1\nsecond,2\n")
    reader = CsvDataReader(path, Row)
    assert list(reader.iter_rows()) == [Row(name="first", count=1), Row(name="second", count=2)]


def test_csv_invalid_row_reports_location(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nx,bad\n")
    with pytest.raises(TimeNetConnectorError, match="line 2") as exc:
        list(CsvDataReader(path, Row).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "missing or duplicate CSV header"),
        ("\n", "missing or duplicate CSV header"),
        ("\nname,count\nx,1\n", "missing or duplicate CSV header"),
        ("name,name\nx,y\n", "missing or duplicate CSV header"),
        ("name,count\nx,1,extra\n", "line 2: too many cells"),
        ("name,count\nx\n", "line 2: too few cells"),
    ],
)
def test_csv_rejects_ambiguous_rows(tmp_path, content, message):
    path = tmp_path / "source.csv"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=message):
        list(CsvDataReader(path, Row).iter_rows())


def test_csv_checks_expected_header_even_without_rows(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("count,name\n")
    with pytest.raises(TimeNetConnectorError, match="expected columns"):
        list(CsvDataReader(path, Row, expected_columns=("name", "count")).iter_rows())


def test_closing_a_partially_consumed_csv_reader_closes_its_file(tmp_path, monkeypatch):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nfirst,1\nsecond,2\n")
    handles = []
    open_file = Path.open

    def track_open(file_path, *args, **kwargs):
        handle = open_file(file_path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track_open)
    with closing(CsvDataReader(path, Row).iter_rows()) as rows:
        assert next(rows) == Row(name="first", count=1)
        assert not handles[0].closed
    assert handles[0].closed


def test_parquet_reads_rows_in_order(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["first", "second"], "count": [1, 2]}), path)
    reader = ParquetDataReader(path, Row)
    assert list(reader.iter_rows()) == [Row(name="first", count=1), Row(name="second", count=2)]


def test_parquet_strict_validation_and_error_context(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["sensor"], "count": ["42"]}), path)
    assert list(ParquetDataReader(path, Row).iter_rows()) == [Row(name="sensor", count=42)]
    with pytest.raises(TimeNetConnectorError, match="row 1") as exc:
        list(ParquetDataReader(path, StrictRow).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


def test_parquet_reads_only_selected_columns(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["x"], "count": [1], "other": [3]}), path)
    assert list(ParquetDataReader(path, Row, selected_columns=("name", "count")).iter_rows()) == [
        Row(name="x", count=1)
    ]


def test_parquet_rejects_missing_selected_columns(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["x"], "count": [1]}), path)
    with pytest.raises(TimeNetConnectorError, match="missing columns"):
        list(ParquetDataReader(path, Row, selected_columns=("missing",)).iter_rows())


def test_parquet_rejects_duplicate_columns(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table([pa.array(["x"]), pa.array(["y"])], names=["name", "name"]), path)
    with pytest.raises(TimeNetConnectorError, match="duplicate Parquet column"):
        list(ParquetDataReader(path, Row).iter_rows())


def test_parquet_conversion_error_has_path_and_batch_rows(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"date": pa.array([0, 3_000_000], type=pa.date32())}), path)
    with pytest.raises(TimeNetConnectorError, match="rows 1-2") as exc:
        list(ParquetDataReader(path, DataRow).iter_rows())
    assert str(path) in str(exc.value)
    assert isinstance(exc.value.__cause__, OverflowError)


class ArrowRow(DataRow):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    values: pa.ListScalar


def _arrow_file(path: Path, count: int = 3) -> None:
    values: list[list[float] | None] = [[float(index), float(index) + 0.5] for index in range(count)]
    values[1] = None
    pq.write_table(pa.table({"name": [f"row{index}" for index in range(count)], "values": values}), path)


def test_parquet_keeps_arrow_columns_as_arrow_scalars(tmp_path):
    path = tmp_path / "source.parquet"
    _arrow_file(path)
    rows = list(ParquetDataReader(path, ArrowRow, arrow_columns=("values",)).iter_rows())
    assert [row.name for row in rows] == ["row0", "row1", "row2"]
    assert all(isinstance(row.values, pa.ListScalar) for row in rows)
    assert rows[0].values.values.to_pylist() == [0.0, 0.5]
    assert not rows[1].values.is_valid
    assert rows[2].values.values.to_pylist() == [2.0, 2.5]


def test_parquet_keeps_arrow_columns_across_batches(tmp_path):
    path = tmp_path / "source.parquet"
    count = 65537
    _arrow_file(path, count)
    last = None
    for index, row in enumerate(ParquetDataReader(path, ArrowRow, arrow_columns=("values",)).iter_rows()):
        last = (index, row)
    assert last is not None
    assert last[0] == count - 1
    assert last[1].name == f"row{count - 1}"
    assert last[1].values.values.to_pylist() == [float(count - 1), float(count - 1) + 0.5]


def test_parquet_reads_rows_when_every_column_stays_arrow(tmp_path):
    class AllArrow(DataRow):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: pa.StringScalar

    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["a", "b"]}), path)
    rows = list(ParquetDataReader(path, AllArrow, arrow_columns=("name",)).iter_rows())
    assert [row.name.as_py() for row in rows] == ["a", "b"]


def test_parquet_rejects_arrow_columns_that_are_not_selected(tmp_path):
    path = tmp_path / "source.parquet"
    with pytest.raises(TimeFValidationError, match="arrow_columns must be selected columns"):
        ParquetDataReader(path, ArrowRow, selected_columns=("name",), arrow_columns=("values",))


def test_parquet_rejects_missing_arrow_columns(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["x"]}), path)
    with pytest.raises(TimeNetConnectorError, match="missing columns"):
        list(ParquetDataReader(path, ArrowRow, arrow_columns=("values",)).iter_rows())


def test_parquet_reads_across_batches_in_order(tmp_path):
    path = tmp_path / "source.parquet"
    count = 65537
    pq.write_table(pa.table({"name": ["sensor"] * count, "count": range(count)}), path)
    rows = ParquetDataReader(path, Row).iter_rows()
    for expected_count, row in enumerate(rows):
        assert row == Row(name="sensor", count=expected_count)
    assert expected_count == count - 1


def test_closing_a_partially_consumed_parquet_reader_closes_its_file(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["first", "second"], "count": [1, 2]}), path)
    opened_files = []
    open_parquet = pq.ParquetFile

    def track_open(*args, **kwargs):
        parquet = open_parquet(*args, **kwargs)
        opened_files.append(parquet)
        return parquet

    monkeypatch.setattr(pq, "ParquetFile", track_open)
    with closing(ParquetDataReader(path, Row).iter_rows()) as rows:
        assert next(rows) == Row(name="first", count=1)
        assert not opened_files[0].closed
    assert opened_files[0].closed


def test_default_respects_row_schema_strictness(tmp_path):
    class StrictCount(DataRow):
        model_config = ConfigDict(strict=True)
        count: int

    path = tmp_path / "source.json"
    path.write_text('[{"count":"42"}]')
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(JsonArrayDataReader(path, StrictCount).iter_rows())


@pytest.mark.parametrize("reader_type", [JsonArrayDataReader, JsonlDataReader])
def test_json_readers_validate_rows_in_order(tmp_path, reader_type):
    path = tmp_path / "source"
    values = [{"name": "first", "count": 1}, {"name": "second", "count": 2}]
    path.write_text(json.dumps(values) if reader_type is JsonArrayDataReader else "\n".join(map(json.dumps, values)))
    reader = reader_type(path, Row)
    assert list(reader.iter_rows()) == [Row(name="first", count=1), Row(name="second", count=2)]


@pytest.mark.parametrize(
    ("reader_type", "content", "location"),
    [
        (JsonArrayDataReader, '[{"name":"x","count":"bad"}]', "entry 1"),
        (JsonlDataReader, '{"name":"x","count":"bad"}\n', "line 1"),
    ],
)
def test_json_invalid_row_has_source_context(tmp_path, reader_type, content, location):
    path = tmp_path / "source"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=location) as exc:
        list(reader_type(path, Row).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


@pytest.mark.parametrize("reader_type", [JsonArrayDataReader, JsonlDataReader])
def test_json_strict_validation(tmp_path, reader_type):
    path = tmp_path / "source"
    value = {"name": "sensor", "count": "42"}
    path.write_text(json.dumps([value] if reader_type is JsonArrayDataReader else value))
    assert list(reader_type(path, Row).iter_rows()) == [Row(name="sensor", count=42)]
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(reader_type(path, StrictRow).iter_rows())


def test_json_requires_array_and_jsonl_rejects_invalid_line(tmp_path):
    path = tmp_path / "source"
    path.write_text("{}")
    with pytest.raises(TimeNetConnectorError, match="array"):
        list(JsonArrayDataReader(path, Row).iter_rows())
    path.write_text("{}\nnot json\n")
    with pytest.raises(TimeNetConnectorError, match="line 2"):
        list(JsonlDataReader(path, DataRow).iter_rows())


@pytest.mark.parametrize("reader_type", [JsonArrayDataReader, JsonlDataReader])
def test_deeply_nested_json_has_source_context(tmp_path, reader_type):
    path = tmp_path / "source.json"
    depth = 100_000
    nested = "[" * depth + "0" + "]" * depth
    path.write_text(nested if reader_type is JsonArrayDataReader else "{}\n" + nested + "\n")
    with pytest.raises(TimeNetConnectorError) as exc:
        list(reader_type(path, DataRow).iter_rows())
    assert str(path) in str(exc.value)
    if reader_type is JsonlDataReader:
        assert "line 2" in str(exc.value)
    assert isinstance(exc.value.__cause__, RecursionError)


@pytest.mark.parametrize(
    ("reader_type", "content", "location"),
    [
        (JsonArrayDataReader, '[{"name":"first","name":"second","count":1}]', "source"),
        (JsonlDataReader, '{"name":"first","name":"second","count":1}\n', "line 1"),
        (JsonArrayDataReader, '[{"nested":{"x":1,"x":2}}]', "source"),
        (JsonlDataReader, '{"nested":{"x":1,"x":2}}\n', "line 1"),
    ],
)
def test_json_rejects_duplicate_keys(tmp_path, reader_type, content, location):
    path = tmp_path / "source"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match="duplicate key") as exc:
        list(reader_type(path, Row).iter_rows())
    assert location in str(exc.value)
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


def test_closing_a_partially_consumed_jsonl_reader_closes_its_file(tmp_path, monkeypatch):
    path = tmp_path / "source.jsonl"
    path.write_text('{"name":"first","count":1}\n{"name":"second","count":2}\n')
    handles = []
    open_file = Path.open

    def track_open(file_path, *args, **kwargs):
        handle = open_file(file_path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track_open)
    with closing(JsonlDataReader(path, Row).iter_rows()) as rows:
        assert next(rows) == Row(name="first", count=1)
        assert not handles[0].closed
    assert handles[0].closed
