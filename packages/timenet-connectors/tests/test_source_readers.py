from contextlib import closing
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ConfigDict, ValidationError
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import CsvDataReader, DataRow, JsonlReader, JsonReader, ParquetDataReader


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
    class StrictRow(DataRow):
        model_config = ConfigDict(strict=True)
        count: int

    path = tmp_path / "source.json"
    path.write_text('[{"count":"42"}]')
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(JsonReader(path, StrictRow).iter_rows())
    assert list(JsonReader(path, StrictRow, strict=False).iter_rows()) == [StrictRow(count=42)]


@pytest.mark.parametrize("reader_type", [JsonReader, JsonlReader])
def test_json_readers_validate_rows_in_order(tmp_path, reader_type):
    path = tmp_path / "source"
    values = [{"name": "first", "count": 1}, {"name": "second", "count": 2}]
    path.write_text(json.dumps(values) if reader_type is JsonReader else "\n".join(map(json.dumps, values)))
    reader = reader_type(path, Row, strict=True)
    assert list(reader.iter_rows()) == [Row(name="first", count=1), Row(name="second", count=2)]


@pytest.mark.parametrize(
    ("reader_type", "content", "location"),
    [
        (JsonReader, '[{"name":"x","count":"bad"}]', "entry 1"),
        (JsonlReader, '{"name":"x","count":"bad"}\n', "line 1"),
    ],
)
def test_json_invalid_row_has_source_context(tmp_path, reader_type, content, location):
    path = tmp_path / "source"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=location) as exc:
        list(reader_type(path, Row).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


@pytest.mark.parametrize("reader_type", [JsonReader, JsonlReader])
def test_json_strict_validation(tmp_path, reader_type):
    path = tmp_path / "source"
    value = {"name": "sensor", "count": "42"}
    path.write_text(json.dumps([value] if reader_type is JsonReader else value))
    assert list(reader_type(path, Row).iter_rows()) == [Row(name="sensor", count=42)]
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(reader_type(path, Row, strict=True).iter_rows())


def test_json_requires_array_and_jsonl_rejects_invalid_line(tmp_path):
    path = tmp_path / "source"
    path.write_text("{}")
    with pytest.raises(TimeNetConnectorError, match="array"):
        list(JsonReader(path, Row).iter_rows())
    path.write_text("{}\nnot json\n")
    with pytest.raises(TimeNetConnectorError, match="line 2"):
        list(JsonlReader(path, DataRow).iter_rows())


@pytest.mark.parametrize(
    ("reader_type", "content", "location"),
    [
        (JsonReader, '[{"name":"first","name":"second","count":1}]', "source"),
        (JsonlReader, '{"name":"first","name":"second","count":1}\n', "line 1"),
        (JsonReader, '[{"nested":{"x":1,"x":2}}]', "source"),
        (JsonlReader, '{"nested":{"x":1,"x":2}}\n', "line 1"),
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
    with closing(JsonlReader(path, Row).iter_rows()) as rows:
        assert next(rows) == Row(name="first", count=1)
        assert not handles[0].closed
    assert handles[0].closed
