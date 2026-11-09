from contextlib import closing
import json

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ConfigDict
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import CsvReader, DataRow, JsonlReader, JsonReader, ParquetReader


class Row(DataRow):
    name: str
    count: int


@pytest.mark.parametrize("format_name", ["csv", "json", "jsonl", "parquet"])
def test_readers_validate_rows_in_order(tmp_path, format_name):
    path = tmp_path / format_name
    values = [{"name": "first", "count": 1}, {"name": "second", "count": 2}]
    if format_name == "csv":
        path.write_text("name,count\nfirst,1\nsecond,2\n")
        reader = CsvReader(path, Row)
    elif format_name == "json":
        path.write_text(json.dumps(values))
        reader = JsonReader(path, Row)
    elif format_name == "jsonl":
        path.write_text("\n".join(map(json.dumps, values)))
        reader = JsonlReader(path, Row)
    else:
        pq.write_table(pa.Table.from_pylist(values), path)
        reader = ParquetReader(path, Row)
    assert list(reader.iter_rows()) == [Row.model_validate(value) for value in values]
    with closing(reader.iter_rows()) as rows:
        assert next(rows) == Row.model_validate(values[0])


@pytest.mark.parametrize(
    ("reader_type", "content", "location"),
    [
        (CsvReader, "name,count\nx,bad\n", "line 2"),
        (JsonlReader, '{"name":"x","count":"bad"}\n', "line 1"),
        (JsonReader, '[{"name":"x","count":"bad"}]', "entry 1"),
    ],
)
def test_invalid_row_has_source_context(tmp_path, reader_type, content, location):
    path = tmp_path / "source"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=location) as exc:
        list(reader_type(path, Row).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


def test_parquet_invalid_row_has_source_context(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["x"], "count": ["bad"]}), path)
    with pytest.raises(TimeNetConnectorError, match="row 1"):
        list(ParquetReader(path, Row).iter_rows())


@pytest.mark.parametrize("content", ["name,name\nx,y\n", "name,count\nx,1,extra\n", "name,count\nx\n"])
def test_csv_rejects_ambiguous_rows(tmp_path, content):
    path = tmp_path / "source.csv"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError):
        list(CsvReader(path, Row).iter_rows())


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


@pytest.mark.parametrize("format_name", ["json", "jsonl", "parquet"])
def test_decoded_readers_offer_strict_validation(tmp_path, format_name):
    path = tmp_path / format_name
    value = {"name": "sensor", "count": "42"}
    if format_name == "json":
        path.write_text(json.dumps([value]))
        reader_type = JsonReader
    elif format_name == "jsonl":
        path.write_text(json.dumps(value) + "\n")
        reader_type = JsonlReader
    else:
        pq.write_table(pa.Table.from_pylist([value]), path)
        reader_type = ParquetReader

    assert list(reader_type(path, Row).iter_rows()) == [Row(name="sensor", count=42)]
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(reader_type(path, Row, strict=True).iter_rows())


def test_strict_mode_accepts_decoded_integer_and_csv_strings(tmp_path):
    json_path = tmp_path / "source.json"
    json_path.write_text('[{"name":"sensor","count":42}]')
    assert list(JsonReader(json_path, Row, strict=True).iter_rows()) == [Row(name="sensor", count=42)]
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("name,count\nsensor,42\n")
    assert list(CsvReader(csv_path, Row, strict=True).iter_rows()) == [Row(name="sensor", count=42)]


def test_default_respects_row_schema_strictness(tmp_path):
    class StrictRow(DataRow):
        model_config = ConfigDict(strict=True)
        count: int

    path = tmp_path / "source.json"
    path.write_text('[{"count":"42"}]')
    with pytest.raises(TimeNetConnectorError, match="count"):
        list(JsonReader(path, StrictRow).iter_rows())
    assert list(JsonReader(path, StrictRow, strict=False).iter_rows()) == [StrictRow(count=42)]
