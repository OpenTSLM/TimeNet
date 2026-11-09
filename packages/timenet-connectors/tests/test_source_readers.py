from contextlib import closing
import json

import pyarrow as pa
import pyarrow.parquet as pq
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
