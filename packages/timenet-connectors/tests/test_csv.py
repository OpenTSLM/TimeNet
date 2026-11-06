from pathlib import Path
from typing import Annotated

from pydantic import Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.csv import iter_csv_rows
from timenet_connectors.sources.rows import NonBlankText, SourceRow, validate_row


class Row(SourceRow):
    pid: NonBlankText = Field(alias="id")
    value: NonBlankText
    age: Annotated[int, Field(ge=0, le=120)] | None = None


def test_aliases_optional_columns_and_multiline_locations_preserve_text(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('value,id\r\n"first\nsecond",001\r\n"  text, kept  ",002\r\n')
    rows = list(iter_csv_rows(path, Row, unique="pid"))
    assert [(line, row.pid, row.value, row.age) for line, row in rows] == [
        (2, "001", "first\nsecond", None),
        (4, "002", "  text, kept  ", None),
    ]


@pytest.mark.parametrize("header", ["", "id\n", "id,id\n", "id,other\n", "id,value,extra\n"])
def test_invalid_headers_fail_even_without_data(tmp_path, header):
    path = tmp_path / "table.csv"
    path.write_text(header)
    with pytest.raises(TimeNetConnectorError, match="expected the columns"):
        list(iter_csv_rows(path, Row))


@pytest.mark.parametrize(
    "row,field", [("002, ,3", "value"), ("002,text,121", "age"), ("002,text,3.5", "age"), (",text,3", "id")]
)
def test_validation_after_multiline_rows_keeps_physical_location(tmp_path, row, field):
    path = tmp_path / "table.csv"
    path.write_text('id,value,age\n001,"first\nsecond",3\n' + row + "\n")
    with pytest.raises(TimeNetConnectorError, match=rf"table.csv:4: {field}:"):
        list(iter_csv_rows(path, Row))


def test_csv_parses_typed_values_but_decoded_rows_do_not_coerce(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("id,value,age\n001,text,42\n")
    row = next(iter_csv_rows(path, Row))[1]
    assert row.age == 42 and row.pid == "001"
    assert validate_row({"id": "001", "value": "text", "age": 42}, Row, location="data.json row 2") == row
    with pytest.raises(TimeNetConnectorError, match=r"data\.json row 2: age:"):
        validate_row({"id": "001", "value": "text", "age": "42"}, Row, location="data.json row 2")


@pytest.mark.parametrize("row", ["002\n", "002,text,extra\n", "\n"])
def test_ragged_and_blank_rows_are_structural_errors(tmp_path, row):
    path = tmp_path / "table.csv"
    path.write_text('id,value\n001,"first\nsecond"\n' + row)
    with pytest.raises(TimeNetConnectorError, match=r"table\.csv:4:"):
        list(iter_csv_rows(path, Row))


@pytest.mark.parametrize(
    "content,line", [('id,"value"x\n', 1), ('id,value\n001,"unclosed\n', 2), ('id,value\n001,"first\nsecond"x\n', 3)]
)
def test_csv_syntax_errors_keep_failure_line(tmp_path, content, line):
    path = tmp_path / "table.csv"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=rf"table.csv:{line}:"):
        list(iter_csv_rows(path, Row))


def test_uniqueness_names_both_physical_lines(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('id,value\n001,"first\nsecond"\n001,text\n')
    with pytest.raises(TimeNetConnectorError, match=r"table\.csv:4: pid: duplicate '001'; first listed on line 2"):
        list(iter_csv_rows(path, Row, unique="pid"))


def test_blank_optional_cell_is_not_silently_defaulted(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("id,value,age\n001,text,\n")
    with pytest.raises(TimeNetConnectorError, match="age:"):
        list(iter_csv_rows(path, Row))


@pytest.mark.parametrize("content", [None, b"\xff\xfe"])
def test_read_errors_keep_path_and_cause(tmp_path, content):
    path = tmp_path / "table.csv"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(TimeNetConnectorError) as error:
        list(iter_csv_rows(path, Row))
    assert str(error.value).startswith(str(path)) and error.value.__cause__ is not None


def test_reader_is_lazy_and_releases_file_when_closed(tmp_path, monkeypatch):
    path = tmp_path / "table.csv"
    path.write_text("id,value\n001,text\n002,text\n")
    opened = []
    original = Path.open

    def track(self, *args, **kwargs):
        handle = original(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track)
    rows = iter_csv_rows(path, Row)
    assert not opened
    next(rows)
    rows.close()
    assert opened[0].closed
