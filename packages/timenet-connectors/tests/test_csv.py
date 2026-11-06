from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.csv import check_unique, read_csv_table


def test_reordered_columns_preserve_text_and_physical_starting_lines(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text('value,id\r\n"first\nsecond",001\r\n"  text, kept  ",002\r\n', encoding="utf-8")
    rows = read_csv_table(path, columns=("id", "value"))
    assert [(row.line, row.values) for row in rows] == [
        (2, {"id": "001", "value": "first\nsecond"}),
        (4, {"id": "002", "value": "  text, kept  "}),
    ]


@pytest.mark.parametrize("header", ["", "id\n", "id,id\n", "id,other\n", "id,value,extra\n"])
def test_invalid_headers_are_rejected(tmp_path, header):
    path = tmp_path / "table.csv"
    path.write_text(header, encoding="utf-8")
    with pytest.raises(TimeNetConnectorError) as error:
        read_csv_table(path, columns=("id", "value"))
    assert str(error.value).startswith(f"{path}: expected the columns")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("\n", "blank row"),
        ("002\n", "no value for 'value'"),
        ("002,\n", "no value for 'value'"),
        (",text\n", "no value for 'id'"),
        ("002,text,extra\n", "more fields than the header names"),
    ],
)
def test_bad_rows_after_multiline_fields_have_physical_locations(tmp_path, row, message):
    path = tmp_path / "table.csv"
    path.write_text('id,value\n001,"first\nsecond"\n' + row, encoding="utf-8")
    with pytest.raises(TimeNetConnectorError) as error:
        read_csv_table(path, columns=("id", "value"))
    assert str(error.value) == f"{path}:4: {message}"


def test_header_only_table_is_allowed(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("id,value\n", encoding="utf-8")
    assert read_csv_table(path, columns=("id", "value")) == []


@pytest.mark.parametrize("content", [None, b"\xff\xfe", b'id,value\n001,"unclosed\n'])
def test_read_errors_keep_the_path_and_cause(tmp_path, content):
    path = tmp_path / "table.csv"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(TimeNetConnectorError) as error:
        read_csv_table(path, columns=("id", "value"))
    assert error.value.__cause__ is not None
    assert str(error.value).startswith(f"{path}: ")


def test_duplicates_identify_the_first_and_repeated_physical_lines():
    path = Path("metadata.csv")
    with pytest.raises(TimeNetConnectorError) as error:
        check_unique(path, iter([(2, "P001"), (4, "P002"), (7, "P001")]), name="pid")
    assert str(error.value) == "metadata.csv:7: a pid is listed twice: 'P001'; first listed on line 2"


@pytest.mark.parametrize("values", [[], [(2, "001"), (3, "1")]])
def test_unique_values_are_not_normalized(values):
    check_unique(Path("metadata.csv"), iter(values), name="id")
