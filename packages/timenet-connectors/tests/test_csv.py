from enum import Enum
from typing import Annotated

from pydantic import Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataRow, NonBlankText, PublishedEnum, RowSource
from timenet_connectors.sources.readers.csv import CsvLocation, CsvRowSource


class Mode(Enum):
    DAILY = "daily"


class LabelRow(DataRow):
    identifier: NonBlankText = Field(alias="id")
    label: NonBlankText
    age: Annotated[int, Field(ge=0, le=120)] | None = None


@pytest.fixture
def table(tmp_path):
    return tmp_path / "labels.csv"


def test_source_contract_preserves_text_order_and_physical_locations(table):
    table.write_bytes(b'label,id\r\n"first\r\nsecond",001\r\n"  text, kept  ",002\r\n')
    source: RowSource[LabelRow, CsvLocation] = CsvRowSource(table, LabelRow, unique="identifier")
    with source.open_rows() as rows:
        first, second = rows
    assert first.row == LabelRow(id="001", label="first\r\nsecond")
    assert second.row == LabelRow(id="002", label="  text, kept  ")
    assert first.location == CsvLocation(table, 2)
    assert second.location == CsvLocation(table, 4)
    assert second.location.context == f"{table}:4"


@pytest.mark.parametrize(
    "content,message",
    [
        ("", "no header row"),
        ("id,label,id\n", "repeated columns"),
        ("id\n", "missing columns.*label"),
        ("id,label,extra\n", "unexpected columns.*extra"),
        ("id,label\n\n", "labels.csv:2: blank row"),
        ("id,label\n001\n", "labels.csv:2: expected 2 fields, got 1"),
        ("id,label\n001,a,b\n", "labels.csv:2: expected 2 fields, got 3"),
        ('id,"label"x\n', "labels.csv:1:"),
        ('id,label\n001,"unclosed\n', "labels.csv:2:"),
    ],
)
def test_invalid_structure_fails_with_source_context(table, content, message):
    table.write_text(content)
    with CsvRowSource(table, LabelRow).open_rows() as rows, pytest.raises(TimeNetConnectorError, match=message):
        list(rows)


@pytest.mark.parametrize("tail,field", [("002, ,3", "label"), ("002,text,121", "age"), ("002,text,", "age")])
def test_invalid_values_after_multiline_record_name_starting_line(table, tail, field):
    table.write_text('id,label,age\n001,"first\nsecond",42\n' + tail + "\n")
    with CsvRowSource(table, LabelRow).open_rows() as rows:
        assert next(rows).row.age == 42
        with pytest.raises(TimeNetConnectorError, match=rf"labels\.csv:4: {field}:") as error:
            next(rows)
    assert error.value.__cause__ is not None


def test_uniqueness_compares_validated_values_and_names_both_locations(table):
    table.write_text('id,label,age\n001,"first\nsecond",042\n002,text,42\n')
    with (
        CsvRowSource(table, LabelRow, unique="age").open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"labels\.csv:4: age: duplicate 42; first at .*labels\.csv:2"),
    ):
        list(rows)


def test_repeated_headers_are_explicit_and_keep_locations(table):
    class EnumRow(DataRow):
        mode: PublishedEnum[Mode]

    table.write_text("mode\ndaily\nmode\ndaily\n")
    with (
        CsvRowSource(table, EnumRow).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"labels\.csv:3: mode:"),
    ):
        list(rows)
    with CsvRowSource(table, EnumRow, skip_repeated_header=True).open_rows() as rows:
        entries = list(rows)
    assert [entry.location.line for entry in entries] == [2, 4]
    assert all(entry.row.mode is Mode.DAILY for entry in entries)
