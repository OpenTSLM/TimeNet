from dataclasses import replace
from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataRow, NonBlankText
from timenet_connectors.sources.readers.jsonl import JsonlLocation, JsonlRowSource


class Row(DataRow):
    name: NonBlankText
    count: int


def test_utf8_crlf_and_missing_final_newline_preserve_bytes_and_rereads(tmp_path):
    path = tmp_path / "rows.jsonl"
    first = '{"name":"café", "count":1152921504606846977}\r\n'.encode()
    last = b'{"name":"two", "count":2}'
    path.write_bytes(first + last)
    source = JsonlRowSource(path, Row)
    with source.open_rows() as rows:
        entries = list(rows)
    assert [entry.row.name for entry in entries] == ["café", "two"]
    assert [entry.row.count for entry in entries] == [2**60 + 1, 2]
    assert [entry.location for entry in entries] == [
        JsonlLocation(path, 1, 0, len(first)),
        JsonlLocation(path, 2, len(first), len(last)),
    ]
    assert [source.read_bytes(entry.location) for entry in entries] == [first, last]
    assert [source.read_row(entry.location) for entry in entries] == entries


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"\n", "Expecting value"),
        (b"{bad}", "Expecting property name"),
        (b'{"name":"x","count":"2"}', "count:"),
        (b'{"name":"x","count":1,"extra":{"a":1,"a":2}}', "duplicate key 'a'"),
        (b'{"name":"\xff","count":1}', "utf-8"),
    ],
)
def test_invalid_lines_report_original_physical_location(tmp_path, raw, message):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"name":"good","count":1}\r\n' + raw)
    with JsonlRowSource(path, Row).open_rows() as rows:
        assert next(rows).row.name == "good"
        with pytest.raises(TimeNetConnectorError, match=message) as error:
            next(rows)
    assert f"{path} (line 2)" in str(error.value)
    assert error.value.__cause__ is not None


def test_nonobject_line_reports_its_source_location(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"[]\n")
    with (
        JsonlRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match="expected a row object") as error,
    ):
        next(rows)
    assert f"{path} (line 1)" in str(error.value)


def test_reread_revalidates_changed_source_values(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"name":"good","count":100}\n')
    source = JsonlRowSource(path, Row)
    with source.open_rows() as rows:
        entry = next(rows)
    # Equal-length edits preserve the byte range but must still satisfy the source schema.
    path.write_bytes(b'{"name":"good","count":"x"}\n')
    with pytest.raises(TimeNetConnectorError, match=r"line 1.*count:") as error:
        source.read_row(entry.location)
    assert str(path) in str(error.value)


def test_reread_reports_a_truncated_range(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"name":"good","count":1}\n')
    source = JsonlRowSource(path, Row)
    with source.open_rows() as rows:
        location = next(rows).location
    path.write_bytes(b"{")
    with pytest.raises(TimeNetConnectorError, match=r"line 1.*truncated") as error:
        source.read_row(location)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("field,value", [("path", Path("other.jsonl")), ("line", 0), ("offset", -1), ("length", 0)])
def test_invalid_locations_fail_before_file_access(tmp_path, monkeypatch, field, value):
    path = tmp_path / "missing.jsonl"
    source = JsonlRowSource(path, Row)
    location = replace(JsonlLocation(path, 1, 0, 20), **{field: value})

    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid locations must fail before file access")

    monkeypatch.setattr(Path, "open", unexpected_open)
    with pytest.raises(TimeNetConnectorError, match="expected a positive line and byte range"):
        source.read_row(location)


def test_empty_file_yields_no_rows(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    with JsonlRowSource(path, Row).open_rows() as rows:
        assert list(rows) == []


def test_duplicate_key_errors_preserve_the_connector_cause(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"name":"x","count":1,"count":2}\n')
    with (
        JsonlRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match="duplicate key 'count'") as error,
    ):
        next(rows)
    assert isinstance(error.value.__cause__, TimeNetConnectorError)
    assert f"{path} (line 1)" in str(error.value)
