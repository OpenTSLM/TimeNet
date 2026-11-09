from dataclasses import replace
import json

from pydantic import Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import json_object
from timenet_connectors.sources.readers.base import DataRow
from timenet_connectors.sources.readers.json_object import JsonObjectRowSource


class Row(DataRow):
    identifier: int = Field(alias="id")
    text: str | None


def test_scans_and_rereads_preserve_keys_exact_bytes_and_values(tmp_path, monkeypatch):
    path = tmp_path / "rows.json"
    first = '{"id": 1152921504606846977, "text": " Doña Ana "}'.encode()
    second = b'{"id": 2, "text": null}'
    path.write_bytes(b'{"a": ' + first + b',\r\n "b": ' + second + b"}\r\n")
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 3)
    source = JsonObjectRowSource(path, Row)
    with source.open_rows() as rows:
        entries = list(rows)
    assert [entry.location.key for entry in entries] == ["a", "b"]
    assert entries[0].row.identifier == 2**60 + 1
    assert entries[0].row.text == " Doña Ana "
    assert entries[1].row.text is None
    for entry, raw in zip(entries, (first, second), strict=True):
        location = entry.location
        assert path.read_bytes()[location.offset : location.offset + location.length] == raw
        assert source.read_row(location) == entry


@pytest.mark.parametrize(
    "text,message",
    [
        ('{"a": {"id": 1, "text": "x"}, "a": {"id": 2, "text": "y"}}', "duplicate key 'a'"),
        ('{"a": {"id": 1, "id": 2, "text": "x"}}', "duplicate key 'id'"),
        ('{"a": {"id": "1", "text": "x"}}', r"key 'a'.*: id:"),
        ('{"a": [1, 2]}', r"key 'a'.*: row:"),
        ('{"a": {"id": 1, "text": "x"}} []', "unexpected content"),
        ("[]", "expected '{'"),
        ('{"a": {"id": 1, "text": "x"}', "expected ','"),
    ],
)
def test_invalid_documents_and_rows_include_path_and_context(tmp_path, text, message):
    path = tmp_path / "rows.json"
    path.write_text(text)
    with (
        JsonObjectRowSource(path, Row).open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=message) as error,
    ):
        list(rows)
    assert str(path) in str(error.value)
    assert error.value.__cause__ is not None


def test_uniqueness_names_both_keys(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"first": {"id": 1, "text": "a"}, "second": {"id": 1, "text": "b"}}))
    with (
        JsonObjectRowSource(path, Row, unique="identifier").open_rows() as rows,
        pytest.raises(TimeNetConnectorError, match=r"key 'second'.*duplicate 1; first at .*key 'first'"),
    ):
        list(rows)


def test_reread_revalidates_changed_values_and_reports_truncation(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('{"a": {"id": 1, "text": "x"}}')
    source = JsonObjectRowSource(path, Row)
    with source.open_rows() as rows:
        entry = next(rows)
    path.write_text('{"a": {"id": "", "text":"x"}}')
    with pytest.raises(TimeNetConnectorError, match=r"key 'a'.*: id:"):
        source.read_row(entry.location)
    path.write_text('{"a": {')
    with pytest.raises(TimeNetConnectorError, match="ends after"):
        source.read_row(entry.location)


def test_reread_is_bounded_and_rejects_locations_from_another_source(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('{"a": {"id": 1, "text": "x"}, "b": invalid}')
    source = JsonObjectRowSource(path, Row)
    with source.open_rows() as rows:
        entry = next(rows)
    # Rereading a retained value does not parse unrelated trailing content.
    assert source.read_row(entry.location) == entry
    with pytest.raises(TimeNetConnectorError, match="expected a location"):
        source.read_row(replace(entry.location, path=tmp_path / "other.json"))
    for location in (replace(entry.location, offset=-1), replace(entry.location, length=0)):
        with pytest.raises(TimeNetConnectorError, match="nonnegative offset and positive length"):
            source.read_row(location)


def test_empty_object_yields_no_rows(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text("{}")
    with JsonObjectRowSource(path, Row).open_rows() as rows:
        assert list(rows) == []
