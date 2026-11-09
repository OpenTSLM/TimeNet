"""Typed JSONL streaming and byte-addressed rereads."""

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.jsonl import iter_jsonl_rows, read_jsonl_row
from timenet_connectors.sources.rows import NonBlankText, SourceRow


class Row(SourceRow):
    name: NonBlankText
    count: int


def test_utf8_crlf_offsets_and_rereads(tmp_path):
    path = tmp_path / "rows.jsonl"
    first = '{"name":"café", "count":1}\r\n'.encode()
    last = b'{"name":"two", "count":2}'
    path.write_bytes(first + last)
    rows = list(iter_jsonl_rows(path, Row))
    assert [row.value.name for row in rows] == ["café", "two"]
    assert [row.location.line for row in rows] == [1, 2]
    assert [row.location.offset for row in rows] == [0, len(first)]
    assert [row.raw for row in rows] == [first, last]
    assert [read_jsonl_row(row.location, Row) for row in rows] == rows


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\n", "Expecting value"),
        (b"{bad}", "Expecting property name"),
        (b"[]", "expected a row object"),
        (b'{"name":"x","count":true}', "count:"),
        (b'{"name":"x","count":"2"}', "count:"),
        (b'{"name":"x","count":1,"count":2}', "duplicate key 'count'"),
        (b'{"name":"x","count":1,"extra":{"a":1,"a":2}}', "duplicate key 'a'"),
        (b'{"name":"\xff","count":1}', "utf-8"),
    ],
)
def test_invalid_rows_have_context_on_iteration_and_reread(tmp_path, raw, message):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"name":"good","count":1}\n' + raw)
    rows = iter_jsonl_rows(path, Row)
    first = next(rows)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        next(rows)
    assert f"{path} (line 2)" in str(error.value)
    path.write_bytes(raw)
    location = type(first.location)(path, 1, 0, len(raw))
    with pytest.raises(TimeNetConnectorError, match=message):
        read_jsonl_row(location, Row)


def test_empty_missing_and_truncated_files(tmp_path):
    path = tmp_path / "rows.jsonl"
    with pytest.raises(TimeNetConnectorError, match="cannot read source file"):
        list(iter_jsonl_rows(path, Row))
    path.write_bytes(b"")
    assert list(iter_jsonl_rows(path, Row)) == []
    path.write_bytes(b'{"name":"x","count":1}\n')
    row = next(iter_jsonl_rows(path, Row))
    path.write_bytes(b"{")
    with pytest.raises(TimeNetConnectorError, match=r"line 1.*truncated"):
        read_jsonl_row(row.location, Row)
