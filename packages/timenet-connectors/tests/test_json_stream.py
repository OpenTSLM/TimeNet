import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.sources import json_stream


def test_entries_keep_byte_ranges_across_crlf_and_utf8(monkeypatch, tmp_path):
    path = tmp_path / "rows.json"
    path.write_bytes('{"1": {"name": "Doña Ana"},\r\n "2": [1, 2.5], "3": {}}\r\n'.encode())
    # A tiny chunk splits keys, values, and multibyte characters across reads.
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 3)

    entries = list(json_stream.iter_object(path))

    assert [(key, value) for key, value, _, _ in entries] == [("1", {"name": "Doña Ana"}), ("2", [1, 2.5]), ("3", {})]
    for key, value, offset, length in entries:
        assert json_stream.read_entry(path, offset, length) == value, key


def test_empty_object_yields_nothing(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(" {} \n")
    assert list(json_stream.iter_object(path)) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # json.load would silently keep only the last duplicate entry.
        ('{"12": {}, "12": {}}', "distinct string key"),
        ('{"12": {"a": 1, "a": 2}}', "duplicate JSON field 'a'"),
        ('{"12": {}} []', "unexpected content"),
        ("[]", "expected '{'"),
        ('{"12": {}', "expected ','"),
    ],
)
def test_malformed_documents_name_the_file(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeFValidationError, match=message) as error:
        list(json_stream.iter_object(path))
    assert str(path) in str(error.value)
