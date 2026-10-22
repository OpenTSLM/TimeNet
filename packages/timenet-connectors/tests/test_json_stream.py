import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
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


def test_a_number_split_across_reads_decodes_whole(monkeypatch, tmp_path):
    text = '{"a": 12345, "b": -6.5e3, "c": 7}'
    path = tmp_path / "rows.json"
    path.write_text(text)
    # Some chunk size cuts each number short at the buffer's edge.
    for size in range(1, len(text) + 1):
        monkeypatch.setattr(json_stream, "_CHUNK_SIZE", size)
        assert [(key, value) for key, value, _, _ in json_stream.iter_object(path)] == [
            ("a", 12345),
            ("b", -6500.0),
            ("c", 7),
        ], size


def test_a_value_larger_than_a_chunk_takes_few_reads(monkeypatch, tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('{"big": [' + "0, " * 100_000 + "0]}")
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 16)
    reads = []
    read = json_stream.JsonCursor.read
    monkeypatch.setattr(json_stream.JsonCursor, "read", lambda self, size: reads.append(size) or read(self, size))

    [(_, value, _, _)] = json_stream.iter_object(path)

    assert value == [0] * 100_001
    # Doubling reaches the value's 300 KB in a few reads, where fixed 16-byte reads take 18,750.
    assert len(reads) < 30


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


@pytest.mark.parametrize(
    ("text", "length", "message"),
    [
        # The file was truncated since the traversal recorded the range.
        ('{"a": 1}', 20, "ends after 8 bytes"),
        ('{"a": 1, "a": 2}', 16, "duplicate JSON field 'a'"),
        ('{"a": 1', 7, "Expecting"),
        (b'{"a": "\xff"}', 10, "codec"),
    ],
)
def test_a_reread_that_does_not_decode_names_the_range(tmp_path, text, length, message):
    path = tmp_path / "changed.json"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    with pytest.raises(TimeFFormatError, match=message) as error:
        json_stream.read_entry(path, 0, length)
    assert f"{path} (bytes 0-{length})" in str(error.value)


def test_a_reread_of_a_missing_file_raises_a_format_error(tmp_path):
    with pytest.raises(TimeFFormatError, match="No such file"):
        json_stream.read_entry(tmp_path / "gone.json", 0, 2)
