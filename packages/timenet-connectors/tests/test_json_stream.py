import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import json_object


def _entries(path):
    with json_object.JsonCursor.open(path) as cursor:
        for entry in cursor.object_entries():
            yield entry.key, entry.value, entry.offset, entry.length


@pytest.fixture
def reads(monkeypatch):
    """Record the size of every read the cursor makes."""
    sizes = []
    read = json_object.JsonCursor.read
    monkeypatch.setattr(json_object.JsonCursor, "read", lambda self, size: sizes.append(size) or read(self, size))
    return sizes


def test_entries_keep_byte_ranges_across_crlf_and_utf8(monkeypatch, tmp_path):
    path = tmp_path / "rows.json"
    path.write_bytes('{"1": {"name": "Doña Ana"},\r\n "2": [1, 2.5], "3": {}}\r\n'.encode())
    # A tiny chunk splits keys, values, and multibyte characters across reads.
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 3)

    entries = list(_entries(path))

    assert [(key, value) for key, value, _, _ in entries] == [("1", {"name": "Doña Ana"}), ("2", [1, 2.5]), ("3", {})]
    for key, value, offset, length in entries:
        assert json_object.JsonCursor.read_range(path, offset, length) == value, key


def test_a_number_split_across_reads_decodes_whole(monkeypatch, tmp_path):
    text = '{"a": 12345, "b": -6.5e3, "c": 7}'
    path = tmp_path / "rows.json"
    path.write_text(text)
    # Some chunk size cuts each number short at the buffer's edge.
    for size in range(1, len(text) + 1):
        monkeypatch.setattr(json_object, "_CHUNK_SIZE", size)
        assert [(key, value) for key, value, _, _ in _entries(path)] == [
            ("a", 12345),
            ("b", -6500.0),
            ("c", 7),
        ], size


def test_a_string_split_across_reads_decodes_whole(monkeypatch, tmp_path):
    text = '{"a": "Doña Ana, a string far longer than the edge", "b": 1}'
    path = tmp_path / "rows.json"
    path.write_text(text)
    # Small chunks cut the string short well before its closing quote.
    for size in range(1, len(text) + 1):
        monkeypatch.setattr(json_object, "_CHUNK_SIZE", size)
        assert [(key, value) for key, value, _, _ in _entries(path)] == [
            ("a", "Doña Ana, a string far longer than the edge"),
            ("b", 1),
        ], size


def test_a_value_larger_than_a_chunk_takes_few_reads(monkeypatch, tmp_path, reads):
    path = tmp_path / "rows.json"
    path.write_text('{"big": [' + "0, " * 100_000 + "0]}")
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 16)

    [(_, value, _, _)] = _entries(path)

    assert value == [0] * 100_001
    # Doubling reaches the value's 300 KB in a few reads, where fixed 16-byte reads take 18,750.
    assert len(reads) < 30


def test_a_syntax_error_raises_without_reading_the_rest_of_the_file(monkeypatch, tmp_path, reads):
    path = tmp_path / "rows.json"
    path.write_text('{"a": {"x": 1 x}, "b": [' + "0, " * 100_000 + "0]}")
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 64)

    with pytest.raises(TimeNetConnectorError, match="delimiter at byte 14"):
        list(_entries(path))
    assert sum(reads) <= 64


def test_a_non_string_key_raises_without_reading_it(monkeypatch, tmp_path, reads):
    path = tmp_path / "rows.json"
    path.write_text("{[" + "0, " * 100_000 + "0]: 1}")
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 64)

    with pytest.raises(TimeNetConnectorError, match="expected a string key at byte 1"):
        list(_entries(path))
    assert sum(reads) <= 64


def test_a_syntax_error_names_its_byte_in_the_file(monkeypatch, tmp_path):
    text = '{"é": 1, ' + ", ".join(f'"{n}": {n}' for n in range(1000)) + ', "z": [1 2]}'
    path = tmp_path / "rows.json"
    path.write_text(text)
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 100)

    with pytest.raises(TimeNetConnectorError) as error:
        list(_entries(path))
    # The "2" of [1 2], counted in bytes, where the two-byte "é" sits one past its character index.
    assert f"at byte {text.index('[1 2]') + 4}" in str(error.value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'{"a": "\xc3\xb1", "b": "\xc3"}', "invalid UTF-8 at byte 18"),
        # The end of the file cuts the character short.
        (b'{"a": "\xc3', "invalid UTF-8 at byte 7: unexpected end of data"),
    ],
)
def test_invalid_utf8_names_its_byte(monkeypatch, tmp_path, data, message):
    path = tmp_path / "rows.json"
    path.write_bytes(data)
    monkeypatch.setattr(json_object, "_CHUNK_SIZE", 3)
    with pytest.raises(TimeNetConnectorError, match=message):
        list(_entries(path))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[1, 2", "at byte 5"),
        ('["abc', "^Unterminated string starting at byte 1$"),
        # Python refuses to convert an integer this long.
        ("[" + "9" * 5000 + "]", "in the value at byte 0"),
        ("[" * 100_000 + "]" * 100_000, "in the value at byte 0"),
    ],
)
def test_a_cursor_raises_timenet_errors(tmp_path, text, message):
    path = tmp_path / "rows.json"
    path.write_text(text)
    with json_object.JsonCursor.open(path) as cursor, pytest.raises(TimeNetConnectorError, match=message):
        cursor.value()


def test_empty_object_yields_nothing(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(" {} \n")
    assert list(_entries(path)) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # json.load would silently keep only the last duplicate entry.
        # The byte where the repeated key starts.
        ('{"12": {}, "12": {}}', "duplicate key '12' at byte 11"),
        # The byte where the value holding the repeated field starts.
        ('{"12": {"a": 1, "a": 2}}', "duplicate key 'a' in the value at byte 7"),
        ("{12: {}}", "expected a string key at byte 1"),
        ("[]", "expected '{'"),
        ('{"12": {}', "expected ','"),
    ],
)
def test_malformed_objects_name_the_file_at_the_cursor(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        list(_entries(path))
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    ("text", "length", "message"),
    [
        # The file was truncated since the traversal recorded the range.
        ('{"a": 1}', 20, "ends after 8 bytes"),
        ('{"a": 1, "a": 2}', 16, "duplicate key 'a'"),
        ('{"a": 1', 7, "Expecting"),
        (b'{"a": "\xff"}', 10, "invalid UTF-8 at byte 7"),
        # The traversal rejects a byte order mark, so a reread does too.
        (b'\xef\xbb\xbf{"a": 1}', 11, "Expecting value at byte 0"),
    ],
)
def test_a_reread_that_does_not_decode_names_the_range(tmp_path, text, length, message):
    path = tmp_path / "changed.json"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        json_object.JsonCursor.read_range(path, 0, length)
    assert f"{path} (bytes 0-{length})" in str(error.value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # The "1" after "b", counted in bytes, where the two-byte "é" sits one past its character index.
        ('{"a": "é", "b" 1}', "Expecting ':' delimiter at byte 30"),
        (b'{"a": "\xc3\xa9", "b": "\xff"}', "invalid UTF-8 at byte 32"),
    ],
)
def test_a_reread_error_names_its_byte_in_the_file(tmp_path, value, message):
    prefix = b'{"x": 0, "y": '
    data = value if isinstance(value, bytes) else value.encode()
    path = tmp_path / "changed.json"
    path.write_bytes(prefix + data + b"}")
    with pytest.raises(TimeNetConnectorError, match=message):
        json_object.JsonCursor.read_range(path, len(prefix), len(data))


def test_a_reread_of_a_missing_file_raises_a_connector_error(tmp_path):
    with pytest.raises(TimeNetConnectorError, match="No such file"):
        json_object.JsonCursor.read_range(tmp_path / "gone.json", 0, 2)


def test_object_entries_can_consume_a_nested_object(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text('{"outer": {"a": 1, "b": 2}, "tail": 3}')
    with json_object.JsonCursor.open(path) as cursor:
        outer_keys = cursor.object_keys()
        assert next(outer_keys) == "outer"
        assert [(entry.key, entry.value) for entry in cursor.object_entries()] == [("a", 1), ("b", 2)]
        assert next(outer_keys) == "tail"
        assert cursor.value()[0] == 3
        assert list(outer_keys) == []
        cursor.expect_end()


@pytest.mark.parametrize("offset,length", [(-1, 1), (0, 0), (0, -1)])
def test_invalid_ranges_fail_before_file_access(tmp_path, monkeypatch, offset, length):
    def unexpected_open(*args, **kwargs):
        pytest.fail("invalid byte ranges must fail before file access")

    monkeypatch.setattr(type(tmp_path), "open", unexpected_open)
    with pytest.raises(TimeNetConnectorError, match="nonnegative offset and positive length"):
        json_object.JsonCursor.read_range(tmp_path / "missing.json", offset, length)
