"""Stream large JSON documents entry by entry, keeping each value's byte range.

Some sources publish one JSON document of hundreds of megabytes instead of JSON Lines. Decoding it
whole holds the entire corpus in memory. Instead, decode one value at a time, record its UTF-8 byte
range, and let a lazy loader reread just that range with :func:`read_entry`.
"""

import codecs
from collections.abc import Iterator
from contextlib import contextmanager
import io
import json
from pathlib import Path
import re
from typing import Any, BinaryIO, Self

from timenet.errors import TimeFFormatError, TimeFValidationError


_CHUNK_SIZE = 65536
_NON_WHITESPACE = re.compile(r"[^ \t\r\n]")
_NOT_NUMBER = re.compile(r"[^-+.0-9eE]")
# A value cut short by the buffer's edge fails within this many characters of it, at most at the
# start of a token prefix such as "-Infinit" or "\u12". An unterminated string fails at its opening
# quote instead, under a message from the json module that starts with this prefix; if its wording
# ever changes, test_a_string_split_across_reads_decodes_whole fails.
_EDGE = 16
_UNTERMINATED = "Unterminated string"


def _distinct_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated fields instead of silently overwriting source data.

    Returns:
        The decoded object in source order.

    Raises:
        TimeFValidationError: If an object repeats a field name.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TimeFValidationError(f"duplicate key {key!r}")
        result[key] = value
    return result


class JsonCursor:
    """Read JSON syntax incrementally while tracking consumed UTF-8 bytes.

    The cursor reads raw bytes and decodes UTF-8 itself, so its offsets always count the file's own
    bytes, CRLF included. Dataset-specific key and structure validation belongs to the caller.
    """

    def __init__(self, handle: BinaryIO, offset: int = 0) -> None:
        self.handle = handle
        self.buffer = ""
        # Characters before ``position`` are consumed. They are dropped only when the buffer grows, so
        # consuming a value does not copy the rest of the buffer.
        self.position = 0
        # The file's byte at the handle's start, so offsets name bytes in the file even when the
        # handle holds only part of it.
        self.offset = offset
        self.decoder = json.JSONDecoder(object_pairs_hook=_distinct_fields)
        self.utf8 = codecs.getincrementaldecoder("utf-8")()

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator[Self]:
        """Open a file for reading from its first byte.

        Yields:
            A cursor at the start of the file.
        """
        with path.open("rb") as handle:
            yield cls(handle)

    def _byte_length(self, end: int) -> int:
        """Count the UTF-8 bytes of the unconsumed characters before ``end``.

        Returns:
            The byte count.
        """
        # CPython caches whether a string is ASCII, so this check is constant time, unlike checking
        # a slice, which copies it.
        if self.buffer.isascii():
            return end - self.position
        return len(self.buffer[self.position : end].encode("utf-8"))

    def consume(self, length: int) -> None:
        """Consume ``length`` buffered characters, advancing the byte offset past them."""
        end = self.position + length
        self.offset += self._byte_length(end)
        self.position = end

    def read(self, size: int) -> bool:
        """Append up to ``size`` more bytes of the file to the buffer.

        Returns:
            False at the end of the file.

        Raises:
            TimeFValidationError: If the bytes are not valid UTF-8.
        """
        data = self.handle.read(size)
        try:
            # At the end of the file, reject a multibyte character that it cuts short.
            text = self.utf8.decode(data, final=not data)
        except UnicodeDecodeError as exc:
            # The decoder reports a position in its pending bytes followed by this read's, and those
            # pending bytes follow the ones decoded into the buffer.
            byte = self.offset + self._byte_length(len(self.buffer)) + exc.start
            raise TimeFValidationError(f"invalid UTF-8 at byte {byte}: {exc.reason}") from exc
        if not data:
            return False
        self.buffer = self.buffer[self.position :] + text
        self.position = 0
        return True

    def peek(self) -> str:
        """Skip whitespace and return the next character, or ``""`` at the end of the document.

        Returns:
            The next non-whitespace character.
        """
        while True:
            match = _NON_WHITESPACE.search(self.buffer, self.position)
            if match:
                self.consume(match.start() - self.position)
                return match.group()
            self.consume(len(self.buffer) - self.position)
            if not self.read(_CHUNK_SIZE):
                return ""

    def expect(self, token: str) -> None:
        """Consume one expected structural character.

        Raises:
            TimeFValidationError: If the next character is different.
        """
        if self.peek() != token:
            raise TimeFValidationError(f"expected {token!r} at byte {self.offset}")
        self.consume(1)

    def expect_end(self) -> None:
        """Check that nothing but whitespace follows the document.

        Raises:
            TimeFValidationError: If content follows the document.
        """
        if self.peek():
            raise TimeFValidationError("unexpected content after the JSON document")

    def value(self) -> tuple[Any, int, int]:
        """Decode the next complete value.

        Returns:
            The value, and its UTF-8 byte offset and length.

        Raises:
            TimeFValidationError: If the value is not valid UTF-8 or JSON, with the error's byte
                offset in the message, or repeats a field.
        """
        self.peek()
        start = self.offset
        # Each retry re-parses the value from its start, so double the read size to keep a value
        # much larger than one chunk linear.
        size = _CHUNK_SIZE
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                error = exc
            except (ValueError, RecursionError) as exc:
                # A repeated key from the object hook, an integer too long to convert, or nesting
                # too deep.
                raise TimeFValidationError(f"{exc} in the value at byte {start}") from exc
            else:
                error = None
            if error is None:
                # A number cut by the buffer's edge decodes as its prefix: "12" of "12345", or "-6.5"
                # of "-6.5e3". Only a number starts with number characters, so a run of them that
                # reaches the edge can still continue, unless the file ends there.
                settled = _NOT_NUMBER.search(self.buffer, self.position) is not None
            else:
                # Read on only while the buffer's edge could explain the error. Any other error
                # stays put however much is read, and reading on would buffer the rest of the file
                # first. A string that never closes reads to the end of the file before it fails,
                # since only the end shows that no closing quote follows.
                settled = not error.msg.startswith(_UNTERMINATED) and error.pos < len(self.buffer) - _EDGE
            # Read outside the except clause, so a UTF-8 error is not chained to the decode error.
            if settled or not self.read(size):
                if error is not None:
                    byte = self.offset + self._byte_length(error.pos)
                    # Some messages, such as "Unterminated string starting at", already end in "at".
                    message = error.msg.removesuffix(" at")
                    raise TimeFValidationError(f"{message} at byte {byte}") from error
                self.consume(end - self.position)
                return value, start, self.offset - start
            size *= 2

    def object_keys(self) -> Iterator[str]:
        """Yield object keys, leaving each value for the caller to consume.

        The caller must consume the entire value before requesting the next key. This allows
        nested objects to be traversed without decoding them all at once.

        Yields:
            Distinct string keys, with the cursor positioned after each colon.

        Raises:
            TimeFValidationError: If object syntax or keys are invalid.
        """
        self.expect("{")
        seen: set[str] = set()
        if self.peek() == "}":
            self.consume(1)
            return
        while True:
            # Check before decoding, so a non-string key fails without reading it whole.
            if self.peek() != '"':
                raise TimeFValidationError(f"expected a string key at byte {self.offset}")
            key, key_offset, _ = self.value()
            if key in seen:
                raise TimeFValidationError(f"duplicate key {key!r} at byte {key_offset}")
            seen.add(key)
            self.expect(":")
            yield key
            if self.peek() == "}":
                self.consume(1)
                return
            self.expect(",")


def iter_object(path: Path) -> Iterator[tuple[str, Any, int, int]]:
    """Traverse a document that is one object, without retaining decoded values.

    Yields:
        Each key, its decoded value, and the value's UTF-8 byte offset and length.

    Raises:
        TimeFValidationError: If the document is not one object with distinct string keys, with the
            file and key in the message.
    """
    context = str(path)
    try:
        with JsonCursor.open(path) as cursor:
            for key in cursor.object_keys():
                context = f"{path} (key {key})"
                yield key, *cursor.value()
                context = str(path)
            cursor.expect_end()
    except TimeFValidationError as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def read_entry(path: Path, offset: int, length: int) -> Any:
    """Reread one value at the byte range a traversal recorded.

    Decodes with the traversal's own cursor, so a reread accepts exactly what the traversal did:
    strict UTF-8 without a byte order mark, and no duplicate fields. Errors name their byte in the
    file.

    Nothing records the file's identity, so a change that leaves a valid value in the range goes
    unnoticed. Pin the file's content, by checksum say, where a stale reread would matter.

    Returns:
        The decoded value.

    Raises:
        TimeFFormatError: If the range cannot be read, ends early, or does not decode as one value
            with distinct fields, as happens when the file changed since the traversal.
    """
    context = f"{path} (bytes {offset}-{offset + length})"
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
    except OSError as exc:
        raise TimeFFormatError(f"{context}: {exc}") from exc
    if len(data) != length:
        raise TimeFFormatError(f"{context}: the file ends after {len(data)} bytes of the range")
    cursor = JsonCursor(io.BytesIO(data), offset)
    try:
        # Buffer the whole range up front, so the value decodes in one pass. An empty range reads
        # nothing, and value() reports it.
        cursor.read(length)
        value, _, _ = cursor.value()
        cursor.expect_end()
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"{context}: {exc}") from exc
    return value
