"""Stream large JSON documents entry by entry, keeping each value's byte range.

Some sources publish one JSON document of hundreds of megabytes instead of JSON Lines. Decoding it
whole holds the entire corpus in memory. Instead, decode one value at a time, record its UTF-8 byte
range, and let a lazy loader reread just that range with :func:`read_entry`.
"""

import codecs
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import re
from typing import Any, BinaryIO, Self

from timenet.errors import TimeFFormatError, TimeFValidationError


_CHUNK_SIZE = 65536
_NON_WHITESPACE = re.compile(r"[^ \t\r\n]")
_NOT_NUMBER = re.compile(r"[^-+.0-9eE]")


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
            raise TimeFValidationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


class JsonCursor:
    """Read JSON syntax incrementally while tracking consumed UTF-8 bytes.

    The cursor reads raw bytes and decodes UTF-8 itself, so its offsets always count the file's own
    bytes, CRLF included. Dataset-specific key and structure validation belongs to the caller.
    """

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.buffer = ""
        # Characters before ``position`` are consumed. They are dropped only when the buffer grows, so
        # consuming a value does not copy the rest of the buffer.
        self.position = 0
        self.offset = 0
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

    def consume(self, length: int) -> None:
        """Consume ``length`` buffered characters, advancing the byte offset past them."""
        end = self.position + length
        self.offset += len(self.buffer[self.position : end].encode("utf-8"))
        self.position = end

    def read(self, size: int) -> bool:
        """Append up to ``size`` more bytes of the file to the buffer.

        Returns:
            False at the end of the file.
        """
        data = self.handle.read(size)
        if not data:
            # Reject a multibyte character that the end of the file cuts short.
            self.utf8.decode(b"", final=True)
            return False
        self.buffer = self.buffer[self.position :] + self.utf8.decode(data)
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
            JSONDecodeError: If the document ends before the value is complete or is not valid JSON.
        """
        self.peek()
        start = self.offset
        # Each retry re-parses the value from its start, so double the read size to keep a value
        # much larger than one chunk linear.
        size = _CHUNK_SIZE
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if not self.read(size):
                    raise
            else:
                # A number cut by the buffer's edge decodes as its prefix: "12" of "12345", or "-6.5"
                # of "-6.5e3". Only a number starts with number characters, so a run of them that
                # reaches the edge can still continue, unless the file ends there.
                if _NOT_NUMBER.search(self.buffer, self.position) or not self.read(size):
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
            key, _, _ = self.value()
            if not isinstance(key, str) or key in seen:
                raise TimeFValidationError(f"expected a distinct string key at byte {self.offset}")
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
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def read_entry(path: Path, offset: int, length: int) -> Any:
    """Reread one value at the byte range a traversal recorded.

    Decodes as the traversal does, so a reread also rejects duplicate fields.

    Returns:
        The decoded value.

    Raises:
        TimeFFormatError: If the file changed since the traversal, so the range cannot be read,
            ends early, or does not decode as one value with distinct fields.
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
    try:
        return json.loads(data, object_pairs_hook=_distinct_fields)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFFormatError(f"{context}: {exc}") from exc
