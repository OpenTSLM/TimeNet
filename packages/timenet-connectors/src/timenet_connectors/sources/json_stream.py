"""Stream large JSON documents entry by entry, keeping each value's byte range.

Some sources publish one JSON document of hundreds of megabytes instead of JSON Lines. Decoding it
whole holds the entire corpus in memory. Instead, decode one value at a time, record its UTF-8 byte
range, and let a lazy loader reread just that range with :func:`read_entry`.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any, TextIO

from timenet.errors import TimeFValidationError


_CHUNK_SIZE = 65536


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

    Open the handle with ``encoding="utf-8", newline=""``: offsets must refer to the original bytes,
    including CRLF. Dataset-specific key and structure validation belongs to the caller.
    """

    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.buffer = ""
        self.offset = 0
        self.decoder = json.JSONDecoder(object_pairs_hook=_distinct_fields)

    def consume(self, length: int) -> None:
        """Drop ``length`` buffered characters, advancing the byte offset past them."""
        self.offset += len(self.buffer[:length].encode("utf-8"))
        self.buffer = self.buffer[length:]

    def peek(self) -> str:
        """Skip whitespace and return the next character, or ``""`` at the end of the document.

        Returns:
            The next non-whitespace character.
        """
        while True:
            self.consume(len(self.buffer) - len(self.buffer.lstrip(" \t\r\n")))
            if self.buffer:
                return self.buffer[0]
            chunk = self.handle.read(_CHUNK_SIZE)
            if not chunk:
                return ""
            self.buffer += chunk

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
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                chunk = self.handle.read(_CHUNK_SIZE)
                if not chunk:
                    raise
                self.buffer += chunk
            else:
                self.consume(end)
                return value, start, self.offset - start

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
        with path.open(encoding="utf-8", newline="") as handle:
            cursor = JsonCursor(handle)
            for key in cursor.object_keys():
                context = f"{path} (key {key})"
                yield key, *cursor.value()
                context = str(path)
            cursor.expect_end()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc


def read_entry(path: Path, offset: int, length: int) -> Any:
    """Reread one value at the byte range a traversal recorded.

    Returns:
        The decoded value.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.read(length))
