"""Stream large JSON documents entry by entry, keeping each value's byte range.

Some sources publish one JSON document of hundreds of megabytes instead of JSON Lines. Decoding it
whole holds the entire corpus in memory. Instead, decode one value at a time, record its UTF-8 byte
range, and let a lazy loader reread just that range with :meth:`JsonObjectRowSource.read_row`.
"""

import codecs
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
from typing import Any, BinaryIO, Self

from timenet.errors import TimeNetConnectorError
from timenet.json import JsonValue
from timenet_connectors.sources.readers._json import distinct_fields
from timenet_connectors.sources.readers.base import LocatedRow, RowSource, RowT


_CHUNK_SIZE = 65536
_NON_WHITESPACE = re.compile(r"[^ \t\r\n]")
_NOT_NUMBER = re.compile(r"[^-+.0-9eE]")
# A value cut short by the buffer's edge fails within this many characters of it, at most at the
# start of a token prefix such as "-Infinit" or "\u12". An unterminated string fails at its opening
# quote instead, under a message from the json module that starts with this prefix; if its wording
# ever changes, test_a_string_split_across_reads_decodes_whole fails.
_EDGE = 16
_UNTERMINATED = "Unterminated string"


class JsonCursor:
    """Read JSON syntax incrementally while tracking consumed UTF-8 bytes.

    The cursor reads raw bytes and decodes UTF-8 itself, so its offsets always count the file's own
    bytes, CRLF included. Dataset-specific key and structure validation belongs to the caller.
    """

    def __init__(self, handle: BinaryIO, offset: int = 0, *, path: Path | None = None) -> None:
        self.handle = handle
        self.path = path
        self.buffer = ""
        # Characters before ``position`` are consumed. They are dropped only when the buffer grows, so
        # consuming a value does not copy the rest of the buffer.
        self.position = 0
        # The file's byte at the handle's start, so offsets name bytes in the file even when the
        # handle holds only part of it.
        self.offset = offset
        self.decoder = json.JSONDecoder(object_pairs_hook=distinct_fields)
        self.utf8 = codecs.getincrementaldecoder("utf-8")()

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Generator[Self]:
        """Open a file for reading from its first byte.

        Yields:
            A cursor at the start of the file.

        Raises:
            TimeNetConnectorError: If the source cannot be opened.
        """
        try:
            handle = path.open("rb")
        except OSError as exc:
            raise TimeNetConnectorError(f"{path}: {exc}") from exc
        with handle:
            yield cls(handle, path=path)

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
            TimeNetConnectorError: If the bytes are not valid UTF-8.
        """
        data = self.handle.read(size)
        try:
            # At the end of the file, reject a multibyte character that it cuts short.
            text = self.utf8.decode(data, final=not data)
        except UnicodeDecodeError as exc:
            # The decoder reports a position in its pending bytes followed by this read's, and those
            # pending bytes follow the ones decoded into the buffer.
            byte = self.offset + self._byte_length(len(self.buffer)) + exc.start
            raise TimeNetConnectorError(f"invalid UTF-8 at byte {byte}: {exc.reason}") from exc
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
            TimeNetConnectorError: If the next character is different.
        """
        if self.peek() != token:
            raise TimeNetConnectorError(f"expected {token!r} at byte {self.offset}")
        self.consume(1)

    def expect_end(self) -> None:
        """Check that nothing but whitespace follows the document.

        Raises:
            TimeNetConnectorError: If content follows the document.
        """
        if self.peek():
            raise TimeNetConnectorError("unexpected content after the JSON document")

    def value(self) -> tuple[Any, int, int]:
        """Decode the next complete value.

        Returns:
            The value, and its UTF-8 byte offset and length.

        Raises:
            TimeNetConnectorError: If the value is not valid UTF-8 or JSON, with the error's byte
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
            except (ValueError, RecursionError, TimeNetConnectorError) as exc:
                # A repeated key from the object hook, an integer too long to convert, or nesting
                # too deep.
                raise TimeNetConnectorError(f"{exc} in the value at byte {start}") from exc
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
                    raise TimeNetConnectorError(f"{message} at byte {byte}") from error
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
            TimeNetConnectorError: If object syntax or keys are invalid.
        """
        self.expect("{")
        seen: set[str] = set()
        if self.peek() == "}":
            self.consume(1)
            return
        while True:
            # Check before decoding, so a non-string key fails without reading it whole.
            if self.peek() != '"':
                raise TimeNetConnectorError(f"expected a string key at byte {self.offset}")
            key, key_offset, _ = self.value()
            if key in seen:
                raise TimeNetConnectorError(f"duplicate key {key!r} at byte {key_offset}")
            seen.add(key)
            self.expect(":")
            yield key
            if self.peek() == "}":
                self.consume(1)
                return
            self.expect(",")

    def object_entries(self) -> Iterator["JsonObjectEntry"]:
        """Consume the object at the cursor, retaining keys and exact value byte ranges.

        The object may be nested inside a larger document. The caller owns any whole-document
        end check. File-backed cursors retain the path for malformed-object diagnostics.

        Yields:
            Each decoded member; values may be arbitrary JSON before row-schema validation.

        Raises:
            TimeNetConnectorError: If object traversal fails, with the source path when known.
        """
        source_context = str(self.path) if self.path is not None else "JSON input"
        context = source_context
        try:
            for key in self.object_keys():
                context = f"{source_context}: key {key!r}"
                value, offset, length = self.value()
                yield JsonObjectEntry(key, value, offset, length)
                context = source_context
        except (OSError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{context}: {exc}") from exc

    @classmethod
    def read_range(cls, path: Path, offset: int, length: int) -> JsonValue:
        """Reread one value at the byte range a traversal recorded.

        Decodes with the traversal's own cursor, so a reread accepts exactly what the traversal did:
        strict UTF-8 without a byte order mark, and no duplicate fields. Errors name their byte in the
        file.

        Nothing records the file's identity, so a change that leaves a valid value in the range goes
        unnoticed. Pin the file's content, by checksum say, where a stale reread would matter.

        Returns:
            The decoded value.

        Raises:
            TimeNetConnectorError: If the range is invalid, unreadable, incomplete, or does not
                decode as one JSON value with distinct fields.
        """
        context = f"{path} (bytes {offset}-{offset + length})"
        if offset < 0 or length <= 0:
            raise TimeNetConnectorError(f"{context}: expected a nonnegative offset and positive length")
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(length)
        except OSError as exc:
            raise TimeNetConnectorError(f"{context}: {exc}") from exc
        if len(data) != length:
            raise TimeNetConnectorError(f"{context}: the file ends after {len(data)} bytes of the range")
        cursor = cls(io.BytesIO(data), offset)
        try:
            # Buffer the validated range up front so decoding is limited to this value's bytes.
            cursor.read(length)
            value, _, _ = cursor.value()
            cursor.expect_end()
        except TimeNetConnectorError as exc:
            raise TimeNetConnectorError(f"{context}: {exc}") from exc
        return value


@dataclass(frozen=True)
class JsonObjectEntry:
    """A decoded JSON member and its exact byte range, before row validation."""

    key: str
    value: JsonValue
    offset: int
    length: int


@dataclass(frozen=True)
class JsonObjectLocation:
    """A file, object key, and UTF-8 byte range of the member's value."""

    path: Path
    key: str
    offset: int
    length: int

    @property
    def context(self) -> str:
        """The source key and byte range for diagnostics."""
        return f"{self.path} (key {self.key!r}, bytes {self.offset}-{self.offset + self.length})"


@dataclass(frozen=True)
class JsonObjectRowSource(RowSource[RowT, JsonObjectLocation]):
    """Stream a JSON object's values as typed rows, keeping their keys and byte ranges.

    The cursor retains one decoded member at a time and a set of encountered keys to reject
    duplicates. Optional field uniqueness additionally retains each distinct value's location.
    Arbitrary JSON values remain accessible through JsonCursor; this interface requires each
    value to satisfy the connector's DataRow schema.
    """

    def _open_rows(self, resources: ExitStack) -> Generator[LocatedRow[RowT, JsonObjectLocation], None, None]:
        cursor = resources.enter_context(JsonCursor.open(self.path))
        return self._iter_rows(cursor)

    def _iter_rows(self, cursor: JsonCursor) -> Generator[LocatedRow[RowT, JsonObjectLocation], None, None]:
        for decoded in cursor.object_entries():
            location = JsonObjectLocation(self.path, decoded.key, decoded.offset, decoded.length)
            row = self.model.from_decoded(decoded.value, context=location.context)
            yield LocatedRow(row, location)
        try:
            cursor.expect_end()
        except (OSError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc

    def read_row(self, location: JsonObjectLocation) -> LocatedRow[RowT, JsonObjectLocation]:
        """Reread and revalidate a scanned value without loading the surrounding document.

        Locations do not fingerprint content. Pin or checksum the source when detecting a changed
        but still valid value matters. The key is retained provenance, not re-read from the range.

        Returns:
            The validated row at the supplied location.

        Raises:
            TimeNetConnectorError: If the location names another path or an invalid byte range,
                or the range cannot be decoded or violates the row schema.
        """
        if location.path != self.path:
            raise TimeNetConnectorError(f"{location.context}: expected a location in {self.path}")
        values = JsonCursor.read_range(self.path, location.offset, location.length)
        row = self.model.from_decoded(values, context=location.context)
        return LocatedRow(row, location)
