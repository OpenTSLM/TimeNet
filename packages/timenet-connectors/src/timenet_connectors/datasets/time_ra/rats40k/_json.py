"""Traverse nested split/key JSON objects one entry at a time.

The source files are not JSONL. Retain each entry's UTF-8 byte range for direct lazy signal reads
without keeping the whole decoded corpus in memory.
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


class _Cursor:
    """Read JSON syntax incrementally while tracking consumed UTF-8 bytes.

    Dataset-specific split and row-key validation belongs in ``iter_entries``.
    """

    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.buffer = ""
        self.offset = 0
        self.decoder = json.JSONDecoder(object_pairs_hook=_distinct_fields)

    def consume(self, length: int) -> None:
        self.offset += len(self.buffer[:length].encode("utf-8"))
        self.buffer = self.buffer[length:]

    def peek(self) -> str:
        while True:
            self.consume(len(self.buffer) - len(self.buffer.lstrip(" \t\r\n")))
            if self.buffer:
                return self.buffer[0]
            chunk = self.handle.read(_CHUNK_SIZE)
            if not chunk:
                return ""
            self.buffer += chunk

    def expect(self, token: str) -> None:
        if self.peek() != token:
            raise TimeFValidationError(f"expected {token!r} at byte {self.offset}")
        self.consume(1)

    def value(self) -> tuple[Any, int, int]:
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


def iter_entries(path: Path) -> Iterator[tuple[str, str, Any, int, int]]:
    """Traverse split/key objects without retaining decoded rows.

    Yields:
        Split, original key, decoded row, and its UTF-8 byte offset and length.

    Raises:
        TimeFValidationError: If the document or split structure is malformed.
    """
    context = str(path)
    try:
        # Disable newline translation: offsets must refer to the original bytes, including CRLF.
        with path.open(encoding="utf-8", newline="") as handle:
            cursor = _Cursor(handle)
            splits: set[str] = set()
            for split in cursor.object_keys():
                if split not in {"TSAD_train", "TSAD_test"}:
                    raise TimeFValidationError(f"unknown split {split!r}")
                splits.add(split)
                context = f"{path} ({split})"
                for key in cursor.object_keys():
                    context = f"{path} ({split}, key {key})"
                    if not key.isascii() or not key.isdecimal():
                        raise TimeFValidationError(f"{split}: expected a decimal row key, got {key!r}")
                    row, offset, length = cursor.value()
                    yield split, key, row, offset, length
                    context = f"{path} ({split})"
                context = str(path)
            if splits != {"TSAD_train", "TSAD_test"}:
                raise TimeFValidationError("expected TSAD_train and TSAD_test")
            if cursor.peek():
                raise TimeFValidationError("unexpected content after the JSON document")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc
