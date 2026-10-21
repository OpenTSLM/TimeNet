"""Traverse nested split/key JSON objects one entry at a time.

The source files are not JSONL. Retain each entry's UTF-8 byte range for direct lazy signal reads
without keeping the whole decoded corpus in memory.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from timenet.errors import TimeFValidationError
from timenet_connectors.sources.json_stream import JsonCursor


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
            cursor = JsonCursor(handle)
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
            cursor.expect_end()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{context}: {exc}") from exc
