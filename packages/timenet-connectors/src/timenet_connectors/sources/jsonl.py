"""Read typed UTF-8 JSONL rows with byte locations for contextual errors and lazy rereads."""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Generic

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources._json import distinct_fields
from timenet_connectors.sources.rows import RowT, validate_row


@dataclass(frozen=True)
class JsonlLocation:
    """A physical line's location; offsets and lengths count bytes, including its line ending."""

    path: Path
    line: int
    offset: int
    length: int

    @property
    def context(self) -> str:
        """The file and one-based line number for diagnostics."""
        return f"{self.path} (line {self.line})"


@dataclass(frozen=True)
class JsonlRow(Generic[RowT]):
    """One validated row with its location and original bytes, available for caller-owned hashing."""

    value: RowT
    location: JsonlLocation
    raw: bytes


def _decode(raw: bytes, location: JsonlLocation, model: type[RowT]) -> JsonlRow[RowT]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=distinct_fields)
    except (ValueError, RecursionError, TimeNetConnectorError) as exc:
        raise TimeNetConnectorError(f"{location.context}: {exc}") from exc
    if not isinstance(value, dict):
        raise TimeNetConnectorError(f"{location.context}: expected a row object")
    return JsonlRow(validate_row(value, model, location=location.context), location, raw)


def iter_jsonl_rows(path: Path, model: type[RowT]) -> Iterator[JsonlRow[RowT]]:
    """Yield typed rows without skipping blank lines or repairing malformed input.

    An empty file yields no rows. Missing final newlines and CRLF are supported.
    Duplicate keys are rejected at every nesting level. Numeric constraints belong
    to the row model or the connector's signal validation.

    Yields:
        Each validated row with its physical location and original bytes.

    Raises:
        TimeNetConnectorError: If a file cannot be read or a row is invalid.
    """
    try:
        with path.open("rb") as handle:
            line = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    return
                line += 1
                yield _decode(raw, JsonlLocation(path, line, offset, len(raw)), model)
    except OSError as exc:
        raise TimeNetConnectorError(f"{path}: cannot read source file: {exc}") from exc


def read_jsonl_row(location: JsonlLocation, model: type[RowT]) -> JsonlRow[RowT]:
    """Return a typed row reread from its recorded byte range.

    This uses the same decoding and validation as iteration. Callers own content
    comparison and whole-file change detection; a location is not a file snapshot.

    Raises:
        TimeNetConnectorError: If the range cannot be read completely or its row is invalid.
    """
    try:
        with location.path.open("rb") as handle:
            handle.seek(location.offset)
            raw = handle.read(location.length)
    except OSError as exc:
        raise TimeNetConnectorError(f"{location.context}: cannot reread source file: {exc}") from exc
    if len(raw) != location.length:
        raise TimeNetConnectorError(f"{location.context}: source row was truncated")
    return _decode(raw, location, model)
