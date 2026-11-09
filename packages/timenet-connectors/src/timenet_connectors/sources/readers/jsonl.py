"""Stream typed UTF-8 JSONL rows with physical lines and reusable byte locations."""

from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from typing import BinaryIO

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers._json import distinct_fields
from timenet_connectors.sources.readers.base import LocatedRow, RowSource, RowT


@dataclass(frozen=True)
class JsonlLocation:
    """A one-based physical line and its byte range, including any line ending."""

    path: Path
    line: int
    offset: int
    length: int

    @property
    def context(self) -> str:
        """The file and physical line for diagnostics."""
        return f"{self.path} (line {self.line})"


@dataclass(frozen=True)
class JsonlRowSource(RowSource[RowT, JsonlLocation]):
    """Read one row object per line without skipping blanks or repairing input.

    Empty files and missing final newlines are supported. Binary reads preserve UTF-8 byte
    offsets and CRLF line endings. Duplicate keys are rejected at every nesting level; decoded
    values use the connector's strict row schema. Use read_bytes for original byte access and
    connector-owned fingerprints. Locations do not snapshot files or detect valid changes.
    """

    def _open_rows(self, resources: ExitStack) -> Generator[LocatedRow[RowT, JsonlLocation], None, None]:
        try:
            handle = resources.enter_context(self.path.open("rb"))
        except OSError as exc:
            raise TimeNetConnectorError(f"{self.path}: cannot read source file: {exc}") from exc
        return self._iter_rows(handle)

    def _iter_rows(self, handle: BinaryIO) -> Generator[LocatedRow[RowT, JsonlLocation], None, None]:
        line = 0
        while True:
            try:
                offset = handle.tell()
                raw = handle.readline()
            except OSError as exc:
                raise TimeNetConnectorError(f"{self.path} (line {line + 1}): {exc}") from exc
            if not raw:
                return
            line += 1
            location = JsonlLocation(self.path, line, offset, len(raw))
            yield self._decode(raw, location)

    def _decode(self, raw: bytes, location: JsonlLocation) -> LocatedRow[RowT, JsonlLocation]:
        try:
            values = json.loads(raw.decode("utf-8"), object_pairs_hook=distinct_fields)
        except (ValueError, RecursionError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{location.context}: {exc}") from exc
        if not isinstance(values, dict):
            raise TimeNetConnectorError(f"{location.context}: expected a row object")
        row = self.model.from_decoded(values, context=location.context)
        return LocatedRow(row, location)

    def read_bytes(self, location: JsonlLocation) -> bytes:
        """Read an original line's bytes, including its line ending, from its recorded range.

        Bytes reflect the file at read time. Callers own content comparison and source fingerprints;
        retain the returned bytes or a digest when that identity must survive later source changes.

        Returns:
            The complete recorded byte range without decoding or normalization.

        Raises:
            TimeNetConnectorError: If the location is invalid, unreadable, or truncated.
        """
        if location.path != self.path or location.line < 1 or location.offset < 0 or location.length <= 0:
            raise TimeNetConnectorError(f"{location.context}: expected a positive line and byte range in {self.path}")
        try:
            with self.path.open("rb") as handle:
                handle.seek(location.offset)
                raw = handle.read(location.length)
        except OSError as exc:
            raise TimeNetConnectorError(f"{location.context}: cannot reread source file: {exc}") from exc
        if len(raw) != location.length:
            raise TimeNetConnectorError(f"{location.context}: source row was truncated")
        return raw

    def read_row(self, location: JsonlLocation) -> LocatedRow[RowT, JsonlLocation]:
        """Reread and validate a line through the same path as a streaming scan.

        Returns:
            The typed row paired with its original location. Uniqueness is a scan-wide constraint
            and is not checked for an isolated reread.
        """
        return self._decode(self.read_bytes(location), location)
