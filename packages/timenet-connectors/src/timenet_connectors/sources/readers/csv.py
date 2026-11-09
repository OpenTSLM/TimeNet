"""Stream small CSV metadata tables through the shared typed-row contract."""

from collections import Counter
from collections.abc import Generator
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import LocatedRow, RowSource, RowT


@dataclass(frozen=True)
class CsvLocation:
    """A file and the one-based physical starting line of a CSV row."""

    path: Path
    line: int

    @property
    def context(self) -> str:
        """The file and physical line for diagnostics."""
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class CsvRowSource(RowSource[RowT, CsvLocation]):
    """Read a UTF-8 CSV table using a connector's declared row schema.

    Cells retain their text; blank cells are not implicitly null. Missing optional columns use
    model defaults. ``unique`` names a model field, not its alias, and compares validated values.
    Enable ``skip_repeated_header`` only for sources that publish exact embedded header repeats.
    Each read retains only its current row and, if requested, an index of unique values.
    """

    skip_repeated_header: bool = False

    def _open_rows(self, resources: ExitStack) -> Generator[LocatedRow[RowT, CsvLocation], None, None]:
        try:
            handle = resources.enter_context(self.path.open(newline="", encoding="utf-8"))
        except OSError as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
        return self._iter_rows(handle)

    def _check_header(self, header: list[str]) -> None:
        if not header:
            raise TimeNetConnectorError(f"{self.path}: no header row")
        repeated = [name for name, count in Counter(header).items() if count > 1]
        if repeated:
            raise TimeNetConnectorError(f"{self.path}: repeated columns {repeated}")
        self.model.check_columns(header, context=str(self.path))

    def _parse_row(self, header: list[str], cells: list[str], location: CsvLocation) -> LocatedRow[RowT, CsvLocation]:
        if not cells:
            raise TimeNetConnectorError(f"{location.context}: blank row")
        if len(cells) != len(header):
            raise TimeNetConnectorError(f"{location.context}: expected {len(header)} fields, got {len(cells)}")
        row = self.model.from_strings(dict(zip(header, cells, strict=True)), context=location.context)
        return LocatedRow(row, location)

    def _iter_rows(self, handle: TextIO) -> Generator[LocatedRow[RowT, CsvLocation], None, None]:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, [])
            self._check_header(header)
            while True:
                # csv.reader counts physical lines consumed, including quoted multiline cells.
                location = CsvLocation(self.path, reader.line_num + 1)
                cells = next(reader, None)
                if cells is None:
                    return
                if self.skip_repeated_header and cells == header:
                    continue
                yield self._parse_row(header, cells, location)
        except csv.Error as exc:
            raise TimeNetConnectorError(f"{self.path}:{reader.line_num}: {exc}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
