"""Read strict CSV metadata tables with physical source locations."""

from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path

from timenet.errors import TimeNetConnectorError


@dataclass(frozen=True)
class CsvRow:
    """A row's physical starting line and unmodified values keyed by column name."""

    line: int
    values: dict[str, str]


def read_csv_table(path: Path, *, columns: Sequence[str]) -> list[CsvRow]:
    """Read a small CSV table, retaining each row's starting file line.

    Columns may appear in any order. Blank rows, missing values and extra fields are errors.
    Cell text is preserved; quoted fields may span lines. A header-only table is allowed.
    This helper materializes small metadata tables, not large signal recordings.

    Returns:
        Each row's starting line and its values keyed by column.

    Raises:
        TimeNetConnectorError: If the file cannot be read or its header or rows are malformed.
    """
    columns = list(columns)
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            if len(header) != len(columns) or len(set(header)) != len(header) or set(header) != set(columns):
                raise TimeNetConnectorError(f"{path}: expected the columns {columns}, got {header}")
            while True:
                line = reader.line_num + 1
                cells = next(reader, None)
                if cells is None:
                    break
                if not cells:
                    raise TimeNetConnectorError(f"{path}:{line}: blank row")
                if len(cells) > len(header):
                    raise TimeNetConnectorError(f"{path}:{line}: more fields than the header names")
                row = dict(zip(header, cells, strict=False))
                for column in columns:
                    if not row.get(column):
                        raise TimeNetConnectorError(f"{path}:{line}: no value for {column!r}")
                rows.append(CsvRow(line=line, values=row))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    return rows


def check_unique(path: Path, values: Iterable[tuple[int, str]], *, name: str) -> None:
    """Reject a duplicate value, naming both its line and its first occurrence.

    Raises:
        TimeNetConnectorError: If a value appears twice.
    """
    first_lines: dict[str, int] = {}
    for line, value in values:
        if value in first_lines:
            raise TimeNetConnectorError(
                f"{path}:{line}: a {name} is listed twice: {value!r}; first listed on line {first_lines[value]}"
            )
        first_lines[value] = line
