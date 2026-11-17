"""Read CSV source rows."""

from collections.abc import Generator, Mapping, Sequence
import csv
from pathlib import Path
from typing import TypeGuard

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataReader, RowT


class CsvDataReader(DataReader[RowT]):
    """Read a UTF-8 CSV file with a header."""

    def __init__(
        self,
        path: Path,
        row_type: type[RowT],
        *,
        expected_columns: tuple[str, ...] | None = None,
        max_field_chars: int | None = None,
    ) -> None:
        """Set up a reader for one CSV file.

        Args:
            path: The CSV file.
            row_type: The schema of each row.
            expected_columns: The exact header, in order, or ``None`` to accept any header.
            max_field_chars: The largest cell that the file may hold, in characters, or ``None`` for the
                limit of the ``csv`` module, which is 131,072 by default. Use it for a source whose cells
                are known to be larger, such as long series in one cell.
        """
        super().__init__(path, row_type)
        self.expected_columns = expected_columns
        self.max_field_chars = max_field_chars

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated rows parsed from CSV cells.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or a row is invalid.
        """
        self._allow_field_chars()
        try:
            with self.path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                header = reader.fieldnames
                if not self._has_unique_header(header):
                    raise TimeNetConnectorError(f"{self.path}: missing or duplicate CSV header")
                if not self._has_expected_columns(header):
                    raise TimeNetConnectorError(
                        f"{self.path.name}: expected columns {self.expected_columns}, got {header}"
                    )
                for cells in reader:
                    if self._has_extra_cells(cells):
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too many cells")
                    if self._has_missing_cells(cells):
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too few cells")
                    yield self._validate(cells, f"line {reader.line_num}", strings=True)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc

    def _allow_field_chars(self) -> None:
        """Raise the cell limit of the ``csv`` module to ``max_field_chars``, if it is lower.

        The ``csv`` module has one limit for the whole process, and it checks the limit while it parses. The
        reader never lowers it, because another reader may be in the middle of a file with larger cells.
        """
        if self.max_field_chars is not None and self.max_field_chars > csv.field_size_limit():
            csv.field_size_limit(self.max_field_chars)

    def _has_expected_columns(self, header: Sequence[str]) -> bool:
        """Return whether the header has exactly the expected columns in order, or no columns are expected."""
        return self.expected_columns is None or tuple(header) == self.expected_columns

    @staticmethod
    def _has_unique_header(header: Sequence[str] | None) -> TypeGuard[Sequence[str]]:
        """Return whether the CSV has a header row with no repeated column name."""
        return header is not None and len(header) > 0 and len(set(header)) == len(header)

    @staticmethod
    def _has_extra_cells(cells: Mapping[str | None, object]) -> bool:
        """Return whether a row has more cells than the header.

        ``csv.DictReader`` stores extra cells under the ``None`` key.
        """
        return None in cells

    @staticmethod
    def _has_missing_cells(cells: Mapping[str | None, object]) -> bool:
        """Return whether a row has fewer cells than the header.

        ``csv.DictReader`` sets each missing cell to ``None``.
        """
        return any(value is None for value in cells.values())
