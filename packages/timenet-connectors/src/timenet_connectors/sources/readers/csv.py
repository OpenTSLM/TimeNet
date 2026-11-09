"""Read CSV source rows."""

from collections.abc import Generator
import csv
from pathlib import Path

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataReader, RowT


class CsvReader(DataReader[RowT]):
    """Read a UTF-8 CSV file with a header."""

    def __init__(
        self, path: Path, row_type: type[RowT], *, strict: bool | None = None, columns: tuple[str, ...] | None = None
    ) -> None:
        super().__init__(path, row_type, strict=strict)
        self.columns = columns

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated rows parsed from CSV cells.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or a row is invalid.
        """
        try:
            with self.path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise TimeNetConnectorError(f"{self.path}: missing or duplicate CSV header")
                if self.columns is not None and tuple(reader.fieldnames) != self.columns:
                    raise TimeNetConnectorError(
                        f"{self.path.name}: expected columns {self.columns}, got {reader.fieldnames}"
                    )
                for cells in reader:
                    if None in cells:
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too many cells")
                    if any(value is None for value in cells.values()):
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too few cells")
                    yield self._validate(cells, f"line {reader.line_num}", strings=True)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
