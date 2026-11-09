"""Read CSV source rows."""

from collections.abc import Generator
import csv

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import RowReader, RowT


class CsvReader(RowReader[RowT]):
    """Read a UTF-8 CSV file with a header."""

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
                for cells in reader:
                    if None in cells:
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too many cells")
                    if any(value is None for value in cells.values()):
                        raise TimeNetConnectorError(f"{self.path}: line {reader.line_num}: too few cells")
                    yield self._validate(cells, f"line {reader.line_num}", strings=True)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
