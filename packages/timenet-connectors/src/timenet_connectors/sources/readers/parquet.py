"""Read Parquet source rows."""

from collections.abc import Generator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataReader, RowT


class ParquetDataReader(DataReader[RowT]):
    """Read a Parquet file in batches, yielding individual rows."""

    def __init__(
        self, path: Path, row_type: type[RowT], *, strict: bool | None = None, columns: Sequence[str] | None = None
    ) -> None:
        super().__init__(path, row_type, strict=strict)
        self.columns = None if columns is None else list(columns)

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated rows in file order.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or a row is invalid.
        """
        try:
            with pq.ParquetFile(self.path) as parquet:
                names = parquet.schema_arrow.names
                if not self._has_unique_columns(names):
                    raise TimeNetConnectorError(f"{self.path}: duplicate Parquet column names")
                missing = self._missing_columns(names)
                if missing:
                    raise TimeNetConnectorError(f"{self.path}: missing columns {missing}")
                number = 0
                for batch in parquet.iter_batches(columns=self.columns):
                    for value in batch.to_pylist():
                        number += 1
                        yield self._validate(value, f"row {number}")
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc

    def _missing_columns(self, names: Sequence[str]) -> list[str]:
        """Return the selected columns that the file does not have, in selection order."""
        return [name for name in self.columns or () if name not in names]

    @staticmethod
    def _has_unique_columns(names: Sequence[str]) -> bool:
        """Return whether no column name repeats in the file schema."""
        return len(set(names)) == len(names)
