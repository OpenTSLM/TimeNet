"""Read Parquet source rows."""

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataReader, RowT


class ParquetDataReader(DataReader[RowT]):
    """Read a Parquet file in batches, yielding individual rows.

    A column in ``arrow_columns`` stays Arrow. The row holds its value as a ``pyarrow.Scalar``, and the
    reader does not convert it to Python objects. Use this for large numeric list columns, where the
    conversion is the main cost of reading. The row schema must allow Arrow types with
    ``ConfigDict(arbitrary_types_allowed=True)``.
    """

    def __init__(
        self,
        path: Path,
        row_type: type[RowT],
        *,
        selected_columns: Sequence[str] | None = None,
        arrow_columns: Sequence[str] = (),
    ) -> None:
        super().__init__(path, row_type)
        self.selected_columns = None if selected_columns is None else list(selected_columns)
        self.arrow_columns = list(arrow_columns)
        outside = [name for name in self.arrow_columns if selected_columns is not None and name not in selected_columns]
        if outside:
            raise TimeFValidationError(f"arrow_columns must be selected columns, but {outside} are not")

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
                row_number = 0
                for batch in parquet.iter_batches(columns=self.selected_columns):
                    try:
                        values = self._to_pylist(batch)
                    except (ValueError, OverflowError) as exc:
                        raise TimeNetConnectorError(
                            f"{self.path}: rows {row_number + 1}-{row_number + batch.num_rows}: {exc}"
                        ) from exc
                    arrow = [(name, batch.column(name)) for name in self.arrow_columns]
                    for index, value in enumerate(values):
                        for name, column in arrow:
                            value[name] = column[index]
                        row_number += 1
                        yield self._validate(value, f"row {row_number}")
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc

    def _to_pylist(self, batch: pa.RecordBatch) -> list[dict[str, Any]]:
        """Convert the columns of a batch to Python objects, except the Arrow columns.

        Returns:
            One dict per row, without the Arrow columns.
        """
        if not self.arrow_columns:
            return batch.to_pylist()
        kept = [name for name in batch.schema.names if name not in self.arrow_columns]
        if not kept:
            return [{} for _ in range(batch.num_rows)]
        return batch.select(kept).to_pylist()

    def _missing_columns(self, names: Sequence[str]) -> list[str]:
        """Return the selected and Arrow columns that the file does not have, in order."""
        return [name for name in (*(self.selected_columns or ()), *self.arrow_columns) if name not in names]

    @staticmethod
    def _has_unique_columns(names: Sequence[str]) -> bool:
        """Return whether no column name repeats in the file schema."""
        return len(set(names)) == len(names)
