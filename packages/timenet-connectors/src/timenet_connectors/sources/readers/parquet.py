"""Read Parquet source rows."""

from collections.abc import Generator

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import RowReader, RowT


class ParquetReader(RowReader[RowT]):
    """Read a Parquet file in batches, yielding individual rows."""

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated rows in file order.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or a row is invalid.
        """
        try:
            with pq.ParquetFile(self.path) as parquet:
                number = 0
                for batch in parquet.iter_batches():
                    for value in batch.to_pylist():
                        number += 1
                        yield self._validate(value, f"row {number}")
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
