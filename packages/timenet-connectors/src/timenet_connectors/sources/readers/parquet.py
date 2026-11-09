"""Read Parquet rows through the shared typed-source contract."""

from collections import Counter
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import LocatedRow, RowSource, RowT


_BATCH_ROWS = 65536  # Bound decoded rows retained while streaming a large file.


@dataclass(frozen=True)
class ParquetLocation:
    """A file and a zero-based row index across all its row groups."""

    path: Path
    index: int

    @property
    def context(self) -> str:
        """The file and row position for diagnostics."""
        return f"{self.path} row {self.index}"


@dataclass(frozen=True)
class ParquetRowSource(RowSource[RowT, ParquetLocation]):
    """Stream strictly validated decoded rows with optional schema-owned column projection.

    By default every file column must belong to the model. Set ``project_columns`` explicitly
    to read only the model's declared columns, leaving unrelated payloads undecoded. Missing
    required columns still fail, including for an empty file; optional ones use model defaults.
    Arrow decodes values to Python without a dataframe or a common numeric dtype conversion.
    """

    project_columns: bool = False

    def _open_rows(self, resources: ExitStack) -> Generator[LocatedRow[RowT, ParquetLocation], None, None]:
        try:
            parquet = resources.enter_context(pq.ParquetFile(self.path))
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
        return self._iter_rows(parquet)

    def _columns(self, parquet: pq.ParquetFile) -> list[str]:
        available = parquet.schema_arrow.names
        repeated = [name for name, count in Counter(available).items() if count > 1]
        if repeated:
            raise TimeNetConnectorError(f"{self.path}: repeated columns {repeated}")
        columns = available
        if self.project_columns:
            columns = [name for name in self.model.column_names() if name in available]
        self.model.check_columns(columns, context=str(self.path))
        return columns

    def _iter_rows(self, parquet: pq.ParquetFile) -> Generator[LocatedRow[RowT, ParquetLocation], None, None]:
        index = 0
        try:
            columns = self._columns(parquet)
            for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=columns):
                for values in batch.to_pylist():
                    location = ParquetLocation(self.path, index)
                    row = self.model.from_decoded(values, context=location.context)
                    yield LocatedRow(row, location)
                    index += 1
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.path} row {index}: {exc}") from exc
