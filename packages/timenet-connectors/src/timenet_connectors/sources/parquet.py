"""Read parquet source files row by row."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeNetConnectorError


_BATCH_ROWS = 65536  # parquet rows decoded per batch


def iter_parquet_rows(path: Path, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yield one plain dict per row of a parquet file, decoding a batch at a time.

    Args:
        path: The parquet file.
        columns: The columns to read, which leaves out the others' decoding cost. All columns by default.

    Yields:
        Each row, keyed by column name, in file order.

    Raises:
        TimeNetConnectorError: If the file is not readable parquet, a page in it is corrupt, or
            ``columns`` names a column the file doesn't have.
    """
    # A missing file raises FileNotFoundError here, as for any other source file.
    try:
        parquet_file = pq.ParquetFile(path)
    except pa.ArrowException as error:
        raise TimeNetConnectorError(f"{path}: cannot read the parquet file: {error}") from error
    if columns is not None:
        missing = [name for name in columns if name not in parquet_file.schema_arrow.names]
        if missing:
            raise TimeNetConnectorError(f"{path}: no column {missing}")
    try:
        batches = parquet_file.iter_batches(batch_size=_BATCH_ROWS, columns=None if columns is None else list(columns))
        for batch in batches:
            yield from batch.to_pylist()
    # Arrow raises a plain OSError for a page that it cannot decode.
    except (pa.ArrowException, OSError) as error:
        raise TimeNetConnectorError(f"{path}: cannot read the parquet file: {error}") from error
