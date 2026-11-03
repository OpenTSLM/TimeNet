"""Read parquet source files row by row."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


_BATCH_ROWS = 65536  # parquet rows decoded per batch


def iter_parquet_rows(path: Path, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yield one plain dict per row of a parquet file, decoding a batch at a time.

    Args:
        path: The parquet file.
        columns: The columns to read, which leaves out the others' decoding cost. All columns by default.

    Yields:
        Each row, keyed by column name, in file order.
    """
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=_BATCH_ROWS, columns=None if columns is None else list(columns)
    ):
        yield from batch.to_pylist()
