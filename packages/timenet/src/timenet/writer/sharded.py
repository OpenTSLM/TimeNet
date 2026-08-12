"""Stream control-table rows into byte-budgeted Parquet parts, recording per-part skip metadata."""

from collections.abc import Callable, Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.manifest import PartStat
from timenet.writer import encodings


def write_sharded_table(  # noqa: PLR0913
    rows: Iterable[dict],
    schema: pa.Schema,
    part_path: Callable[[int], str],
    key_of: Callable[[dict], tuple] | None,
    *,
    staging_dir: Path,
    control_target_bytes: int,
    row_group_target_bytes: int,
    dictionary_columns: list[str],
    column_encoding: dict[str, str] | None,
    compression: str,
    compression_level: int,
) -> list[PartStat]:
    """Write ``rows`` as one or more Parquet parts, rotating on the control-table byte budget.

    Rows are buffered until the buffer's estimated size reaches ``control_target_bytes``, then flushed
    as one part with internal row groups sized by ``row_group_target_bytes``. Boundaries are sized from
    the in-memory Arrow ``nbytes`` and are not a stable contract. An empty input still writes exactly
    one empty part so the schema stays on disk.

    Args:
        rows: The already-encoded payload rows, consumed lazily.
        schema: The Arrow schema for the table.
        part_path: Maps a part index to its relative path.
        key_of: Maps a row to its boundary key tuple, or ``None`` to record no keys.
        staging_dir: The version's staging directory; parts are written under it.
        control_target_bytes: Rotate to a new part once a part's estimated size exceeds this.
        row_group_target_bytes: Target size of one row group inside a part.
        dictionary_columns: Columns to dictionary-encode.
        column_encoding: Optional per-column encoding overrides.
        compression: Parquet codec name.
        compression_level: Codec level (applied for zstd).

    Returns:
        The parts written, in order.
    """
    kwargs = encodings.parquet_kwargs(
        dictionary_columns=dictionary_columns,
        column_encoding=column_encoding,
        compression=compression,
        compression_level=compression_level,
    )
    parts: list[PartStat] = []
    buffer: list[dict] = []
    bytes_per_row: int | None = None
    part_idx = 0

    def flush() -> None:
        nonlocal buffer, bytes_per_row, part_idx
        table = pa.Table.from_pylist(buffer, schema=schema)
        bytes_per_row = max(1, table.nbytes // max(1, len(table)))
        rows_per_group = max(1, row_group_target_bytes // bytes_per_row)
        rel = part_path(part_idx)
        path = staging_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=rows_per_group, **kwargs)
        parts.append(
            PartStat(
                path=rel,
                n_rows=len(table),
                first_key=key_of(buffer[0]) if key_of else None,
                last_key=key_of(buffer[-1]) if key_of else None,
            )
        )
        part_idx += 1
        buffer = []

    for row in rows:
        buffer.append(row)
        if bytes_per_row is None:
            bytes_per_row = max(1, pa.Table.from_pylist([row], schema=schema).nbytes)
        if len(buffer) >= max(1, control_target_bytes // bytes_per_row):
            flush()
    if buffer:
        flush()
    if not parts:
        rel = part_path(0)
        path = staging_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([], schema=schema), path, **kwargs)
        parts.append(PartStat(path=rel, n_rows=0, first_key=None, last_key=None))
    return parts
