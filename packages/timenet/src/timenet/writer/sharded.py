"""The one sharding implementation: stream Arrow row groups into byte-budgeted parquet parts.

:class:`RotatingPartWriter` is the shared core. The control-plane tables use it through
:func:`write_sharded_table` here. The Parquet values plane uses it directly (see
``values_backends.parquet.writer``), so both planes shard through one code path.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer import encodings


class RotatingPartWriter:
    """Writes caller-provided row groups into byte-budgeted parquet parts, rotating parts by budget.

    The caller decides each row group's content (one Arrow table per :meth:`write`) and its size. This
    opens a :class:`pyarrow.parquet.ParquetWriter` per part and writes each table as one row group.
    It rotates to a new numbered part once the current part's accumulated size exceeds the target,
    and it reports where each row group landed. Both the control plane and the values plane sit on
    it, so there is one sharding implementation.
    """

    def __init__(  # noqa: PLR0913
        self,
        staging_dir: Path,
        schema: pa.Schema,
        part_path: Callable[[int], str],
        *,
        part_target_bytes: int,
        parquet_kwargs: dict[str, Any],
        on_part_closed: Callable[[str, int], None] | None = None,
    ) -> None:
        """Bind the writer to its staging directory, schema, path template, and byte budget.

        Args:
            staging_dir: The version's staging directory. The writer writes parts under it.
            schema: The Arrow schema the writer uses for every part.
            part_path: Maps a part index to its relative path.
            part_target_bytes: Rotate to a new part once a part's accumulated size exceeds this.
            parquet_kwargs: Keyword arguments for :class:`pyarrow.parquet.ParquetWriter`.
            on_part_closed: Optional callback invoked ``(rel_path, parts_written)`` as each part closes.
        """
        self._staging_dir = staging_dir
        self._schema = schema
        self._part_path = part_path
        self._part_target_bytes = part_target_bytes
        self._parquet_kwargs = parquet_kwargs
        self._on_part_closed = on_part_closed
        self._writer: pq.ParquetWriter | None = None
        self._row_group = 0
        self._part_bytes = 0
        self.parts: list[str] = []

    def write(self, table: pa.Table, size_bytes: int) -> tuple[str, int]:
        """Append ``table`` as one row group, opening a new part first if the current one is full.

        Args:
            table: The row group's content, matching the schema.
            size_bytes: The row group's uncompressed size, charged against the part budget.

        Returns:
            The relative path of the part the row group landed in and its index within that part.
        """
        writer = self._writer if self._writer is not None else self._open()
        writer.write_table(table)
        landed = (self.parts[-1], self._row_group)
        self._row_group += 1
        self._part_bytes += size_bytes
        if self._part_bytes >= self._part_target_bytes:
            self._close()
        return landed

    def finish(self, *, write_empty_part: bool = False) -> list[str]:
        """Close the current part and return every part written, in order.

        Args:
            write_empty_part: If the caller never wrote a row group, still write one empty part so
                the schema stays on disk. The control tables need this. The values plane does not.

        Returns:
            The relative paths of the parts written.
        """
        self._close()
        if not self.parts and write_empty_part:
            self._open()
            self._close()
        return self.parts

    def _open(self) -> pq.ParquetWriter:
        rel = self._part_path(len(self.parts))
        path = self._staging_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, self._schema, **self._parquet_kwargs)
        self._writer = writer
        self.parts.append(rel)
        self._row_group = 0
        self._part_bytes = 0
        return writer

    def _close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self._on_part_closed is not None:
            self._on_part_closed(self.parts[-1], len(self.parts))


# While the bytes-per-row rate is unknown, the writer measures rows one at a time. After this many
# rows, it extrapolates the rate from the sample, so the expensive per-row measurement stays
# bounded regardless of size.
_PROBE_ROWS = 1024


def write_sharded_table(  # noqa: PLR0913
    rows: Iterable[dict],
    schema: pa.Schema,
    part_path: Callable[[int], str],
    *,
    staging_dir: Path,
    control_target_bytes: int,
    row_group_target_bytes: int,
    encoding: encodings.ParquetEncoding,
) -> list[str]:
    """Write ``rows`` as one or more parquet parts through :class:`RotatingPartWriter`.

    The writer buffers rows into row groups. The part writer rotates to a new part once a part
    reaches ``control_target_bytes``. The writer flushes a group once it reaches
    ``min(row_group_target_bytes, control_target_bytes)``. Until the bytes-per-row rate is known, the
    writer measures each row. This way, it sizes the first group exactly, so even a small table or a
    tiny target still shards. After ``_PROBE_ROWS`` rows, the writer extrapolates the rate from the
    sample. It then sizes later groups by row count, and each flush re-measures the rate for free.
    The table is ordered by a random id, so adjacent groups sample the same size distribution and
    stay evenly sized. This keeps each group's statistics tight enough to prune on. An empty input
    still writes exactly one empty part so the schema stays on disk.

    Args:
        rows: The already-encoded payload rows, consumed lazily.
        schema: The Arrow schema for the table.
        part_path: Maps a part index to its relative path.
        staging_dir: The version's staging directory.
        control_target_bytes: Rotate to a new part once a part's estimated size exceeds this.
        row_group_target_bytes: Target size of one row group inside a part.
        encoding: Dictionary/column encodings and compression applied to every part.

    Returns:
        The relative paths of the parts written, in order.
    """
    core = RotatingPartWriter(
        staging_dir,
        schema,
        part_path,
        part_target_bytes=control_target_bytes,
        parquet_kwargs=encodings.parquet_kwargs(
            dictionary_columns=encoding.dictionary_columns,
            column_encoding=encoding.column_encoding,
            compression=encoding.compression,
            compression_level=encoding.compression_level,
        ),
    )
    row_group_bytes = min(row_group_target_bytes, control_target_bytes)
    buffer: list[dict] = []
    buffer_bytes = 0  # measured per row while the bytes-per-row rate is still unknown
    bytes_per_row = 0  # set once a group is measured, then re-set from every flush

    def flush() -> int:
        table = pa.Table.from_pylist(buffer, schema=schema)
        core.write(table, table.nbytes)
        rate = max(1, table.nbytes // len(buffer))
        buffer.clear()
        return rate

    for row in rows:
        buffer.append(row)
        if bytes_per_row:
            # Rate known: size groups by row count, re-measured at each flush.
            if len(buffer) >= max(1, row_group_bytes // bytes_per_row):
                bytes_per_row = flush()
        else:
            # Learning: measure each row until a group fills or the probe is large enough to extrapolate.
            buffer_bytes += pa.Table.from_pylist([row], schema=schema).nbytes
            if buffer_bytes >= row_group_bytes:
                bytes_per_row = flush()
            elif len(buffer) >= _PROBE_ROWS:
                bytes_per_row = max(1, buffer_bytes // len(buffer))
    if buffer:
        flush()
    return core.finish(write_empty_part=True)
