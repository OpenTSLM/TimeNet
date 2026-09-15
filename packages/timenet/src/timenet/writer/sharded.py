"""Stream Arrow row groups into byte-budgeted parquet parts.

:class:`RotatingPartWriter` is what the Parquet values plane shards through (see
``values_backends.parquet.writer``).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class RotatingPartWriter:
    """Writes caller-provided row groups into byte-budgeted parquet parts. It rotates them by budget.

    The caller decides each row group's content (one Arrow table per :meth:`write`) and its size. This
    opens a :class:`pyarrow.parquet.ParquetWriter` per part and writes each table as one row group.
    It rotates to a new numbered part once the current part's accumulated size exceeds the target,
    and it reports where each row group landed.
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
        """Append ``table`` as one row group, and open a new part first when the current one is full.

        Args:
            table: The row group's content, which matches the schema.
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
