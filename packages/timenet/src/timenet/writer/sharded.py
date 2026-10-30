"""The one sharding implementation: stream Arrow row groups into byte-budgeted parquet parts.

:class:`RotatingPartWriter` is the shared core. The control-plane tables use it through
:func:`write_sharded_table` here. The Parquet values plane uses it directly (see
``values_backends.parquet.writer``), so both planes shard through one code path.
"""

from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.writer import encodings


class PartPool:
    """Writes closed parts on an executor, with at most ``workers`` parts in flight.

    Each part's completion callback runs on the calling thread, in submission order. So progress
    reports and checks see the parts in the same order as a serial write. A part's bytes depend only
    on its row groups, so the output is the same for any worker count.
    """

    def __init__(self, executor: Executor, workers: int) -> None:
        """Bind the pool to its executor and in-flight bound.

        Args:
            executor: Runs the part writes. Its workers must not share a part.
            workers: The most parts in flight at once. Submitting one more first waits for the oldest.
        """
        self._executor = executor
        self._workers = workers
        self._pending: deque[tuple[Future[None], Callable[[], None]]] = deque()

    def submit(self, write: Callable[[], None], done: Callable[[], None]) -> None:
        """Run ``write`` on the executor, and ``done`` on this thread once it and every earlier part finish.

        Args:
            write: Writes one whole part.
            done: Runs after the part is on disk.
        """
        while len(self._pending) >= self._workers:
            self._finish_oldest()
        self._pending.append((self._executor.submit(write), done))

    def drain(self) -> None:
        """Wait for every part in flight and run its callback, in submission order."""
        while self._pending:
            self._finish_oldest()

    def _finish_oldest(self) -> None:
        future, done = self._pending.popleft()
        future.result()
        done()


def _write_part(path: Path, schema: pa.Schema, tables: list[pa.Table], parquet_kwargs: dict[str, Any]) -> None:
    """Write ``tables`` to ``path`` as one row group each, as the serial path does."""
    with pq.ParquetWriter(path, schema, **parquet_kwargs) as writer:
        for table in tables:
            writer.write_table(table)


class RotatingPartWriter:
    """Writes caller-provided row groups into byte-budgeted parquet parts, rotating parts by budget.

    The caller decides each row group's content (one Arrow table per :meth:`write`) and its size. This
    opens a :class:`pyarrow.parquet.ParquetWriter` per part and writes each table as one row group.
    It rotates to a new numbered part once the current part's accumulated size exceeds the target,
    and it reports where each row group landed. Both the control plane and the values plane sit on
    it, so there is one sharding implementation.

    With a :class:`PartPool`, it holds a part's row groups in memory and hands the whole part to the
    pool when it closes, so parts compress in parallel. Without one, it streams each row group to
    disk as it arrives. Both write the same bytes.
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
        pool: PartPool | None = None,
    ) -> None:
        """Bind the writer to its staging directory, schema, path template, and byte budget.

        Args:
            staging_dir: The version's staging directory. The writer writes parts under it.
            schema: The Arrow schema the writer uses for every part.
            part_path: Maps a part index to its relative path.
            part_target_bytes: Rotate to a new part once a part's accumulated size exceeds this.
            parquet_kwargs: Keyword arguments for :class:`pyarrow.parquet.ParquetWriter`.
            on_part_closed: Optional callback invoked ``(rel_path, parts_written)`` as each part closes.
                With a pool, it runs once the part is on disk.
            pool: Writes each closed part in the background. ``None`` writes on this thread.
        """
        self._staging_dir = staging_dir
        self._schema = schema
        self._part_path = part_path
        self._part_target_bytes = part_target_bytes
        self._parquet_kwargs = parquet_kwargs
        self._on_part_closed = on_part_closed
        self._pool = pool
        self._writer: pq.ParquetWriter | None = None
        self._tables: list[pa.Table] | None = None  # the open part's row groups, with a pool
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
        if self._writer is None and self._tables is None:
            self._open()
        if self._tables is not None:
            self._tables.append(table)
        else:
            cast("pq.ParquetWriter", self._writer).write_table(table)
        landed = (self.parts[-1], self._row_group)
        self._row_group += 1
        self._part_bytes += size_bytes
        if self._part_bytes >= self._part_target_bytes:
            self._close()
        return landed

    def finish(self, *, write_empty_part: bool = False) -> list[str]:
        """Close the current part and return every part written, in order.

        With a pool, the last parts may still be in flight; the pool's owner drains it.

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

    def _open(self) -> None:
        rel = self._part_path(len(self.parts))
        path = self._staging_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._pool is not None:
            self._tables = []
        else:
            self._writer = pq.ParquetWriter(path, self._schema, **self._parquet_kwargs)
        self.parts.append(rel)
        self._row_group = 0
        self._part_bytes = 0

    def _close(self) -> None:
        if self._writer is None and self._tables is None:
            return
        rel, parts_written = self.parts[-1], len(self.parts)

        def done() -> None:
            if self._on_part_closed is not None:
                self._on_part_closed(rel, parts_written)

        if self._tables is None:
            cast("pq.ParquetWriter", self._writer).close()
            self._writer = None
            done()
            return
        path, tables = self._staging_dir / rel, self._tables
        self._tables = None
        cast("PartPool", self._pool).submit(lambda: _write_part(path, self._schema, tables, self._parquet_kwargs), done)


# While the bytes-per-row rate is unknown, the writer measures rows one at a time. After this many
# rows, it extrapolates the rate from the sample, so the expensive per-row measurement stays
# bounded regardless of size.
_PROBE_ROWS = 1024


class ShardedTableWriter:
    """Accepts control-table rows one at a time and streams them into byte-budgeted parquet parts.

    This is the streaming core behind :func:`write_sharded_table`. It buffers rows into row groups,
    flushes a group once it reaches ``min(row_group_target_bytes, control_target_bytes)``, and rotates
    to a new part once a part reaches ``control_target_bytes``. Until the bytes-per-row rate is known,
    it measures each row, so it sizes the first group exactly even for a small table or a tiny target.
    After ``_PROBE_ROWS`` rows it extrapolates the rate from the sample and sizes later groups by row
    count, re-measuring the rate at each flush. Rows never all live in memory at once, so a caller can
    feed millions of rows from a stream. :meth:`finish` closes the last part.
    """

    def __init__(  # noqa: PLR0913
        self,
        schema: pa.Schema,
        part_path: Callable[[int], str],
        *,
        staging_dir: Path,
        control_target_bytes: int,
        row_group_target_bytes: int,
        encoding: encodings.ParquetEncoding,
    ) -> None:
        """Bind the writer to its schema, path template, byte budgets, and encoding.

        Args:
            schema: The Arrow schema for the table.
            part_path: Maps a part index to its relative path.
            staging_dir: The version's staging directory.
            control_target_bytes: Rotate to a new part once a part's estimated size exceeds this.
            row_group_target_bytes: Target size of one row group inside a part.
            encoding: Dictionary/column encodings and compression applied to every part.
        """
        self._schema = schema
        self._core = RotatingPartWriter(
            staging_dir,
            schema,
            part_path,
            part_target_bytes=control_target_bytes,
            parquet_kwargs=encodings.parquet_kwargs(
                dictionary_columns=encoding.dictionary_columns,
                column_encoding=encoding.column_encoding,
                compression=encoding.compression,
                compression_level=encoding.compression_level,
                data_page_size=encoding.data_page_size,
            ),
        )
        self._row_group_bytes = min(row_group_target_bytes, control_target_bytes)
        self._buffer: list[dict] = []
        self._buffer_bytes = 0  # measured per row while the bytes-per-row rate is still unknown
        self._bytes_per_row = 0  # set once a group is measured, then re-set from every flush

    def _flush(self) -> None:
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._core.write(table, table.nbytes)
        self._bytes_per_row = max(1, table.nbytes // len(self._buffer))
        self._buffer.clear()

    def add(self, row: dict) -> None:
        """Append one row, flushing a row group when it fills.

        Args:
            row: The already-encoded payload row, matching the schema.
        """
        self._buffer.append(row)
        if self._bytes_per_row:
            # Rate known: size groups by row count, re-measured at each flush.
            if len(self._buffer) >= max(1, self._row_group_bytes // self._bytes_per_row):
                self._flush()
        else:
            # Learning: measure each row until a group fills or the probe is large enough to extrapolate.
            self._buffer_bytes += pa.Table.from_pylist([row], schema=self._schema).nbytes
            if self._buffer_bytes >= self._row_group_bytes:
                self._flush()
            elif len(self._buffer) >= _PROBE_ROWS:
                self._bytes_per_row = max(1, self._buffer_bytes // len(self._buffer))

    def finish(self, *, write_empty_part: bool = False) -> list[str]:
        """Flush any buffered rows, close the last part, and return every part written.

        Args:
            write_empty_part: Write one empty part when no row was ever added, so the schema stays
                on disk. The control tables need this.

        Returns:
            The relative paths of the parts written, in order.
        """
        if self._buffer:
            self._flush()
        return self._core.finish(write_empty_part=write_empty_part)


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
    """Write ``rows`` as one or more parquet parts through a :class:`ShardedTableWriter`.

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
    writer = ShardedTableWriter(
        schema,
        part_path,
        staging_dir=staging_dir,
        control_target_bytes=control_target_bytes,
        row_group_target_bytes=row_group_target_bytes,
        encoding=encoding,
    )
    for row in rows:
        writer.add(row)
    return writer.finish(write_empty_part=True)
