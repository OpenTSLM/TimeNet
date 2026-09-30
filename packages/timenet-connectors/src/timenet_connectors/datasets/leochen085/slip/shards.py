"""Walk the nested Parquet shards of the SLIP release, and load their series lazily."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import OrdinalAxis, Signal
from timenet.types import TimeSeriesSpec


_BATCH_ROWS = 2048  # rows decoded per batch while walking a shard


@lru_cache(maxsize=1)
def nested_row_group(shard: Path, group: int, column: str) -> pa.ChunkedArray:
    """Read one nested column of one row group.

    The writer reads signals in record order, so the one cached row group serves every signal of
    its rows before the next group is read.

    Returns:
        The column's values for that row group.
    """
    return pq.ParquetFile(shard).read_row_groups([group], columns=[column]).column(column)


@dataclass(frozen=True)
class SlipRow:
    """One row of a shard: its scalar columns, and where its nested series sit."""

    scalars: dict[str, Any]  # every column but the nested one
    lengths: list[int]  # one per series of the nested column
    shard: Path
    group: int  # the row group holding the row
    offset: int  # the row's index within that group
    column: str  # the nested column

    def signals(self, *, spec: TimeSeriesSpec, record_id: str) -> tuple[Signal, ...]:
        """Build one lazy ordinal Signal per series of the row.

        The writer reads values sorted by spec, ``source_id``, name and id, so the record id is the
        source id of every signal of the row. Record ids that sort in row order then keep the writer
        walking the shards forward, one cached row group at a time.

        Returns:
            The signals, named ``s00``, ``s01``, ... in release order. A record lists its signals
            sorted by name, and the release has rows of up to 21 series, so the number is padded.
        """
        return tuple(
            Signal.from_loader(
                id=f"{record_id}-s{index:02d}",
                name=f"s{index:02d}",
                spec=spec,
                time_axis=OrdinalAxis(),
                n_values=length,
                source_id=record_id,
                loader=lambda index=index: (
                    nested_row_group(self.shard, self.group, self.column)[self.offset][index].values
                ),
            )
            for index, length in enumerate(self.lengths)
        )


def iter_rows(shard: Path, column: str) -> Iterator[SlipRow]:
    """Walk one shard row by row.

    The nested column is read for its list lengths only. Its values are loaded later, by the
    Signals a row builds.

    Args:
        shard: The Parquet file.
        column: The nested list-of-lists column holding the row's series.

    Yields:
        One row at a time, in file order.
    """
    reader = pq.ParquetFile(shard)
    scalars = [name for name in reader.schema_arrow.names if name != column]
    for group in range(reader.metadata.num_row_groups):
        offset = 0
        for batch in reader.iter_batches(batch_size=_BATCH_ROWS, row_groups=[group]):
            nested = batch.column(column)
            counts = nested.value_lengths().to_pylist()
            lengths = nested.flatten().value_lengths().to_pylist()
            cursor = 0
            for row, count in zip(batch.select(scalars).to_pylist(), counts, strict=True):
                yield SlipRow(row, lengths[cursor : cursor + count], shard, group, offset, column)
                cursor += count
                offset += 1
