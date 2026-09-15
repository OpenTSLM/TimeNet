"""The Original side of the SLIP row: the release's 17 Parquet shards and its ``meta.csv``.

The canonical item is one row of ``data/``: its 1 to 6 float32 signals and the four captions that
describe them. There are 618,508 of them. A row already holds whole series, so neither side has to
cut an item out of a longer recording.

**The release's own loader is replaced, and the paper declares that.** ``SLIPhfDataset`` builds a
numpy array per row inside its constructor, so it holds all 618,508 windows before it hands over the
first one. That is 1,684,610,695 float32 values, and no resident-set cap in this campaign admits it.
What stays is the release's per-item read: pyarrow reads the row, every column the shard ships, and
the window becomes numpy from there. What goes is the wholesale materialization, and nothing else.

Neither variant reads a row group whole. Shard 08 alone is 1.29 GB and a row group runs to hundreds
of megabytes once its values are decoded, so both walk a shard forward in
:data:`BATCH_ROWS`-row batches over a memory map. That is the shape
:class:`~timenet_connectors.readers.ParquetRowReader` uses for the build, with one batch alive at a
time. The two variants differ in the projection:

``as_shipped``
    Every column of the row, which is what the release's own row access hands a caller.
``lazy_capable``
    The seven columns the item needs: ``category``, ``dataset``, ``caption0`` to ``caption3`` and
    ``time_series``. The connector reads those seven and no more, so a shard that ships nothing else
    makes the projection a no-op and the two variants move the same bytes. It is written as a
    projection anyway, because a column no item needs must not be charged to the read.

Both variants decode identically after the read, so the gap between them is bytes off disk rather
than two different pipelines.

The captions are read and they are not a checksummed plane. The TimeF side resolves the same six
annotations per record inside its own timed region, and its pandas item is the array-cell frame,
which carries no caption column. Reading them keeps the two items equal; checksumming them would
make the two sides' ``bytes_touched`` incomparable.

``meta.csv`` is the only place a cadence exists, because no row of ``data/`` carries a timestamp or
a rate. Its ``Freq`` column is free text with 15 spellings, ``3 h`` and ``3h`` among them, so this
lane parses it with the connector's own
:func:`~timenet_connectors.datasets.leochen085.slip_pretrain.meta.parse_time_axis`, which raises on a
spelling it does not know. A fallback to an ordinal axis would drop the time of a whole corpus
without saying so.

The release publishes 96 all-NaN signals among its DaLiA windows, 37,248 values. TimeF stores them
as they are, so this lane delivers them as NaN. It drops no signal and substitutes no value.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterator, Sequence
import csv
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, ORIGINAL, PANDAS, TORCH


if TYPE_CHECKING:  # pragma: no cover - import only for the annotations
    from timenet.dataset.axis import TimeAxis


DATASET = "leochen085/slip-pretrain"

DATA_DIRNAME = "data"
"""The pretraining split. The 11 downstream evaluation folders use another schema and are out of
scope on both sides of the row."""

SHARD_GLOB = "train-*.parquet"
"""The shards, whose sorted names are the shard order the record ids count in."""

META_CSV = "meta.csv"
"""The control table over the 37 source corpora. It carries the only cadence the release states."""

CATEGORY_COLUMN = "category"
DATASET_COLUMN = "dataset"
CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")
VALUES_COLUMN = "time_series"
"""One row's signals, as a list of per-signal float32 lists."""

ITEM_COLUMNS = (CATEGORY_COLUMN, DATASET_COLUMN, *CAPTION_COLUMNS, VALUES_COLUMN)
"""What ``lazy_capable`` projects to: the labels that pick the time axis, the caption set, and the
window."""

META_COLUMNS = ("Domain", "Dataset", "Freq")
"""The ``meta.csv`` columns this lane reads. ``Domain`` is what a ``data/`` row calls ``category``."""

BATCH_ROWS = 512
"""Rows decoded per batch, the figure the connector's reader was measured at.

It bounds the resident set of a read to one batch of one shard rather than one row group.
"""

OPEN_SHARDS = 2
"""How many shards the handle keeps open.

Canonical order walks one shard forward and a request that straddles a shard boundary needs two. A
larger cache would hold every shard's cursor after one pass, which a block-shuffled read must not
get for free.
"""


class Window(NamedTuple):
    """One row of ``data/``, read and decoded.

    Attributes:
        category: The ``category`` cell, which names the source corpus with ``dataset``.
        dataset: The ``dataset`` cell.
        captions: The four captions, in column order.
        signals: The row's signals, float32, in stored order.
    """

    category: str
    dataset: str
    captions: tuple[str, ...]
    signals: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Shard:
    """One shard, and where its rows sit in the canonical enumeration.

    Attributes:
        path: The shard file.
        index: Its position in shard order, which the record id carries.
        first_ordinal: Canonical ordinal of its first row.
        group_starts: First row of each row group, and the shard's row count last.
    """

    path: Path
    index: int
    first_ordinal: int
    group_starts: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        """Return how many rows the shard holds.

        Returns:
            The row count over every row group.
        """
        return self.group_starts[-1]

    def locate(self, row: int) -> tuple[int, int]:
        """Return which row group a row sits in, and where in it.

        Args:
            row: Row index inside the shard, counted over the whole file.

        Returns:
            The row-group index and the row index inside that group.
        """
        group = bisect.bisect_right(self.group_starts, row) - 1
        return group, row - self.group_starts[group]


def read_layout(root: Path) -> tuple[Shard, ...]:
    """Read the shard footers, which is all it takes to address any row.

    This reads no values and no row group. It is what makes the enumeration O(1) in memory: an item
    id is arithmetic over the shard index and the row, so the lane never holds 618,508 strings.

    Args:
        root: The release directory, holding ``data/`` and ``meta.csv``.

    Returns:
        One entry per shard, in shard order.

    Raises:
        LaneError: If the release holds no shard, or a footer cannot be read.
    """
    # A lane dependency, imported per lane rather than at module import.
    import pyarrow.parquet as pq  # noqa: PLC0415

    paths = sorted((root / DATA_DIRNAME).glob(SHARD_GLOB))
    if not paths:
        raise LaneError(f"{root / DATA_DIRNAME} holds no SLIP shard, so the release carries no item")
    shards: list[Shard] = []
    first_ordinal = 0
    for index, path in enumerate(paths):
        try:
            metadata = pq.read_metadata(path)
        except (OSError, ValueError) as error:
            raise LaneError(f"could not read the footer of the SLIP shard {path}: {error}") from error
        starts = [0]
        for group in range(metadata.num_row_groups):
            starts.append(starts[-1] + metadata.row_group(group).num_rows)
        shards.append(Shard(path=path, index=index, first_ordinal=first_ordinal, group_starts=tuple(starts)))
        first_ordinal += starts[-1]
    return tuple(shards)


def source_axes(root: Path) -> dict[tuple[str, str], TimeAxis]:
    """Read ``meta.csv`` into the time axis every window of each source corpus sits on.

    The spellings are parsed with the connector's own table, so an unknown one raises rather than
    costing a whole corpus its time.

    Args:
        root: The release directory.

    Returns:
        Per ``(category, dataset)`` pair, the axis its windows use.

    Raises:
        LaneError: If the table is missing, empty, or does not carry the columns this lane reads.
        TimeFValidationError: If a row states a frequency the release's own table does not use.
    """  # noqa: DOC502 (raised by parse_time_axis, not directly here)
    # A lane dependency, imported per lane rather than at module import.
    from timenet_connectors.datasets.leochen085.slip_pretrain.meta import parse_time_axis  # noqa: PLC0415

    path = root / META_CSV
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise LaneError(f"could not read the SLIP control table {path}: {error}") from error
    if not rows:
        raise LaneError(f"{path} holds no rows, so no window of the release has a time axis")
    missing = [column for column in META_COLUMNS if column not in rows[0]]
    if missing:
        raise LaneError(f"{path} is missing the column(s) {missing}, which name a corpus and its cadence")
    return {(row["Domain"], row["Dataset"]): parse_time_axis(row["Freq"]) for row in rows}


def item_ids(root: Path) -> tuple[str, ...]:
    """Return the canonical enumeration: one id per row, in shard then row order.

    The ids come from the connector's own :func:`record_id`, so they are the TimeF record ids and the
    item registry can gate the two sides against each other. They are zero-padded, so this order is
    also the sorted order the TimeF side stores its records in.

    Args:
        root: The release directory.

    Returns:
        The item ids.
    """
    from timenet_connectors.datasets.leochen085.slip_pretrain.connector import record_id  # noqa: PLC0415

    return tuple(record_id(shard.index, row) for shard in read_layout(root) for row in range(shard.n_rows))


def n_items(root: Path) -> int:
    """Return how many canonical items the release holds, without building their ids.

    Args:
        root: The release directory.

    Returns:
        The row count over every shard.
    """
    return sum(shard.n_rows for shard in read_layout(root))


def tier_a_bytes(root: Path) -> int:
    """Return the bytes this lane opens: the 17 shards plus the control table.

    Args:
        root: The release directory.

    Returns:
        The apparent size of those files.

    Raises:
        LaneError: If the control table is not beside the shards.
    """
    meta = root / META_CSV
    if not meta.is_file():
        raise LaneError(f"no {META_CSV} in {root}, so no window of the release has a time axis")
    return meta.stat().st_size + sum(shard.path.stat().st_size for shard in read_layout(root))


def decode_signals(listed: Any) -> tuple[np.ndarray, ...]:
    """Decode one row's signals to float32, keeping every value the release published.

    A signal of nothing but NaN is a signal, so it comes back whole rather than dropped or filled.
    Both variants decode here, so they differ in what they read rather than in what they decode.

    Args:
        listed: The signals of one ``time_series`` cell.

    Returns:
        One float32 array per signal, in stored order.
    """
    return tuple(signal.values.to_numpy(zero_copy_only=False) for signal in listed)


class _ShardCursor:
    """A forward, batched read over one shard, projected to one set of columns.

    It follows :class:`~timenet_connectors.readers.ParquetRowReader`: a memory map, one row group of
    :data:`BATCH_ROWS`-row batches at a time, one batch alive. It differs in reading a whole row in
    one pass rather than one column per pass, which is what a read of a row costs.
    """

    def __init__(self, path: Path, columns: list[str] | None) -> None:
        """Open a shard.

        Args:
            path: The shard.
            columns: The columns to project to, or ``None`` for every column the shard ships.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415 - a lane dependency

        self._handle = pq.ParquetFile(path, memory_map=True)
        self._columns = columns
        self._group = -1
        self._batch: Any = None
        self._start = 0
        self._batches: Iterator[Any] = iter(())

    def row(self, row_group: int, row: int) -> tuple[Any, int]:
        """Return the batch that holds one row, and where in it.

        A rising row of the open row group continues the pass. Any other row starts the pass again
        at the first row of its row group.

        Args:
            row_group: Row-group index in the shard.
            row: Row index inside that row group.

        Returns:
            The record batch and the row's offset in it.
        """
        if row_group != self._group or row < self._start:
            self._open(row_group)
        while self._batch is None or row - self._start >= self._batch.num_rows:
            if self._batch is not None:
                self._start += self._batch.num_rows
            self._batch = next(self._batches)
        return self._batch, row - self._start

    def close(self) -> None:
        """Drop the batch in hand and close the shard."""
        self._batch = None
        self._batches = iter(())
        self._handle.close()

    def _open(self, row_group: int) -> None:
        """Start a batch iterator at the first row of one row group.

        Args:
            row_group: Row-group index in the shard.
        """
        self._group = row_group
        self._batch = None
        self._start = 0
        self._batches = self._handle.iter_batches(batch_size=BATCH_ROWS, row_groups=[row_group], columns=self._columns)


@dataclass
class SlipHandle:
    """An open release directory, delivering rows as one consumer's items.

    Attributes:
        shards: The canonical enumeration, one entry per shard in shard order.
        axes: Per source corpus, the time axis its windows sit on.
        consumer: ``pandas`` or ``torch``.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        cursors: The open shards, at most :data:`OPEN_SHARDS` of them.
        starts: Canonical ordinal of each shard's first row.
        total: How many items the release holds.
    """

    shards: tuple[Shard, ...]
    axes: dict[tuple[str, str], TimeAxis]
    consumer: str
    loader_variant: str
    cursors: OrderedDict[int, _ShardCursor] = field(default_factory=OrderedDict)
    starts: tuple[int, ...] = field(init=False, default=())
    total: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Fold the layout into the two figures the walk needs, so it rebuilds neither per item."""
        self.starts = tuple(shard.first_ordinal for shard in self.shards)
        self.total = sum(shard.n_rows for shard in self.shards)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The row count over every shard.
        """
        return self.total

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of rows and convert them to the consumer's type.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal, in the order asked for.
        """
        from timenet_connectors.datasets.leochen085.slip_pretrain.connector import record_id  # noqa: PLC0415

        for ordinal in ordinals:
            shard, row = self._locate(ordinal)
            item_id = record_id(shard.index, row)
            buffers = self._buffers(item_id, self._read(shard, row))
            yield DeliveredItem(ordinal=ordinal, item_id=item_id, buffers=buffers)

    def group_of(self, ordinal: int) -> object:
        """Return the row group this item came from.

        Read off the shard footers outside every timed region.

        Returns:
            The shard's file name and its row-group index.
        """
        shard, row = self._locate(ordinal)
        return (shard.path.name, shard.locate(row)[0])

    def close(self) -> None:
        """Close every shard the handle was holding open."""
        for cursor in self.cursors.values():
            cursor.close()
        self.cursors.clear()

    def _locate(self, ordinal: int) -> tuple[Shard, int]:
        """Map a canonical ordinal to its shard and its row in that shard.

        Returns:
            The shard and the row index inside it.

        Raises:
            LaneError: If the ordinal is outside the enumeration.
        """
        if not 0 <= ordinal < self.total:
            raise LaneError(f"a request asked for an ordinal outside the {self.total} items")
        shard = self.shards[bisect.bisect_right(self.starts, ordinal) - 1]
        return shard, ordinal - shard.first_ordinal

    def _cursor(self, shard: Shard) -> _ShardCursor:
        """Return the open cursor of one shard, opening it and closing an old one as needed.

        Returns:
            The cursor.
        """
        held = self.cursors.get(shard.index)
        if held is not None:
            self.cursors.move_to_end(shard.index)
            return held
        columns = None if self.loader_variant == AS_SHIPPED else list(ITEM_COLUMNS)
        cursor = _ShardCursor(shard.path, columns)
        self.cursors[shard.index] = cursor
        while len(self.cursors) > OPEN_SHARDS:
            _, evicted = self.cursors.popitem(last=False)
            evicted.close()
        return cursor

    def _read(self, shard: Shard, row: int) -> Window:
        """Read one row: its labels, its four captions and its signals.

        Returns:
            The decoded window.

        Raises:
            LaneError: If the shard does not carry a column the item needs.
        """
        from timenet_connectors.datasets.leochen085.slip_pretrain.connector import signals  # noqa: PLC0415

        group, offset = shard.locate(row)
        try:
            batch, position = self._cursor(shard).row(group, offset)
            captions = tuple(str(batch[column][position].as_py()) for column in CAPTION_COLUMNS)
            return Window(
                category=str(batch[CATEGORY_COLUMN][position].as_py()),
                dataset=str(batch[DATASET_COLUMN][position].as_py()),
                captions=captions,
                signals=decode_signals(signals(batch[VALUES_COLUMN][position])),
            )
        except KeyError as error:
            raise LaneError(f"the SLIP shard {shard.path.name} carries no column {error}") from error

    def _buffers(self, item_id: str, window: Window) -> tuple[Buffer, ...]:
        """Build the consumer's own type and return its value planes.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the row names a source corpus ``meta.csv`` does not describe, or the
                consumer is not one this lane implements.
        """
        key = (window.category, window.dataset)
        if key not in self.axes:
            raise LaneError(
                f"item {item_id} names the source corpus {key}, which {META_CSV} does not describe, "
                f"so its time axis is unknown"
            )
        if self.consumer == PANDAS:
            axis = self.axes[key]
            frame = array_frame(item_id, [(f"ch{index}", axis, values) for index, values in enumerate(window.signals)])
            return array_buffers(frame)
        if self.consumer == TORCH:
            import torch  # noqa: PLC0415 - as above

            return tuple(as_buffer(torch.from_numpy(_writable(values)).numpy()) for values in window.signals)
        raise LaneError(f"the SLIP lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class SlipLane:
    """One Original lane over the SLIP release: a directory, a consumer and a read variant.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The release directory holding ``data/`` and ``meta.csv``.
        tier_a_bytes: Tier A bytes of this release, which sizes the blocks.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        dataset: Dataset id.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    tier_a_bytes: int
    loader_variant: str = LAZY_CAPABLE
    dataset: str = DATASET
    name: str = ""
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the variant is unknown, or Tier A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the SLIP lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in {AS_SHIPPED, LAZY_CAPABLE}:
            raise LaneError(f"the SLIP lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {self.loader_variant!r}")
        if self.tier_a_bytes < 1:
            raise LaneError(f"the SLIP lane needs positive Tier A bytes to size its blocks, got {self.tier_a_bytes}")
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{ORIGINAL}/{self.consumer}/{self.loader_variant}")

    def preload(self) -> None:
        """Import everything this lane reads with, without touching the release directory.

        Both variants read Parquet through pyarrow and parse the control table with the connector's
        own frequency table, and the consumer decides the delivered type. All of them are imported on
        first use in the normal path, which is right for a timed read but wrong for the memory
        baseline. This pulls them forward so the baseline can be taken after them, which matters most
        for torch: it costs a few hundred megabytes before it touches data.
        """
        modules = [
            "pyarrow.parquet",
            "timenet.dataset.axis",
            "timenet_connectors.datasets.leochen085.slip_pretrain.connector",
            "timenet_connectors.datasets.leochen085.slip_pretrain.meta",
            *(("pandas", "timenet.pandas") if self.consumer == PANDAS else ("torch",)),
        ]
        for module in modules:
            importlib.import_module(module)

    def open(self) -> SlipHandle:
        """Open the release: read the shard footers and the control table.

        No value is decoded here. The footers are what it takes to address a row, and the table is
        what it takes to lay one out in time, so opening costs 18 small reads and the first delivery
        costs the one batch it needs.

        Returns:
            The open handle.

        Raises:
            LaneError: If the release directory is not there.
        """
        if not (self.root / META_CSV).is_file():
            raise LaneError(f"lane {self.name} could not open {self.root}: no {META_CSV} in it")
        return SlipHandle(
            shards=read_layout(self.root),
            axes=source_axes(self.root),
            consumer=self.consumer,
            loader_variant=self.loader_variant,
        )


def _writable(values: np.ndarray) -> np.ndarray:
    """Return an array torch can wrap without copying it twice.

    Arrow hands out a read-only view of its own buffer, and ``torch.from_numpy`` refuses one. The
    TimeF side copies for the same reason, so this costs the two sides the same.

    Returns:
        The array itself when it is already writable, else a copy.
    """
    return values if values.flags.writeable else values.copy()
