"""The TimeF side of the lane protocol, for the pandas and the PyTorch consumer.

Both consumers stream. The lane holds a :class:`~timenet.reader.TimeFReader` and asks it for one
request of record ids at a time, so the resident set is one request rather than the corpus. That is
what makes the full read of a corpus larger than memory possible at all, and it is what makes the
first-item cell measure opening rather than materializing.

The two consumers differ only in what an item becomes: a pandas array-cell frame through
``timenet.pandas``, or a tuple of tensors through ``timenet.torch``. Everything else, the request
shape, the block plan and the checksum, is shared with the Original side through
:mod:`benchmarks.paper.lanes`.

The pandas item is one row per record with one column per signal, not the long frame this lane used
to build. A long frame restates the record id, the series id and the signal name on every row, which
on a Sleep-EDF record is 2.35 GB against 96 MB of values, and building it was 99.7 percent of what
the pandas cells measured. It also refused a record whose series disagree on dtype, so it could
never have carried the mixed-rate rows at all.

The lane can also say which row group each record's first chunk landed in, read off
``time_series_index`` outside the timed region, which is what turns the locality claim into a
measured switch count.

An item is a record on five of the six rows. Sleep-EDF is the sixth: a record there is a whole
polysomnogram and the canonical item is one scored 30 s epoch of it, so the lane cuts the epoch
rather than delivering the recording. Set ``epoch_us`` and the lane reads the epoch's step window
out of each series through :meth:`~timenet.dataset.TimeSeries.read_steps`, which resolves to the
values backend's own range read and touches only the chunks the window crosses. That is the
comparison the row is for: the Original side has to decode a 100 MB recording to reach one epoch.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import NATIVE, PANDAS, TIMEF, TORCH


INDEX_GLOB = "time_series_index/part-*.parquet"
"""Where the reader's index parts live. It is a directory of parts, not one file."""

EPOCH_SEPARATOR = "#"
"""What an epoch item id puts between the record it sits on and its epoch index."""

OPEN_RECORDS = 8
"""How many records an epoch lane keeps resolved at once.

A resolved record is its axes and its lazy loaders, not its samples, so this holds kilobytes. The
Original side's lazy variant holds the same number of recordings open, and both are there for the
same reason: consecutive epochs sit on one record, so a request that reopened it per epoch would
measure the lookup rather than the read.
"""

EpochPlane = tuple[str, Any, np.ndarray]
"""One series' window of one epoch: its signal name, the axis of the window, and its values."""


@dataclass
class TimeFHandle:
    """An open TimeF version, delivering records as one consumer's items.

    Attributes:
        reader: The open reader. It resolves nothing until an item asks for it.
        item_ids: The canonical enumeration order, one item id per ordinal.
        consumer: ``pandas`` or ``torch``.
        groups: Per record id, the row group its first chunk landed in.
        epoch_us: Length of one epoch, for a lane whose item is a window of a record. ``0`` makes
            the item the record itself.
        max_open: How many records an epoch lane keeps resolved at once.
    """

    reader: Any
    item_ids: tuple[str, ...]
    consumer: str
    groups: dict[str, object] = field(default_factory=dict)
    epoch_us: int = 0
    max_open: int = OPEN_RECORDS
    _open: OrderedDict[str, tuple[Any, ...]] = field(default_factory=OrderedDict, repr=False)
    _ordinal_of: dict[str, int] = field(default_factory=dict, repr=False)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.item_ids)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of records and convert them to the consumer's type.

        The reader prunes the row groups that cannot hold the requested ids, and it returns them in
        stored order rather than the order they were asked for, so the ordinal comes back off the
        record id rather than off the request.

        An epoch lane takes the other path: an item is a window of a record, so it reads the window
        per item and keeps the order the request asked for.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal.

        Raises:
            LaneError: If an ordinal is outside the enumeration, or the reader returned a record
                that is not in it.
        """
        if self.epoch_us:
            yield from self._deliver_epochs(ordinals)
            return
        wanted = self._record_ids(ordinals)
        ordinal_of = self._ordinals()
        # The checksum walks value planes, and the Original lane delivers values and nothing else.
        # Resolving annotations would charge this side a lookup and a JSON parse per record for
        # data neither side compares.
        for record in self.reader.iter_records(list(wanted), with_annotations=False):
            ordinal = ordinal_of.get(record.record_id)
            if ordinal is None:
                raise LaneError(f"the reader returned record {record.record_id!r}, which is not a canonical item")
            yield DeliveredItem(ordinal=ordinal, item_id=record.record_id, buffers=self._buffers(record))

    def group_of(self, ordinal: int) -> object:
        """Return the row group this item's first chunk landed in.

        Read off the index outside every timed region. An epoch sits in the row groups of the
        record it was cut from, which is what the index names.

        Returns:
            The group, or ``None`` when the index was not loaded.
        """
        item_id = self.item_ids[ordinal]
        return self.groups.get(split_epoch_id(item_id)[0] if self.epoch_us else item_id)

    def close(self) -> None:
        """Close the reader and release its shard handles."""
        self._open.clear()
        self.reader.close()

    def _ordinals(self) -> dict[str, int]:
        """Return the map from record id to ordinal, built one time.

        The reader returns records in stored order. A request must therefore look each record up
        to find the ordinal that it asked under. A map built for each request costs the whole
        enumeration each time. SLIP holds 618,508 items and a request holds 256, so one read makes
        1.5 billion entries.

        Returns:
            One entry for each canonical item.
        """
        if not self._ordinal_of:
            self._ordinal_of = {record_id: ordinal for ordinal, record_id in enumerate(self.item_ids)}
        return self._ordinal_of

    def _record_ids(self, ordinals: Sequence[int]) -> tuple[str, ...]:
        """Map canonical ordinals to record ids.

        Returns:
            The ids, in the order asked for.

        Raises:
            LaneError: If an ordinal is outside the enumeration.
        """
        try:
            return tuple(self.item_ids[ordinal] for ordinal in ordinals)
        except IndexError as error:
            raise LaneError(f"a request asked for an ordinal outside the {len(self.item_ids)} items") from error

    def _deliver_epochs(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of epochs, one step window per series of the owning record.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal, in the order asked for.
        """
        for ordinal, item_id in zip(ordinals, self._record_ids(ordinals), strict=True):
            record_id, index = split_epoch_id(item_id)
            planes = self._epoch_planes(record_id, index * self.epoch_us)
            yield DeliveredItem(ordinal=ordinal, item_id=item_id, buffers=self._epoch_buffers(record_id, planes))

    def _epoch_planes(self, record_id: str, start_us: int) -> tuple[EpochPlane, ...]:
        """Read one epoch out of every series of a record, each at its own cadence.

        The window is clipped where the series stops, because a scoring may run past the signals it
        scores: 509 Sleep-EDF epochs do, and the connector wrote them anyway. Those deliver the
        samples that exist, which for most of them is none, and that is what the Original side
        delivers for them too.

        Returns:
            One plane per series, in stored order.

        Raises:
            LaneError: If a series states no cadence, so no epoch can be cut on it.
        """
        # Lane dependencies, imported here rather than at module import.
        import pyarrow as pa  # noqa: PLC0415

        from timenet.dataset.axis import RegularAxis  # noqa: PLC0415

        planes: list[EpochPlane] = []
        for series in self._series(record_id):
            axis = series.time_axis
            if not isinstance(axis, RegularAxis):
                raise LaneError(
                    f"series {series.time_series_id!r} of record {record_id!r} carries "
                    f"{type(axis).__name__}, which states no cadence to cut an epoch on"
                )
            first = int(start_us // axis.period_us)
            window = series.read_steps(first, first + int(self.epoch_us // axis.period_us))
            values = (
                window.to_numpy_ndarray()
                if isinstance(window, pa.FixedShapeTensorArray)
                else window.to_numpy(zero_copy_only=False)
            )
            planes.append((series.signal, axis.at_index(first), values))
        return tuple(planes)

    def _series(self, record_id: str) -> tuple[Any, ...]:
        """Return one record's series, keeping a few records resolved and evicting the rest.

        Returns:
            The series, in stored order. Their values are still unread.

        Raises:
            LaneError: If the artifact holds no such record.
        """
        held = self._open.get(record_id)
        if held is not None:
            self._open.move_to_end(record_id)
            return held
        record = next(iter(self.reader.iter_records([record_id])), None)
        if record is None:
            raise LaneError(f"the artifact holds no record {record_id!r}, which an epoch item names")
        self._open[record_id] = record.time_series
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return record.time_series

    def _epoch_buffers(self, record_id: str, planes: tuple[EpochPlane, ...]) -> tuple[Buffer, ...]:
        """Convert one epoch into the consumer's type and return its value planes.

        The SDK builds both consumer types off a whole record, and an epoch is a window of one, so
        this builds the same two shapes off the windows instead: the array-cell frame
        :func:`~timenet.pandas.record_array_frame` returns, and one tensor per series.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            return array_buffers(array_frame(record_id, planes))
        if self.consumer == TORCH:
            import torch  # noqa: PLC0415 - the lane's own consumer, as in _buffers

            # copy() gives a writable C-contiguous array, which is what timenet.torch does with a
            # whole series and for the same reason: Arrow's zero-copy view is read-only.
            return tuple(as_buffer(torch.from_numpy(values.copy()).numpy()) for _, _, values in planes)
        raise LaneError(f"the TimeF lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")

    def _buffers(self, record: Any) -> tuple[Buffer, ...]:
        """Convert one record into the consumer's type and return its value planes.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            # The lane's own consumer, imported once per lane rather than at module import.
            from timenet.pandas import record_array_frame  # noqa: PLC0415

            return array_buffers(record_array_frame(record))
        if self.consumer == TORCH:
            from timenet.torch import record_item  # noqa: PLC0415 - as above

            return tuple(as_buffer(tensor.numpy()) for tensor in record_item(record)["series"])
        raise LaneError(f"the TimeF lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class TimeFLane:
    """One TimeF lane: a version directory, an enumeration, and a consumer.

    Attributes:
        dataset: Dataset id.
        consumer: ``pandas`` or ``torch``.
        root: The committed version directory.
        item_ids: The canonical enumeration order, one item id per ordinal.
        tier_a_bytes: Tier A bytes of this artifact, which sizes the blocks.
        name: The lane name that reaches the cell record.
        with_groups: Read the index for switch statistics when the lane opens.
        epoch_us: Length of one epoch, for a row whose item is a window of a record rather than the
            record. Every item id is then ``<record id>#<epoch index>``. ``0`` makes the item the
            record itself, which is what five of the six rows carry.
    """

    dataset: str
    consumer: str
    root: Path
    item_ids: tuple[str, ...]
    tier_a_bytes: int
    name: str = ""
    with_groups: bool = False
    epoch_us: int = 0
    representation: str = TIMEF
    loader_variant: str = NATIVE

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer is unknown, the enumeration is empty, Tier A bytes are not
                positive, or the epoch length is negative.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the TimeF lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if not self.item_ids:
            raise LaneError(f"the TimeF lane for {self.dataset} has no items to enumerate")
        if self.tier_a_bytes < 1:
            raise LaneError(f"the TimeF lane for {self.dataset} needs positive Tier A bytes to size its blocks")
        if self.epoch_us < 0:
            raise LaneError(f"the TimeF lane for {self.dataset} needs a non-negative epoch, got {self.epoch_us}")
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{TIMEF}/{self.consumer}")

    def preload(self) -> None:
        """Import the reader and this lane's consumer, without touching the version directory.

        The consumer is imported on first delivery in the normal path, which is right for a timed
        read but wrong for the memory baseline. This pulls the import forward so the baseline can be
        taken after it.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the TimeF lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")
        # Lane dependencies, pulled in here rather than at module import.
        for module in ("timenet.reader", "timenet.registry", f"timenet.{self.consumer}"):
            importlib.import_module(module)

    def open(self) -> TimeFHandle:
        """Open the version.

        The reader decodes nothing here: it holds the parsed manifest and resolves tasks,
        annotations and values on first use. So this is the cheap half of the first-item cell, and
        the expensive half is the first delivery.

        Returns:
            The open handle.

        Raises:
            LaneError: If the version directory cannot be opened.
        """
        # Imported per lane, not at module import: the harness stays importable without a build.
        from timenet.reader import TimeFReader  # noqa: PLC0415
        from timenet.registry import DatasetVersion  # noqa: PLC0415

        try:
            reader = TimeFReader(DatasetVersion.open_local(self.root))
        except (OSError, ValueError) as error:
            raise LaneError(f"lane {self.name} could not open {self.root}: {error}") from error
        groups = row_groups_by_record(self.root) if self.with_groups else {}
        return TimeFHandle(
            reader=reader,
            item_ids=self.item_ids,
            consumer=self.consumer,
            groups=groups,
            epoch_us=self.epoch_us,
        )


def split_epoch_id(item_id: str) -> tuple[str, int]:
    """Split an epoch item id into the record it sits on and its epoch index.

    The id is the one :func:`~benchmarks.paper.items.epoch_grid_items` writes on the Original side,
    so the two representations address the same epoch by the same name.

    Args:
        item_id: An id of the form ``<record id>#<epoch index>``.

    Returns:
        The record id and the epoch index.

    Raises:
        LaneError: If the id names no epoch.
    """
    record_id, separator, index = item_id.rpartition(EPOCH_SEPARATOR)
    if not separator or not record_id or not index.isdigit():
        raise LaneError(f"epoch item {item_id!r} is not '<record id>{EPOCH_SEPARATOR}<epoch index>'")
    return record_id, int(index)


def record_ids_of(root: Path) -> tuple[str, ...]:
    """Return a version's record ids in stored order, which is the canonical enumeration.

    This reads the records control table and nothing else, so it is cheap enough to run once per
    campaign to build the item registry.

    Args:
        root: The committed version directory.

    Returns:
        The record ids, in stored order.

    Raises:
        LaneError: If the version cannot be opened.
    """
    # Imported here, not at module import, so the harness stays importable without a build.
    from timenet.reader import TimeFReader  # noqa: PLC0415
    from timenet.registry import DatasetVersion  # noqa: PLC0415

    try:
        with TimeFReader(DatasetVersion.open_local(root)) as reader:
            return tuple(record.record_id for record in reader.iter_records())
    except (OSError, ValueError) as error:
        raise LaneError(f"could not enumerate {root}: {error}") from error


def epoch_ids_of(root: Path, *, epoch_us: int, target_schema: str, keep_labels: frozenset[str]) -> tuple[str, ...]:
    """Return a version's epoch items, read off the tasks that score them.

    A record of this shape is a whole recording, so the artifact states its items as tasks: one
    classification task per scored window, scoped to that window of that record. This names each
    one the way the Original side names it, and drops the labels the grid drops, so the two sides
    can be checked against each other rather than trusted.

    It runs once per campaign, in the survey, and it decodes every task partition.

    Args:
        root: The committed version directory.
        epoch_us: Length of one epoch, in microseconds.
        target_schema: The vocabulary an epoch task scores against. No other task carries it.
        keep_labels: The labels that produce an item.

    Returns:
        The epoch item ids, in stored task order.

    Raises:
        LaneError: If the version cannot be opened, or an epoch task names no record window.
    """
    # Imported here, not at module import, so the harness stays importable without a build.
    from timenet.reader import TimeFReader  # noqa: PLC0415
    from timenet.registry import DatasetVersion  # noqa: PLC0415
    from timenet.types import ClassificationTask, TimeInterval  # noqa: PLC0415

    try:
        with TimeFReader(DatasetVersion.open_local(root)) as reader:
            scored = reader.tasks
    except (OSError, ValueError) as error:
        raise LaneError(f"could not enumerate the epochs of {root}: {error}") from error
    found: list[str] = []
    for task in scored:
        if not isinstance(task, ClassificationTask) or task.target_schema != target_schema:
            continue
        if task.target not in keep_labels:
            continue
        if not isinstance(task.scope, TimeInterval) or not task.record_ids:
            raise LaneError(f"task {task.id!r} scores {target_schema!r} but names no record window")
        found.append(f"{task.record_ids[0]}{EPOCH_SEPARATOR}{task.scope.start_us // epoch_us:09d}")
    return tuple(found)


def row_groups_by_record(root: Path) -> dict[str, object]:
    """Return the row group each record's first chunk landed in.

    The index is a directory of parts, not one file, so this is a dataset scan. It reads only the
    three columns it needs, and it runs outside every timed region.

    Args:
        root: The committed version directory.

    Returns:
        Per record id, a ``(chunk file, row group)`` pair. Empty when the version has no index.
    """
    # Lane dependencies, imported here rather than at module import.
    import pyarrow.dataset as pads  # noqa: PLC0415

    from timenet.format.schemas import UUID16, IdCodec  # noqa: PLC0415

    parts = sorted(root.glob(INDEX_GLOB))
    if not parts:
        return {}
    scan = pads.dataset([str(part) for part in parts], format="parquet")
    table = scan.to_table(columns=["record_id", "chunk_file", "chunk_major_idx"])
    # The index stores ids the way the records table does, so a uuid16 build holds raw bytes here
    # and decoded strings in Record.record_id. Decode, or nothing joins.
    codec = IdCodec.from_uuid16({"record_id"} if scan.schema.field("record_id").type == UUID16 else set())
    groups: dict[str, object] = {}
    for record_id, chunk_file, major in zip(
        table.column("record_id").to_pylist(),
        table.column("chunk_file").to_pylist(),
        table.column("chunk_major_idx").to_pylist(),
        strict=True,
    ):
        groups.setdefault(codec.decode("record_id", record_id), (chunk_file, major))
    return groups
