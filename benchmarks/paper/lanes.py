"""The lane protocol: four operations both representations implement the same way.

A lane is one (dataset, representation, consumer, loader) path from a directory on disk to items in
the consumer's own type. It offers four things: open, first item, full read, and iterate in a given
order.

Full read is not a separate implementation. It is :func:`read_pass` over the ``B = n_items`` plan,
running the identical delivery loop with the identical checksum, so the only difference between the
two access patterns is the permutation. Without that, "sequential bounds block-shuffled from above"
is a statement about two differently-constructed measurements rather than about the format.

The checksum runs inside the timed region and touches every value. That is what makes a full read a
full read: a loader that returns lazy handles and never decodes them would otherwise post an
excellent time.

Every lane also reports the bytes it moved, so a rate cell carries bytes/s beside items/s. Without
that pair the rate columns are algebraically entangled with the storage column, and the paper cannot
tell the reader whether a row is I/O bound or decode bound.

The Original side of this protocol is a later phase. It implements :class:`Lane` and
:class:`LaneHandle`, and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from benchmarks.paper.blocks import (
    DEFAULT_SEED,
    BlockPlan,
    Locality,
    SwitchStats,
    locality_of,
    plan_blocks,
    plan_full_read,
    switch_stats,
)
from benchmarks.paper.errors import LaneError


Buffer = memoryview | bytes
"""What a lane hands the checksum: one contiguous view per value plane of the item."""

DEFAULT_REQUEST_ITEMS = 256
"""Largest number of items a lane asks its storage for at once.

Both passes use it, so a full read and a block-shuffled read issue the same shape of request and
differ only in which items each request holds. A full read that asked for all 458,161 ids in one
predicate would be a different code path, and then the two numbers would not be comparable.
"""


@dataclass(frozen=True)
class DeliveredItem:
    """One canonical item, in the consumer's own type, ready to be checksummed.

    Attributes:
        ordinal: The item's position in the canonical enumeration order.
        item_id: The item's identity.
        buffers: One contiguous view per value plane. The checksum walks all of them.
    """

    ordinal: int
    item_id: str
    buffers: tuple[Buffer, ...]

    @property
    def nbytes(self) -> int:
        """Return how many bytes this item delivered.

        Returns:
            The sum over the buffers.
        """
        return sum(_nbytes(buffer) for buffer in self.buffers)


@runtime_checkable
class LaneHandle(Protocol):
    """An opened lane. It holds whatever the loader needs and delivers items on request."""

    def n_items(self) -> int:
        """Return how many canonical items this lane holds."""
        ...

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of items.

        Args:
            ordinals: Canonical ordinals to read. A lane may return them in its own stored order.

        Yields:
            One item per ordinal.
        """
        ...

    def group_of(self, ordinal: int) -> object:
        """Return the storage group an item came from, or ``None`` when the lane cannot say.

        A group is a Parquet row group on the TimeF side and a source file on the Original side.
        This is called after the clock stops, so it never enters a timed region.
        """
        ...

    def close(self) -> None:
        """Release whatever the lane opened."""
        ...


@runtime_checkable
class Lane(Protocol):
    """One measured path from a directory on disk to items in a consumer's type."""

    dataset: str
    representation: str
    consumer: str
    loader_variant: str
    name: str
    tier_a_bytes: int

    def preload(self) -> None:
        """Import everything this lane needs, without opening anything.

        The memory metric takes its baseline between this and :meth:`open`. A lane that imports its
        consumer lazily on the first delivery would otherwise charge that import to the read, and
        torch costs a few hundred megabytes before it touches data.
        """
        ...

    def open(self) -> LaneHandle:
        """Open the lane. Everything expensive happens here or later, never in the constructor."""
        ...


@dataclass(frozen=True)
class PassResult:
    """What one timed pass produced.

    Attributes:
        seconds: Wall clock of the timed region, including the checksum.
        items: Items delivered.
        bytes_touched: Bytes the checksum walked.
        checksum: Digest of every value touched, computed inside the timed region.
        plan: The block plan the pass ran.
        locality: What locality the emitted order achieved.
        switches: Group switches, when the lane could name its groups.
    """

    seconds: float
    items: int
    bytes_touched: int
    checksum: str
    plan: BlockPlan
    locality: Locality
    switches: SwitchStats | None = None

    @property
    def items_per_s(self) -> float:
        """Return the item rate.

        Returns:
            Items over seconds.
        """
        return self.items / self.seconds

    @property
    def bytes_per_s(self) -> float:
        """Return the byte rate, recorded beside every item rate.

        Returns:
            Bytes over seconds.
        """
        return self.bytes_touched / self.seconds

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of the pass and its derived rates.
        """
        payload: dict[str, Any] = {
            "seconds": self.seconds,
            "items": self.items,
            "bytes_touched": self.bytes_touched,
            "items_per_s": self.items_per_s,
            "bytes_per_s": self.bytes_per_s,
            "checksum": self.checksum,
            "plan": self.plan.as_dict(),
            "locality": self.locality.as_dict(),
        }
        payload["switches"] = self.switches.as_dict() if self.switches is not None else None
        return payload


@dataclass(frozen=True)
class FirstItemResult:
    """What the first-item measurement produced.

    Attributes:
        seconds: From opening the lane to holding the first usable item.
        item_id: Which item came back first.
        bytes_touched: Bytes that item delivered.
        checksum: Digest of that item's values.
    """

    seconds: float
    item_id: str
    bytes_touched: int
    checksum: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field.
        """
        return {
            "seconds": self.seconds,
            "item_id": self.item_id,
            "bytes_touched": self.bytes_touched,
            "checksum": self.checksum,
        }


@contextmanager
def opened(lane: Lane) -> Iterator[LaneHandle]:
    """Open a lane and close it again.

    Yields:
        The open handle.

    Raises:
        LaneError: If the lane could not be opened.
    """
    try:
        handle = lane.open()
    except LaneError:
        raise
    except Exception as error:
        raise LaneError(f"lane {lane.name} could not open: {error}") from error
    try:
        yield handle
    finally:
        handle.close()


def first_item(lane: Lane) -> FirstItemResult:
    """Time the path from opening a lane to holding its first usable item.

    The clock covers the open, because that is what the column claims to measure: how long before a
    consumer can start work. A representation that must materialize a corpus before it can hand over
    one item pays for that here, which is the point.

    Args:
        lane: The lane to measure.

    Returns:
        The elapsed seconds and the item that came back.

    Raises:
        LaneError: If the lane delivered nothing.
    """
    hasher = hashlib.blake2b(digest_size=16)
    handle: LaneHandle | None = None
    try:
        started = time.perf_counter()
        handle = lane.open()
        item = next(iter(handle.deliver((0,))), None)
        if item is None:
            raise LaneError(f"lane {lane.name} delivered no first item")
        touched = _absorb(hasher, item)
        elapsed = time.perf_counter() - started
        return FirstItemResult(
            seconds=max(elapsed, 1e-9),
            item_id=item.item_id,
            bytes_touched=touched,
            checksum=hasher.hexdigest(),
        )
    finally:
        if handle is not None:
            handle.close()


def read_pass(
    handle: LaneHandle,
    plan: BlockPlan,
    *,
    request_items: int = DEFAULT_REQUEST_ITEMS,
) -> PassResult:
    """Run one pass over a lane and time it.

    This is the only read loop in the harness. A full read is this function over
    :func:`~benchmarks.paper.blocks.plan_full_read`, and a block-shuffled read is this function over
    :func:`~benchmarks.paper.blocks.plan_blocks`, so the two differ in nothing but the permutation.

    Locality and switch statistics are computed after the clock stops, from the order the lane
    actually delivered.

    Args:
        handle: An open lane.
        plan: Which items, in which order, in which blocks.
        request_items: Largest number of items in one request to the lane.

    Returns:
        The pass.

    Raises:
        LaneError: If ``request_items`` is below one, or the lane delivered the wrong item count.
    """
    if request_items < 1:
        raise LaneError(f"a request must hold at least one item, got {request_items}")
    hasher = hashlib.blake2b(digest_size=16)
    delivered: list[int] = []
    touched = 0

    started = time.perf_counter()
    for block in plan.blocks:
        for start in range(0, len(block), request_items):
            for item in handle.deliver(block[start : start + request_items]):
                touched += _absorb(hasher, item)
                delivered.append(item.ordinal)
    elapsed = time.perf_counter() - started

    if len(delivered) != plan.n_items:
        raise LaneError(f"the plan holds {plan.n_items} items but the lane delivered {len(delivered)}")
    groups = [handle.group_of(ordinal) for ordinal in delivered]
    return PassResult(
        seconds=max(elapsed, 1e-9),
        items=len(delivered),
        bytes_touched=touched,
        checksum=hasher.hexdigest(),
        plan=plan,
        locality=locality_of(delivered),
        switches=switch_stats(groups) if any(group is not None for group in groups) else None,
    )


def full_read(handle: LaneHandle, tier_a_bytes: int, *, request_items: int = DEFAULT_REQUEST_ITEMS) -> PassResult:
    """Read every item once, in canonical order, touching every value.

    Returns:
        The pass.
    """
    plan = plan_full_read(n_items=handle.n_items(), tier_a_bytes=tier_a_bytes)
    return read_pass(handle, plan, request_items=request_items)


def block_shuffled(
    handle: LaneHandle,
    tier_a_bytes: int,
    *,
    seed: int = DEFAULT_SEED,
    request_items: int = DEFAULT_REQUEST_ITEMS,
) -> PassResult:
    """Read every item once, with the blocks permuted and canonical order kept inside a block.

    Returns:
        The pass.
    """
    plan = plan_blocks(n_items=handle.n_items(), tier_a_bytes=tier_a_bytes, seed=seed)
    return read_pass(handle, plan, request_items=request_items)


def as_buffer(values: Any) -> Buffer:
    """Return a contiguous view of an array, for the checksum.

    A numeric array becomes a memoryview of its own bytes, with no copy when it is already
    contiguous. Anything the buffer protocol cannot express, such as an object column of Python
    strings, is encoded to UTF-8 instead. Neither path changes the values; both make them hashable.

    Args:
        values: A numpy array, a string, or bytes.

    Returns:
        A view the checksum can walk.

    Raises:
        LaneError: If the value has no byte form at all.
    """
    if isinstance(values, bytes | bytearray | memoryview):
        return memoryview(values)
    if isinstance(values, str):
        return values.encode("utf-8")
    if isinstance(values, np.ndarray):
        if values.dtype.kind in "OUS":
            return "\x1f".join(str(value) for value in values.ravel().tolist()).encode("utf-8")
        return memoryview(np.ascontiguousarray(values)).cast("B")
    raise LaneError(f"a lane delivered {type(values).__name__}, which has no byte form to check")


def array_frame(item_id: str, signals: Sequence[tuple[str, Any, Any]]) -> Any:
    """Build the array-cell pandas item: one row, one column per signal, each cell a value array.

    Every lane of this harness delivers this shape, so it is built here once rather than seven
    times. The Original side calls this and the TimeF side calls
    :func:`timenet.pandas.record_array_frame`, which builds the same frame off a record. Two
    builders, because an Original lane holds no record and must not construct one: that would put
    TimeF's own object graph inside a measurement of the release.

    Args:
        item_id: The canonical item id, which is the record id both sides carry.
        signals: Per signal, its name, its :class:`~timenet.dataset.axis.TimeAxis` and its values.
            The order is the column order, and both sides have to agree on it.

    Returns:
        The one-row frame.

    Raises:
        LaneError: If two signals of one item carry the same name. The SDK's builder gives those
            two columns by appending the series id, and no release in this campaign holds one, so a
            lane that hit it would be delivering a different item from the TimeF side rather than
            the same one.
    """
    # Lane dependencies, imported here rather than at module import: a torch lane needs neither.
    import pandas as pd  # noqa: PLC0415

    from timenet.pandas import ARRAY_META_COLUMNS, RECORD_ID_COLUMN, TIME_AXIS_COLUMN  # noqa: PLC0415

    names = [signal for signal, _, _ in signals]
    if len(set(names)) != len(names) or set(names) & set(ARRAY_META_COLUMNS):
        raise LaneError(f"item {item_id} names the signals {sorted(names)}, which are not one column each")
    columns: dict[str, Any] = {
        RECORD_ID_COLUMN: [item_id],
        TIME_AXIS_COLUMN: [{signal: axis for signal, axis, _ in signals}],
    }
    for signal, _, values in signals:
        columns[signal] = [values]
    return pd.DataFrame(columns, copy=False)


def array_buffers(frame: Any) -> tuple[Buffer, ...]:
    """Return the value planes of an array-cell frame, for the checksum.

    The signal cells go in as themselves, so what the checksum walks is the values and not a
    restatement of them. The two meta cells go in as text: the record id, and the axes keyed by
    signal name. That keeps the signal names and their cadences inside the digest, which the column
    headers alone would leave out, at a cost of a hundred-odd bytes against an item of values.

    Args:
        frame: A frame from :func:`array_frame` or
            :func:`timenet.pandas.record_array_frame`.

    Returns:
        One buffer per column, in column order.
    """
    from timenet.pandas import ARRAY_META_COLUMNS  # noqa: PLC0415 - a lane dependency

    meta = len(ARRAY_META_COLUMNS)
    row = frame.iloc[0]
    return (
        *(as_buffer(str(cell)) for cell in row.iloc[:meta]),
        *(as_buffer(cell) for cell in row.iloc[meta:]),
    )


@dataclass
class RecordingHandle:
    """A handle that delivers a fixed list of items, for exercising the protocol without storage.

    Attributes:
        items: One entry per canonical ordinal.
        requests: The requests the pass issued, recorded in order.
        groups: Optional group id per ordinal.
    """

    items: tuple[DeliveredItem, ...]
    groups: tuple[object, ...] = ()
    requests: list[tuple[int, ...]] = field(default_factory=list)
    closed: bool = False

    def n_items(self) -> int:
        """Return how many items this handle holds.

        Returns:
            The item count.
        """
        return len(self.items)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Return the requested items, recording the request.

        Yields:
            One item per ordinal.
        """
        self.requests.append(tuple(ordinals))
        for ordinal in ordinals:
            yield self.items[ordinal]

    def group_of(self, ordinal: int) -> object:
        """Return the group an item came from.

        Returns:
            The group id, or ``None`` when this handle declares none.
        """
        return self.groups[ordinal] if self.groups else None

    def close(self) -> None:
        """Mark the handle closed."""
        self.closed = True


def _absorb(hasher: Any, item: DeliveredItem) -> int:
    """Fold one item into the running checksum.

    The item id goes in as well as the values, so a pass that delivers the right values against the
    wrong items does not check out.

    Returns:
        How many bytes the item contributed.
    """
    hasher.update(item.item_id.encode("utf-8"))
    total = 0
    for buffer in item.buffers:
        hasher.update(buffer)
        total += _nbytes(buffer)
    return total


def _nbytes(buffer: Buffer) -> int:
    """Return how many bytes a buffer holds.

    Returns:
        The byte count.
    """
    return buffer.nbytes if isinstance(buffer, memoryview) else len(buffer)
