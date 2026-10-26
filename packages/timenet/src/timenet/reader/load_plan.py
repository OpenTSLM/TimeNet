"""Load plans, which say which stored values a read loads, and the step that loads them.

A read builds its objects with every stored value behind an unloaded value handle. Before it hands
the objects out, it collects the handles its plan reaches, reads their values in one batch, and
fills the handles. Objects that existed before the read are never changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from enum import StrEnum, unique
from typing import ClassVar

from timenet.dataset import Record, Signal, Source
from timenet.errors import TimeFValidationError
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import PackedLocation, StoredOffsets, StoredText, StoredTextList
from timenet.types import Annotation, Task


@unique
class Loading(StrEnum):
    """How a read delivers one kind of stored value on the objects it builds."""

    LAZY = "lazy"
    """Left behind its value handle, which reads it on access. Only irregular time offsets are kept
    after their first read."""
    EAGER = "eager"
    """Loaded during the read and kept in its value handle, so access needs no storage."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadPlan:
    """Say, for each kind of stored value, whether a read loads it or leaves it lazy.

    A read applies its plan to the objects it builds: a dataset's records, tasks, and annotations; a
    record's signals, time offsets, and annotations; a task's text, annotations, and the records,
    signals, and annotations it references. It loads nothing on objects passed to it, such as the
    records given to ``read_tasks``, and nothing on tasks it builds only as another task's
    ``from_tasks``. A plan never widens what a read reaches.
    """

    signal_values: Loading = Loading.LAZY
    """Signal arrays."""
    time_offsets: Loading = Loading.LAZY
    """Time offsets of irregular axes, one array per axis that its signals share."""
    text: Loading = Loading.LAZY
    """Packed text: prompts, rationales, text targets, and annotation strings and string lists."""

    ALL_LAZY: ClassVar[LoadPlan]
    """Load nothing. Every read's default."""
    ALL_EAGER: ClassVar[LoadPlan]
    """Load every kind, so the objects a read builds need no storage afterwards."""

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Loading):
                raise TimeFValidationError(f"LoadPlan.{field.name} takes Loading.LAZY or Loading.EAGER, got {value!r}")


LoadPlan.ALL_LAZY = LoadPlan()
LoadPlan.ALL_EAGER = LoadPlan(signal_values=Loading.EAGER, time_offsets=Loading.EAGER, text=Loading.EAGER)


_StoredValue = StoredArray | StoredOffsets | StoredText | StoredTextList
_STORED_VALUES = (StoredArray, StoredOffsets, StoredText, StoredTextList)


def _stored_values(roots: Iterable[object], *, parents: bool) -> Iterator[_StoredValue]:
    """Yield the stored value handles on objects and on the objects they reference directly.

    Args:
        roots: Records, sources, signals, annotations, and tasks. A record or source brings its
            sources, signals, and annotations; a task its text, annotations, inputs, candidate
            records, record and signal targets, and the annotations it references.
        parents: Whether to follow each task's ``from_tasks`` as well.

    Yields:
        Each handle once, in no particular order.
    """
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, _STORED_VALUES):
            yield item
        elif isinstance(item, Record):
            stack.extend((*item.annotations, *item.sources))
        elif isinstance(item, Source):
            stack.extend((*item.annotations, *item.signals, *item.sources))
        elif isinstance(item, Signal):
            stack.extend((item._values, *item.annotations))
            if item._time_offsets is not None:
                stack.append(item._time_offsets)
        elif isinstance(item, Annotation):
            stack.append(item.value)
        elif isinstance(item, Task):
            stack.extend((item.prompt, item.rationale, *(item.targets or ()), *item.annotations, *item.inputs))
            stack.extend((*getattr(item, "candidate_records", ()), *item.input_annotations, *item.target_annotations))
            if parents:
                stack.extend(item.from_tasks)


def _packed_order(location: PackedLocation) -> tuple[str, str, int]:
    """Return a sort key that groups packed values by owner release and payload file."""
    return (location.owner or "", location.path, location.row)


def _chunk_order(location: ChunkLocation) -> tuple[str, str, int, int]:
    """Return a sort key that groups arrays by owner release and values file, in storage order."""
    first = location.chunks[0]
    return (location.owner or "", first.chunk_file, first.data_index.major_idx, first.data_index.minor_idx or 0)


class ValueLoad:
    """Collect the unloaded stored values a plan reaches on a read's new objects, then load them.

    The handles on objects the read did not build are left as they are, so a read changes only the
    objects it hands out.
    """

    def __init__(self, plan: LoadPlan, *, keep: Iterable[object] = ()) -> None:
        """Start an empty load.

        Args:
            plan: The read's load plan.
            keep: Objects the caller already holds, such as the records passed to a task read.
        """
        kinds: list[type[_StoredValue]] = []
        if plan.signal_values is Loading.EAGER:
            kinds.append(StoredArray)
        if plan.time_offsets is Loading.EAGER:
            kinds.append(StoredOffsets)
        if plan.text is Loading.EAGER:
            kinds.extend((StoredText, StoredTextList))
        self._kinds = tuple(kinds)
        # Handles to leave alone, by identity. The objects that hold them outlive this load.
        self._left: set[int] = {id(handle) for handle in _stored_values(keep, parents=True)} if self._kinds else set()
        self._pending: dict[int, _StoredValue] = {}

    def add(self, objects: Iterable[object]) -> None:
        """Collect the plan's unloaded handles on objects the read built."""
        if not self._kinds:
            return
        for handle in _stored_values(objects, parents=False):
            if isinstance(handle, self._kinds) and not handle.is_loaded and id(handle) not in self._left:
                self._pending.setdefault(id(handle), handle)

    def run(self) -> None:
        """Load every collected value in one batch and fill its handle.

        Packed text and time offsets load grouped by owner release and payload file, each axis's
        time offsets once. Arrays find their chunks in the reader's batches of signals, then load
        grouped by owner release and values file, in storage order.
        """
        pending = list(self._pending.values())
        self._pending.clear()
        texts = [handle for handle in pending if isinstance(handle, StoredText | StoredTextList)]
        for text in sorted(texts, key=lambda handle: _packed_order(handle._location)):
            text._fill()
        offsets = [handle for handle in pending if isinstance(handle, StoredOffsets)]
        for axis in sorted(offsets, key=lambda handle: _packed_order(handle._location)):
            # The first read checks the axis's time offsets and keeps a compact copy.
            axis.read()
        # Asking each array for its location runs one chunk-location query per batch of signals.
        arrays = [(handle, handle._location) for handle in pending if isinstance(handle, StoredArray)]
        arrays.sort(key=lambda pair: _chunk_order(pair[1]))
        for array, location in arrays:
            array._fill(array.read(), location)
