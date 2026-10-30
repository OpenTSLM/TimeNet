"""The step that loads the stored values that a load plan names.

A read builds its objects with every stored value behind an unloaded value handle. Before it hands
the objects out, it collects the handles its plan reaches, reads their values in one batch, and
fills the handles. Objects that existed before the read are never changed. An import runs the same
step over its copies.

:class:`LoadPlan` and :class:`Loading` live in :mod:`timenet.types.load_plan`. This module exports
them too.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import singledispatch

from timenet.dataset import Record, Signal, Source
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import PackedLocation, StoredOffsets, StoredText, StoredTextList
from timenet.types import Annotation, Task
from timenet.types.load_plan import Loading, LoadPlan


__all__ = ["LoadPlan", "Loading", "ValueLoad"]


_StoredValue = StoredArray | StoredOffsets | StoredText | StoredTextList
_STORED_VALUES = (StoredArray, StoredOffsets, StoredText, StoredTextList)


@singledispatch
def _load_plan_children(item: object) -> tuple[object, ...]:  # noqa: ARG001 - dispatch picks a function by its type
    """Return the objects that a load plan visits next from ``item``: what it holds or references directly."""
    return ()


@_load_plan_children.register
def _record_children(item: Record) -> tuple[object, ...]:
    """Return a record's annotations and sources."""
    return (*item.annotations, *item.sources)


@_load_plan_children.register
def _source_children(item: Source) -> tuple[object, ...]:
    """Return a source's annotations, signals, and nested sources."""
    return (*item.annotations, *item.signals, *item.sources)


@_load_plan_children.register
def _signal_children(item: Signal) -> tuple[object, ...]:
    """Return a signal's values handle, its time-offsets handle if it has one, and its annotations."""
    time_offsets = () if item._time_offsets is None else (item._time_offsets,)
    return (item._values, *time_offsets, *item.annotations)


@_load_plan_children.register
def _annotation_children(item: Annotation) -> tuple[object, ...]:
    """Return an annotation's value."""
    return (item.value,)


@_load_plan_children.register
def _task_children(item: Task) -> tuple[object, ...]:
    """Return a task's text, annotations, input and candidate records, targets, and the annotations it references."""
    return (
        item.prompt,
        item.rationale,
        *(item.targets or ()),
        *item.annotations,
        *item.inputs,
        *getattr(item, "candidate_records", ()),
        *item.input_annotations,
        *item.target_annotations,
    )


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
            continue
        stack.extend(_load_plan_children(item))
        if parents and isinstance(item, Task):
            stack.extend(item.from_tasks)


def _packed_order(location: PackedLocation) -> tuple[str, str, int]:
    """Return a sort key that groups packed values by source release and payload file."""
    return (location.source or "", location.path, location.row)


def _chunk_order(location: ChunkLocation) -> tuple[str, str, int, int]:
    """Return a sort key that groups arrays by source release and values file, in storage order."""
    first = location.chunks[0]
    return (location.source or "", first.chunk_file, first.data_index.major_idx, first.data_index.minor_idx or 0)


class ValueLoad:
    """Collect the unloaded stored values a plan reaches on a read's new objects, then load them.

    The handles on objects the read did not build are left as they are, so a read changes only the
    objects it hands out. An import runs it over its copies too.
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

    def add(self, objects: Iterable[object], *, parents: bool = False) -> None:
        """Collect the plan's unloaded handles on objects the read built.

        Args:
            objects: The objects the read built.
            parents: Whether to follow each task's ``from_tasks`` as well.
        """
        if not self._kinds:
            return
        for handle in _stored_values(objects, parents=parents):
            if isinstance(handle, self._kinds) and not handle.is_loaded and id(handle) not in self._left:
                self._pending.setdefault(id(handle), handle)

    def run(self) -> None:
        """Load every collected value in one batch and fill its handle.

        Packed text and time offsets share payload files, so they load in one pass, grouped by
        source release and payload file. The reader then decodes each payload file once. Each axis's
        time offsets load once. Arrays find their chunks in the reader's batches of signals, then
        load grouped by source release and values file, in storage order.
        """
        pending = list(self._pending.values())
        self._pending.clear()
        packed = [handle for handle in pending if isinstance(handle, StoredText | StoredTextList | StoredOffsets)]
        for handle in sorted(packed, key=lambda handle: _packed_order(handle._location)):
            if isinstance(handle, StoredOffsets):
                # The first read checks the axis's time offsets and keeps a compact copy.
                handle.read()
            else:
                handle._fill()
        # Asking each array for its location runs one chunk-location query per batch of signals.
        arrays = [(handle, handle._location) for handle in pending if isinstance(handle, StoredArray)]
        arrays.sort(key=lambda pair: _chunk_order(pair[1]))
        for array, location in arrays:
            array._fill(array.read(), location)
