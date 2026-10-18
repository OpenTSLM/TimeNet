"""Walk a dataset's objects in the order the writer stores them.

The writer stores records in tuple order, each record's sources depth first with children in tuple
order, and each source's signals in tuple order. It gives new objects final ids in the same order,
so a first write numbers each kind ``1..N`` and id order is storage order.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Annotation


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset


ID_KINDS = ("record", "source", "signal", "axis", "content", "occurrence", "task")
"""The kinds of entity id. Each kind is numbered on its own."""


def iter_record_sources(record: Record) -> Iterator[Source]:
    """Yield a record's sources in storage order: depth first, children in tuple order.

    Yields:
        Each source before its children.

    Raises:
        TimeFValidationError: If a source is attached twice or the hierarchy contains a cycle.
    """
    seen: set[int] = set()
    stack = list(reversed(record.sources))
    while stack:
        source = stack.pop()
        if id(source) in seen:
            raise TimeFValidationError(f"source {source.id!r} is attached more than once or forms a cycle")
        seen.add(id(source))
        yield source
        stack.extend(reversed(source.sources))


def iter_record_signals(record: Record) -> Iterator[Signal]:
    """Yield a record's signals in storage order: by source in storage order, then in tuple order.

    Yields:
        Each signal of the record.
    """
    for source in iter_record_sources(record):
        yield from source.signals


def iter_record_occurrences(record: Record) -> Iterator[Annotation]:
    """Yield the occurrences on a record, then on each source and its signals, in storage order.

    Yields:
        Each occurrence in the record hierarchy.
    """
    yield from record.annotations
    for source in iter_record_sources(record):
        yield from source.annotations
        for signal in source.signals:
            yield from signal.annotations


def iter_hierarchy_occurrences(
    dataset: "TimeFDataset",
) -> Iterator[tuple[str, Record | Source | Signal | None, Annotation]]:
    """Yield every non-task occurrence in storage order, grouped by owner type.

    The groups come in the order Dataset, Record, Signal, Source, which sorts the same way as the
    stored ``object_type``. Inside a group, owners come in storage order and each owner's
    occurrences in tuple order.

    Yields:
        ``(object_type, owner, occurrence)``. The owner is ``None`` for the Dataset.
    """
    for annotation in dataset.annotations:
        yield "Dataset", None, annotation
    records = dataset.records
    for record in records:
        for annotation in record.annotations:
            yield "Record", record, annotation
    for record in records:
        for signal in iter_record_signals(record):
            for annotation in signal.annotations:
                yield "Signal", signal, annotation
    for record in records:
        for source in iter_record_sources(record):
            for annotation in source.annotations:
                yield "Source", source, annotation
