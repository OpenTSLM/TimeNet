"""Project typed source lineage independently of payload ownership."""

from collections.abc import Iterable, Iterator
from typing import Any

from timenet.dataset.record import Record
from timenet.types import Annotation, Task


def origin_slots(
    records: Iterable[Record] = (), *, annotations: Iterable[Annotation] = (), tasks: Iterable[Task] = ()
) -> Iterator[tuple[str, str, Any, str]]:
    """Yield catalog identity, object, and lineage field without reading content.

    Yields:
        A slot for each object or annotation identity.
    """
    all_annotations = list(annotations)
    for record in records:
        yield "Record", record.id, record, "origin"
        all_annotations.extend(record.annotations)
        for source in record.walk_sources():
            yield "Source", source.id, source, "origin"
            all_annotations.extend(source.annotations)
            for signal in source.signals:
                yield "Signal", signal.id, signal, "origin"
                yield "Axis", signal.time_axis.axis_id, signal.time_axis, "origin"
                all_annotations.extend(signal.annotations)
    for task in tasks:
        yield "Task", task.id, task, "origin"
        all_annotations.extend(task.annotations)
    for annotation in all_annotations:
        yield "AnnotationContent", annotation.content_id, annotation, "content_origin"
        if annotation.occurrence_id is not None:
            yield "AnnotationOccurrence", annotation.occurrence_id, annotation, "occurrence_origin"
