"""Copy an editable catalog graph while retaining immutable payload handles."""

from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.types import (
    Annotation,
    AnnotationContentRef,
    StepInterval,
    StepPoint,
    Task,
    TaskRef,
    TimeInterval,
    TimePoint,
    TSCorrespondenceTask,
)
from timenet.types.references import DatasetRef, ObjectKind, ObjectRef


# Initialize child identities only after deepcopy has detached mutable metadata.
_set_field = object.__setattr__


@dataclass
class ImportedDataset:
    """Imported objects indexed by their original parent IDs."""

    namespace: str
    source: DatasetRef
    records: dict[str, Record]
    sources: dict[str, Source]
    signals: dict[str, Signal]
    annotations: dict[str, Annotation]
    content_ids: dict[str, str]
    _seed: str = field(repr=False)
    _memo: dict[int, Any] = field(repr=False)

    def identity(self, kind: ObjectKind, identity: str) -> str:
        """Return a deterministic child identity for one parent object."""
        return str(uuid5(NAMESPACE_URL, f"{self._seed}:{kind}:{identity}"))

    def task_ref(self, identity: str) -> TaskRef:
        """Return the imported identity of a parent task without hydrating it."""
        return TaskRef(id=self.identity("task", identity))

    def _origin(self, kind: ObjectKind, identity: str) -> ObjectRef:
        return ObjectRef(dataset=self.source, kind=kind, id=identity)

    def _span(self, span: Any) -> Any:
        if isinstance(span, StepPoint | StepInterval):
            return replace(span, time_series_id=self.signals[span.time_series_id].id)
        if isinstance(span, TimePoint | TimeInterval) and span.time_series_ids is not None:
            return replace(span, time_series_ids=tuple(self.signals[identity].id for identity in span.time_series_ids))
        return span

    def _annotation(self, annotation: Annotation, *, index: bool = True) -> None:
        content_id, occurrence_id = annotation.content_id, annotation.occurrence_id
        _set_field(annotation, "id", self.identity("annotation_content", content_id))
        _set_field(annotation, "content_origin", self._origin("annotation_content", content_id))
        _set_field(annotation, "span", self._span(annotation.span))
        if index:
            self.content_ids[content_id] = annotation.content_id
        if occurrence_id is not None:
            _set_field(annotation, "occurrence_id", self.identity("annotation_occurrence", occurrence_id))
            _set_field(annotation, "occurrence_origin", self._origin("annotation_occurrence", occurrence_id))
            if index:
                self.annotations[occurrence_id] = annotation

    def _task(self, task: Task) -> Task:
        copied = deepcopy(task, dict(self._memo))
        copied.id = self.identity("task", task.id)
        copied.origin = self._origin("task", task.id)
        copied.inputs = tuple(self.records[record.id] for record in task.inputs)
        for field_name in ("input_annotations", "target_annotations"):
            setattr(
                copied,
                field_name,
                tuple(self.annotations[annotation.occurrence_id] for annotation in getattr(task, field_name)),
            )
        if isinstance(task, TSCorrespondenceTask):
            copied.candidate_records = tuple(self.records[record.id] for record in task.candidate_records)
        copied.from_tasks = tuple(self.task_ref(reference.id) for reference in task.from_tasks)
        copied.scope = self._span(task.scope)
        if task.targets is not None:
            copied.targets = tuple(
                self.records[target.id]
                if isinstance(target, Record)
                else self.signals[target.id]
                if isinstance(target, Signal)
                else self._span(target)
                for target in task.targets
            )
        schema = getattr(copied, "target_schema", None)
        if isinstance(schema, AnnotationContentRef):
            copied.target_schema = AnnotationContentRef(id=self.identity("annotation_content", schema.id))
        for annotation in copied.annotations:
            self._annotation(annotation, index=False)
        return copied

    def _hierarchy(
        self,
        originals: tuple[Record, ...],
        copied_records: tuple[Record, ...],
        copied_annotations: tuple[Annotation, ...],
        *,
        include_tasks: bool,
    ) -> None:
        annotations = list(copied_annotations)
        for original, record in zip(originals, copied_records, strict=True):
            self.records[original.id] = record
            for old_source, new_source in zip(original.walk_sources(), record.walk_sources(), strict=True):
                self.sources[old_source.id] = new_source
            for old_signal, signal in zip(original.signals, record.signals, strict=True):
                self.signals[old_signal.id] = signal
            annotations.extend(record.walk_annotations())
        seen_axes = set()
        for original, record in zip(originals, copied_records, strict=True):
            record.record_id = self.identity("record", original.id)
            record.origin = self._origin("record", original.id)
            record.task_ids = (
                tuple(self.task_ref(identity).id for identity in original.task_ids) if include_tasks else ()
            )
            for old_source in original.walk_sources():
                new_source = self.sources[old_source.id]
                new_source.id = self.identity("source", old_source.id)
                new_source.origin = self._origin("source", old_source.id)
            for old_signal in original.signals:
                signal = self.signals[old_signal.id]
                _set_field(signal, "id", self.identity("signal", old_signal.id))
                _set_field(signal, "origin", self._origin("signal", old_signal.id))
                axis = signal.time_axis
                if id(axis) not in seen_axes:
                    old_id = axis.axis_id
                    _set_field(axis, "axis_id", self.identity("axis", old_id))
                    _set_field(axis, "origin", self._origin("axis", old_id))
                    seen_axes.add(id(axis))
        seen_annotations = set()
        for annotation in annotations:
            if id(annotation) not in seen_annotations:
                self._annotation(annotation)
                seen_annotations.add(id(annotation))


def copy_from(
    child: TimeFDataset,
    parent: TimeFDataset | TimeFReader,
    *,
    namespace: str,
    record_ids: Iterable[str] | None,
    include_tasks: bool,
) -> ImportedDataset:
    """Import a metadata graph and optionally append its replayable tasks.

    Returns:
        The imported hierarchy and identity map.

    Raises:
        TimeFValidationError: If the namespace is reused, records are missing, or tasks accompany a subset.
    """
    if not namespace or namespace in child._import_namespaces:
        raise TimeFValidationError(f"import namespace {namespace!r} must be non-empty and unique in the child")
    if include_tasks and record_ids is not None:
        raise TimeFValidationError(
            "copying tasks requires the whole parent catalog; select records with include_tasks=False"
        )
    selected = None if record_ids is None else tuple(record_ids)
    if isinstance(parent, TimeFReader):
        originals = tuple(parent.iter_records(selected))
    else:
        known = {record.id: record for record in parent.records}
        missing = set(selected or ()) - known.keys()
        if missing:
            raise TimeFValidationError(f"unknown parent records {sorted(missing)}")
        originals = parent.records if selected is None else tuple(known[identity] for identity in selected)
    source = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    memo: dict[int, Any] = {}
    copied_records, copied_annotations = deepcopy((originals, parent.annotations), memo)
    imported = ImportedDataset(
        namespace, source, {}, {}, {}, {}, {}, f"timenet:{child.metadata.dataset_id}:{namespace}", memo
    )
    imported._hierarchy(originals, copied_records, copied_annotations, include_tasks=include_tasks)
    # Check all graph identities before adding any part of the import.
    for kind, known, identities in (
        ("record", child._records_by_id, [record.id for record in copied_records]),
        ("source", child._sources_by_id, [source.id for source in imported.sources.values()]),
        ("signal", child._signals_by_id, [signal.id for signal in imported.signals.values()]),
    ):
        if len(identities) != len(set(identities)) or known.keys() & set(identities):
            raise TimeFValidationError(f"imported {kind} identities collide with the child catalog")
    for record in copied_records:
        child.add_record(record=record)
    child._annotations.extend(copied_annotations)
    child._import_namespaces.add(namespace)
    child._schema = None
    if include_tasks:
        schema = parent.schema or parent.derive_schema() if isinstance(parent, TimeFDataset) else parent.schema

        def tasks() -> Iterator[Task]:
            source_tasks = (
                parent.iter_tasks(records=originals) if isinstance(parent, TimeFReader) else parent.iter_tasks()
            )
            for task in source_tasks:
                yield imported._task(task)

        child.add_task_stream(schema.tasks, tasks)
    return imported
