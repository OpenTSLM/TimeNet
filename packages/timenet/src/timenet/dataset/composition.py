"""Build-time selection and copying of parent objects into independent child catalogs."""

from collections.abc import Iterable
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.reader.handles import DatasetHandle
from timenet.types import Access, Annotation, StepSpan, Task, TimeSpan
from timenet.types.object_kind import ObjectKind
from timenet.types.references import DatasetRef, InputLock


_set_field = object.__setattr__


def _occurrence(annotation: Annotation) -> str:
    """Return a bound occurrence ID.

    Raises:
        TimeFValidationError: If an annotation is not attached to an owner.
    """
    if annotation.occurrence_id is None:
        raise TimeFValidationError(f"annotation {annotation.id!r} has no occurrence ID")
    return annotation.occurrence_id


@dataclass
class ImportedDataset:
    """Copied objects indexed by their IDs in one exact parent snapshot."""

    namespace: str
    source: DatasetRef
    records: dict[str, Record] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)
    signals: dict[str, Signal] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    included: dict[ObjectKind, tuple[str, ...]] = field(default_factory=dict)
    _axes: dict[str, Any] = field(default_factory=dict, repr=False)

    def identity(self, kind: ObjectKind, original: str) -> str:
        """Return a compact deterministic ID without nested namespace prefixes."""
        return str(uuid5(NAMESPACE_URL, f"timenet:{self.source}:{kind}:{original}"))

    def rows(self) -> Iterable[tuple[ObjectKind, str, str]]:
        """Yield namespace lookup rows without allocating provenance objects on reads.

        Yields:
            Object kind, parent ID, and child ID.
        """
        for kind, objects in (
            (ObjectKind.RECORD, self.records),
            (ObjectKind.SOURCE, self.sources),
            (ObjectKind.SIGNAL, self.signals),
            (ObjectKind.TASK, self.tasks),
            (ObjectKind.ANNOTATION, self.annotations),
        ):
            for original, obj in objects.items():
                yield kind, original, obj.occurrence_id if isinstance(obj, Annotation) else obj.id


class _Import:
    """Resolve a selected parent graph, then copy and remap it at build time."""

    def __init__(self, child: TimeFDataset, parent: DatasetHandle, result: ImportedDataset) -> None:
        self.child = child
        self.parent = parent.reader._read_catalog()
        self.result = result
        self.objects: dict[ObjectKind, dict[str, Any]] = {
            ObjectKind.RECORD: {r.id: r for r in self.parent.records},
            ObjectKind.TASK: {t.id: t for t in self.parent.tasks},
            ObjectKind.SOURCE: {},
            ObjectKind.SIGNAL: {},
            ObjectKind.ANNOTATION: {},
        }
        self.owners: dict[tuple[ObjectKind, str], tuple[ObjectKind, str]] = {}
        self.selected: dict[ObjectKind, set[str]] = {kind: set() for kind in self.objects}
        for record in self.parent.records:
            self._annotations(ObjectKind.RECORD, record)
            for source in record.walk_sources():
                self.objects[ObjectKind.SOURCE][source.id] = source
                self.owners[ObjectKind.SOURCE, source.id] = (ObjectKind.RECORD, record.id)
                self._annotations(ObjectKind.SOURCE, source)
                for signal in source.signals:
                    self.objects[ObjectKind.SIGNAL][signal.id] = signal
                    self.owners[ObjectKind.SIGNAL, signal.id] = (ObjectKind.SOURCE, source.id)
                    self._annotations(ObjectKind.SIGNAL, signal)
        for task in self.parent.tasks:
            self._annotations(ObjectKind.TASK, task)
        for annotation in self.parent.annotations:
            self.objects[ObjectKind.ANNOTATION][_occurrence(annotation)] = annotation
            self.owners[ObjectKind.ANNOTATION, _occurrence(annotation)] = (
                ObjectKind.DATASET,
                self.parent.metadata.dataset_id,
            )

    def _annotations(self, kind: ObjectKind, owner: Any) -> None:
        for annotation in owner.annotations:
            self.objects[ObjectKind.ANNOTATION][_occurrence(annotation)] = annotation
            self.owners[ObjectKind.ANNOTATION, _occurrence(annotation)] = (kind, owner.id)

    def select(self, kind: ObjectKind, identity: str, active: frozenset[str] = frozenset()) -> None:  # noqa: PLR0912 - dependency kinds
        """Include an object and all dependencies needed for a valid child graph.

        Raises:
            TimeFValidationError: If the selection is absent or task dependencies cycle.
        """
        if kind == ObjectKind.DATASET:
            return
        if kind == ObjectKind.TASK and identity in active:
            raise TimeFValidationError(f"task dependency cycle at {identity!r}")
        obj = self.objects[kind].get(identity)
        if obj is None:
            raise TimeFValidationError(f"unknown parent {kind} {identity!r}")
        if identity in self.selected[kind]:
            return
        self.selected[kind].add(identity)
        if kind == ObjectKind.RECORD:
            for source in obj.sources:
                self.select(ObjectKind.SOURCE, source.id)
        elif kind == ObjectKind.SOURCE:
            for source in obj.sources:
                self.select(ObjectKind.SOURCE, source.id)
            for signal in obj.signals:
                self.select(ObjectKind.SIGNAL, signal.id)
        elif kind == ObjectKind.TASK:
            for record in TimeFDataset._task_records(obj):
                self.select(ObjectKind.RECORD, record.id)
            for signal in TimeFDataset._task_signals(obj):
                self.select(ObjectKind.SIGNAL, signal.id)
            for annotation in (*obj.input_annotations, *obj.target_annotations):
                self.select(ObjectKind.ANNOTATION, _occurrence(annotation))
            for task in obj.from_tasks:
                self.select(ObjectKind.TASK, task.id, active | {identity})
        elif kind == ObjectKind.ANNOTATION:
            owner_kind, owner_id = self.owners[kind, identity]
            self.select(owner_kind, owner_id, active)
        # Attached annotations are copied with full objects. They need no reverse traversal.
        if kind != ObjectKind.ANNOTATION:
            self.selected[ObjectKind.ANNOTATION].update(ann.occurrence_id for ann in obj.annotations)

    def _annotation(self, original: Annotation) -> Annotation:
        identity = _occurrence(original)
        if identity not in self.result.annotations:
            self.result.annotations[identity] = replace(
                original,
                id=self.result.identity(ObjectKind.ANNOTATION_CONTENT, original.id),
                occurrence_id=self.result.identity(ObjectKind.ANNOTATION, identity),
                value=deepcopy(original.value),
                metadata=deepcopy(original.metadata),
                occurrence_metadata=deepcopy(original.occurrence_metadata),
            )
        return self.result.annotations[identity]

    def _attach(self, original: Any, target: Any) -> None:
        additions = [
            ann
            for ann in original.annotations
            if ann.occurrence_id in self.selected[ObjectKind.ANNOTATION]
            and ann.occurrence_id not in self.result.annotations
        ]
        # Existing copied occurrences are deliberately not reattached after child edits.
        target.annotations = (*target.annotations, *(self._annotation(ann) for ann in additions))

    def _source(self, original: Source) -> Source | None:
        descendants = tuple(child for source in original.sources if (child := self._source(source)) is not None)
        signals = tuple(signal for signal in original.signals if signal.id in self.selected[ObjectKind.SIGNAL])
        if not descendants and not signals and original.id not in self.selected[ObjectKind.SOURCE]:
            return None
        target = self.result.sources.get(original.id)
        if target is None:
            target = Source(
                id=self.result.identity(ObjectKind.SOURCE, original.id),
                name=original.name,
                metadata=deepcopy(original.metadata),
            )
            self.result.sources[original.id] = target
        for signal in signals:
            if signal.id not in self.result.signals:
                axis = self.result._axes.get(signal.time_axis.axis_id)
                if axis is None:
                    axis = replace(
                        signal.time_axis, axis_id=self.result.identity(ObjectKind.AXIS, signal.time_axis.axis_id)
                    )
                    self.result._axes[signal.time_axis.axis_id] = axis
                cloned = copy(signal)
                _set_field(cloned, "id", self.result.identity(ObjectKind.SIGNAL, signal.id))
                _set_field(cloned, "time_axis", axis)
                _set_field(cloned, "metadata", deepcopy(signal.metadata))
                _set_field(cloned, "annotations", tuple(self._annotation(ann) for ann in signal.annotations))
                self.result.signals[signal.id] = cloned
                target.signals = (*target.signals, cloned)
        known = {source.id for source in target.sources}
        target.sources = (*target.sources, *(source for source in descendants if source.id not in known))
        self._attach(original, target)
        return target

    def _value(self, value: Any) -> Any:  # noqa: PLR0911 - distinct reference kinds
        if isinstance(value, Record):
            return self.result.records[value.id]
        if isinstance(value, Signal):
            return self.result.signals[value.id]
        if isinstance(value, Annotation):
            return self._annotation(value)
        if isinstance(value, StepSpan):
            return replace(value, time_series_id=self.result.signals[value.time_series_id].id)
        if isinstance(value, TimeSpan) and value.time_series_ids is not None:
            return replace(value, time_series_ids=tuple(self.result.signals[key].id for key in value.time_series_ids))
        if isinstance(value, tuple):
            return tuple(self._value(item) for item in value)
        return deepcopy(value)

    def apply(self) -> ImportedDataset:  # noqa: PLR0912 - graph construction
        """Copy the selected hierarchy and remap task references.

        Returns:
            The persistent import mapping, including automatically selected objects.
        """
        for original in self.parent.records:
            sources = tuple(child for source in original.sources if (child := self._source(source)) is not None)
            if not sources and original.id not in self.selected[ObjectKind.RECORD]:
                continue
            target = self.result.records.get(original.id)
            new = target is None
            if target is None:
                target = Record(
                    record_id=self.result.identity(ObjectKind.RECORD, original.id),
                    start_time=deepcopy(original.start_time),
                    time_span=original.time_span,
                    metadata=deepcopy(original.metadata),
                )
                self.result.records[original.id] = target
            known = {source.id for source in target.sources}
            target.sources = (*target.sources, *(source for source in sources if source.id not in known))
            self._attach(original, target)
            if new:
                self.child.add_record(record=target)
            else:
                source_map, signal_map = self.child._index_record_hierarchy(target)
                self.child._sources_by_id.update(source_map)
                self.child._signals_by_id.update(signal_map)
                self.child._signals_by_record_id[target.id] = target.signals
        for annotation in self.parent.annotations:
            if (
                _occurrence(annotation) in self.selected[ObjectKind.ANNOTATION]
                and _occurrence(annotation) not in self.result.annotations
            ):
                self.child._annotations.append(self._annotation(annotation))
        pending = []
        for identity in sorted(self.selected[ObjectKind.TASK]):
            if identity in self.result.tasks:
                continue
            original = self.objects[ObjectKind.TASK][identity]
            task = copy(original)
            task.id = self.result.identity(ObjectKind.TASK, identity)
            for name in (
                "inputs",
                "targets",
                "scope",
                "prompt",
                "rationale",
                "input_annotations",
                "target_annotations",
                "annotations",
                "metadata",
            ):
                setattr(task, name, self._value(getattr(original, name)))
            if hasattr(task, "candidate_records"):
                task.candidate_records = self._value(original.candidate_records)
            self.result.tasks[identity] = task
            pending.append((original, task))
        for original, task in pending:
            task.from_tasks = tuple(self.result.tasks[parent.id] for parent in original.from_tasks)
        if pending:
            self.child.add_tasks(tasks=[task for _, task in pending])
        self.child._schema = None
        self.result.included = {
            kind: tuple(sorted(objects))
            for kind, objects in (
                (ObjectKind.RECORD, self.result.records),
                (ObjectKind.SOURCE, self.result.sources),
                (ObjectKind.SIGNAL, self.result.signals),
                (ObjectKind.TASK, self.result.tasks),
                (ObjectKind.ANNOTATION, self.result.annotations),
            )
        }
        return self.result


def import_from(  # noqa: PLR0913
    child: TimeFDataset,
    parent: DatasetHandle,
    *,
    namespace: str | None = None,
    records: Iterable[str] = (),
    sources: Iterable[str] = (),
    signals: Iterable[str] = (),
    tasks: Iterable[str] = (),
    annotations: Iterable[str] = (),
) -> ImportedDataset:
    """Import selected objects and required dependencies without decoding stored payloads.

    Returns:
        Copied objects indexed by their parent IDs.

    Raises:
        TimeFValidationError: If a namespace is reused for a different parent snapshot.
    """
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    namespace = str(ref) if namespace is None else namespace
    if not namespace.strip():
        raise TimeFValidationError("an import namespace must not be blank")
    for alias, imported in child._imports.items():
        if imported.source == ref and alias != namespace:
            raise TimeFValidationError(f"parent {ref} already uses namespace {alias!r}")
    lock = InputLock(source=ref, metadata=parent.metadata.model_dump(mode="json"))
    ancestors = (*parent.reader._version.manifest.lineage, lock)
    previous = child.input_locks.get(namespace)
    if previous is not None and previous.source != ref:
        raise TimeFValidationError(f"namespace {namespace!r} already names a different parent snapshot")
    result = child._imports.get(namespace)
    if result is None:
        result = ImportedDataset(namespace, ref)
    operation = _Import(child, parent, result)
    for kind, identities in (
        (ObjectKind.RECORD, records),
        (ObjectKind.SOURCE, sources),
        (ObjectKind.SIGNAL, signals),
        (ObjectKind.TASK, tasks),
        (ObjectKind.ANNOTATION, annotations),
    ):
        for identity in identities:
            operation.select(kind, identity)
    result = operation.apply()
    child._imports[namespace] = result
    child.input_locks[namespace] = lock
    for ancestor in ancestors:
        child.lineage[str(ancestor.source)] = ancestor
    ranks = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}
    if ranks[parent.metadata.access] > ranks[child.metadata.access]:
        child._metadata = child.metadata.model_copy(
            update={"access": parent.metadata.access, "access_url": parent.metadata.access_url}
        )
    return result
