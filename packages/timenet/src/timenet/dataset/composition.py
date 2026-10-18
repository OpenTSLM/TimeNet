"""Build-time selection and copying of parent objects into independent child catalogs."""

from collections.abc import Iterable
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from timenet.dataset.axis import TimeAxis
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Access, Annotation
from timenet.types.object_kind import ObjectKind
from timenet.types.references import DatasetRef, InputLock


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.reader.handles import DatasetHandle


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
    _axes: dict[str, TimeAxis] = field(default_factory=dict, repr=False)

    @property
    def included(self) -> dict[ObjectKind, tuple[str, ...]]:
        """Return sorted imported IDs, including copies retained from earlier imports."""
        return {
            kind: tuple(sorted(objects))
            for kind, objects in (
                (ObjectKind.RECORD, self.records),
                (ObjectKind.SOURCE, self.sources),
                (ObjectKind.SIGNAL, self.signals),
                (ObjectKind.ANNOTATION, self.annotations),
            )
        }

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
            (ObjectKind.ANNOTATION, self.annotations),
        ):
            for original, obj in objects.items():
                yield kind, original, obj.occurrence_id if isinstance(obj, Annotation) else obj.id


class _ParentImporter:
    """Resolve a selected parent graph, then copy and remap it at build time."""

    def __init__(self, child: "TimeFDataset", parent: "DatasetHandle", result: ImportedDataset) -> None:
        self.child = child
        self.parent = parent.reader._read_catalog()
        self.result = result
        self.objects: dict[ObjectKind, dict[str, Any]] = {
            ObjectKind.RECORD: {r.id: r for r in self.parent.records},
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

    def include_requested_objects(
        self,
        *,
        records: Iterable[str],
        sources: Iterable[str],
        signals: Iterable[str],
        annotations: Iterable[str],
    ) -> None:
        """Expand the requested IDs to include each object's required dependencies."""
        for kind, identities in (
            (ObjectKind.RECORD, records),
            (ObjectKind.SOURCE, sources),
            (ObjectKind.SIGNAL, signals),
            (ObjectKind.ANNOTATION, annotations),
        ):
            for identity in identities:
                self._include_with_dependencies(kind, identity)

    def _include_with_dependencies(self, kind: ObjectKind, identity: str) -> None:
        """Include an object and all dependencies needed for a valid child graph.

        Args:
            kind: The kind of object to include.
            identity: The object's ID in the parent catalog.

        Raises:
            TimeFValidationError: If the selection is absent.
        """
        if kind == ObjectKind.DATASET:
            return
        obj = self.objects[kind].get(identity)
        if obj is None:
            raise TimeFValidationError(f"unknown parent {kind} {identity!r}")
        if identity in self.selected[kind]:
            return
        self.selected[kind].add(identity)
        if kind == ObjectKind.RECORD:
            self._include_record(obj)
        elif kind == ObjectKind.SOURCE:
            self._include_source(obj)
        elif kind == ObjectKind.ANNOTATION:
            self._include_annotation(identity)
        # Attached annotations are copied with full objects. They need no reverse traversal.
        if kind != ObjectKind.ANNOTATION:
            self.selected[ObjectKind.ANNOTATION].update(ann.occurrence_id for ann in obj.annotations)

    def _include_record(self, record: Record) -> None:
        """Include every Source and Signal in a selected Record."""
        for source in record.sources:
            self._include_with_dependencies(ObjectKind.SOURCE, source.id)

    def _include_source(self, source: Source) -> None:
        """Include a selected Source's complete subtree."""
        for child in source.sources:
            self._include_with_dependencies(ObjectKind.SOURCE, child.id)
        for signal in source.signals:
            self._include_with_dependencies(ObjectKind.SIGNAL, signal.id)

    def _include_annotation(self, identity: str) -> None:
        """Include the owner needed to attach an explicitly selected annotation."""
        owner_kind, owner_id = self.owners[ObjectKind.ANNOTATION, identity]
        self._include_with_dependencies(owner_kind, owner_id)

    def _copy_annotation(self, original: Annotation) -> Annotation:
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

    def _attach_missing_annotations(self, original: Any, target: Any) -> None:
        additions = [
            ann
            for ann in original.annotations
            if ann.occurrence_id in self.selected[ObjectKind.ANNOTATION]
            and ann.occurrence_id not in self.result.annotations
        ]
        # Existing copied occurrences are deliberately not reattached after child edits.
        target.annotations = (*target.annotations, *(self._copy_annotation(ann) for ann in additions))

    def _copy_source_trees(self, sources: Iterable[Source]) -> tuple[Source, ...]:
        """Copy the selected branches below a Record or Source.

        Returns:
            Copied branches, excluding those with no selected objects.
        """
        copied = []
        for source in sources:
            target = self._copy_source_tree(source)
            if target is not None:
                copied.append(target)
        return tuple(copied)

    def _copy_source_tree(self, original: Source) -> Source | None:
        """Extend a copied Source with selected descendants and Signals.

        Returns:
            The copied Source, or ``None`` when this branch has no selected objects.
        """
        descendants = self._copy_source_trees(original.sources)
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
        additions = tuple(self._copy_signal(signal) for signal in signals if signal.id not in self.result.signals)
        target.signals = (*target.signals, *additions)
        known = {source.id for source in target.sources}
        target.sources = (*target.sources, *(source for source in descendants if source.id not in known))
        self._attach_missing_annotations(original, target)
        return target

    def _copy_signal(self, original: Signal) -> Signal:
        """Copy a newly selected Signal while retaining its stored payload loaders.

        Returns:
            The registered copy with remapped identity, axis, and annotations.
        """
        cloned = copy(original)
        _set_field(cloned, "id", self.result.identity(ObjectKind.SIGNAL, original.id))
        _set_field(cloned, "time_axis", self._copy_axis(original.time_axis))
        _set_field(cloned, "metadata", deepcopy(original.metadata))
        _set_field(cloned, "annotations", tuple(self._copy_annotation(ann) for ann in original.annotations))
        self.result.signals[original.id] = cloned
        return cloned

    def _copy_axis(self, original: TimeAxis) -> TimeAxis:
        """Reuse one copied axis across Signals that share it.

        Returns:
            The axis with its identity remapped into the child namespace.
        """
        axis = self.result._axes.get(original.axis_id)
        if axis is None:
            axis = replace(original, axis_id=self.result.identity(ObjectKind.AXIS, original.axis_id))
            self.result._axes[original.axis_id] = axis
        return axis

    def copy_selected_objects(self) -> ImportedDataset:
        """Copy the selected data hierarchy.

        Returns:
            The persistent import mapping, including automatically selected objects.
        """
        self._copy_records()
        self._copy_dataset_annotations()
        self.child._schema = None
        return self.result

    def _copy_records(self) -> None:
        """Create or extend Records while preserving the child's existing objects and edits."""
        for original in self.parent.records:
            sources = self._copy_source_trees(original.sources)
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
            self._attach_missing_annotations(original, target)
            if new:
                self.child.add_record(record=target)
            else:
                source_map, signal_map = self.child._index_record_hierarchy(target)
                self.child._sources_by_id.update(source_map)
                self.child._signals_by_id.update(signal_map)
                self.child._signals_by_record_id[target.id] = target.signals

    def _copy_dataset_annotations(self) -> None:
        """Attach newly selected annotations owned directly by the parent dataset."""
        for annotation in self.parent.annotations:
            if (
                _occurrence(annotation) in self.selected[ObjectKind.ANNOTATION]
                and _occurrence(annotation) not in self.result.annotations
            ):
                self.child._annotations.append(self._copy_annotation(annotation))


def _prepare_import(child: "TimeFDataset", parent: "DatasetHandle", namespace: str | None) -> ImportedDataset:
    """Validate the parent namespace and reuse its persistent import mapping.

    Returns:
        The existing mapping or a new mapping that has not yet been registered on the child.

    Raises:
        TimeFValidationError: If the namespace is blank or conflicts with a previous import.
    """
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    namespace = str(ref) if namespace is None else namespace
    if not namespace.strip():
        raise TimeFValidationError("an import namespace must not be blank")
    for alias, imported in child._imports.items():
        if imported.source == ref and alias != namespace:
            raise TimeFValidationError(f"parent {ref} already uses namespace {alias!r}")
    previous = child.input_locks.get(namespace)
    if previous is not None and previous.source != ref:
        raise TimeFValidationError(f"namespace {namespace!r} already names a different parent snapshot")
    return child._imports.get(namespace) or ImportedDataset(namespace, ref)


def _record_parent_dependency(child: "TimeFDataset", parent: "DatasetHandle", imported: ImportedDataset) -> None:
    """Register the import mapping, parent metadata, and lineage after copying succeeds."""
    lock = InputLock(source=imported.source, metadata=parent.metadata.model_dump(mode="json"))
    child._imports[imported.namespace] = imported
    child.input_locks[imported.namespace] = lock
    for ancestor in (*parent.reader._version.manifest.lineage, lock):
        child.lineage[str(ancestor.source)] = ancestor


def _inherit_access_restrictions(child: "TimeFDataset", parent: "DatasetHandle") -> None:
    """Apply the parent's access requirement when it is stricter than the child's."""
    ranks = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}
    if ranks[parent.metadata.access] > ranks[child.metadata.access]:
        child._metadata = child.metadata.model_copy(
            update={"access": parent.metadata.access, "access_url": parent.metadata.access_url}
        )


def import_from(  # noqa: PLR0913
    child: "TimeFDataset",
    parent: "DatasetHandle",
    *,
    namespace: str | None = None,
    records: Iterable[str] = (),
    sources: Iterable[str] = (),
    signals: Iterable[str] = (),
    annotations: Iterable[str] = (),
) -> ImportedDataset:
    """Import selected objects and required dependencies without decoding stored payloads.

    Returns:
        Copied objects indexed by their parent IDs.

    Raises:
        TimeFValidationError: If the namespace or selection is invalid.
    """  # noqa: DOC502 - namespace and dependency helpers raise TimeFValidationError
    imported = _prepare_import(child, parent, namespace)
    importer = _ParentImporter(child, parent, imported)
    importer.include_requested_objects(records=records, sources=sources, signals=signals, annotations=annotations)
    importer.copy_selected_objects()
    _record_parent_dependency(child, parent, imported)
    _inherit_access_restrictions(child, parent)
    return imported
