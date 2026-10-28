"""Copy selected parent Records, Signals, and tasks into a child dataset, sharing their stored values."""

from collections import deque
from collections.abc import Container, Hashable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, TypeAlias

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import (
    Access,
    Annotation,
    DatasetMetadata,
    Span,
    StepSpan,
    Task,
    TimeInterval,
    TimePoint,
    TimeSpan,
    TSCorrespondenceTask,
)
from timenet.types.ids import EntityId, check_id
from timenet.types.load_plan import LoadPlan
from timenet.types.object_kind import ObjectKind
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.reader import TimeFReader


ImportMap: TypeAlias = dict[tuple[ParentAlias, ObjectKind, EntityId], EntityId]
"""The child ID of each imported Record, Signal, and task, keyed by parent alias, kind, and parent ID.

It is the one record of what a dataset imported. A copy's ID is the next ID of its kind in the child,
so nothing derives it from the parent's: an import looks a parent object up here and then finds the
copy by its child ID among the dataset's own objects, and a copy the dataset no longer holds is
copied again. The map holds plain IDs and copies with ``dict(...)``. The writer stores the entries
whose copy the dataset holds in the ``imported_objects`` table, and a dataset read back from storage
gets them back.
"""

_STRICTNESS = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the objects that one ``import_from`` call selected, by their parent IDs.

    Each copy is one the call made or one the child already held. The view lists only what the call
    selected: a Record the call copied to hold a selected Signal or for a selected task is in the
    dataset, not in ``records``, and so is a task that a selected task derives from.
    """

    alias: ParentAlias
    """The parent alias the objects were imported under."""
    parent: DatasetRef
    """The exact parent release."""
    records: dict[EntityId, Record] = field(default_factory=dict)
    """The copy of each Record selected with ``records=``, keyed by its parent ID."""
    signals: dict[EntityId, Signal] = field(default_factory=dict)
    """The copy of each Signal selected with ``signals=``, keyed by its parent ID."""
    tasks: dict[EntityId, Task] = field(default_factory=dict)
    """The copy of each task selected with ``tasks=``, keyed by its parent ID."""


def _signal_paths(record: Record) -> Iterator[tuple[tuple[Source, ...], Signal]]:
    """Yield each Signal of a Record with the Sources above it, from its root Source down.

    Yields:
        The Sources from the root to the Signal's own Source, and the Signal.
    """
    pending: list[tuple[tuple[Source, ...], Source]] = [((source,), source) for source in record.sources]
    while pending:
        path, source = pending.pop()
        for signal in source.signals:
            yield path, signal
        pending.extend(((*path, child), child) for child in source.sources)


def _pair_sources(original: Record, copy: Record, held: dict[EntityId, Signal]) -> dict[int, Source]:
    """Match the parent Sources to their copies, through the copies of the Signals below them.

    Args:
        original: The parent Record.
        copy: The child's copy of it.
        held: The child copy of each parent Signal the child holds, by parent ID.

    Returns:
        The copy of each parent Source the child holds, keyed by the ``id()`` of the parent Source.
    """
    copied_paths = {id(signal): path for path, signal in _signal_paths(copy)}
    pairs: dict[int, Source] = {}
    for path, signal in _signal_paths(original):
        held_signal = held.get(signal.id)
        copied_path = None if held_signal is None else copied_paths.get(id(held_signal))
        if copied_path is not None and len(copied_path) == len(path):
            pairs.update(zip(map(id, path), copied_path, strict=True))
    return pairs


def _fits(annotation: Annotation, signals: tuple[Signal, ...], owner: str, time_span: TimeInterval | None) -> bool:
    """Return whether an annotation's span lies inside the windows of the given Signals.

    The rule is the one ``annotate`` applies, without its warning: an annotation without a span
    fits, and a Record's ``time_span``, when it has one, replaces the Signals' windows.

    Args:
        annotation: The annotation to place.
        signals: The Signals its owner holds.
        owner: The owner, named for the check's message.
        time_span: The owning Record's session span, or ``None`` for a Source.
    """
    if annotation.span is None:
        return True
    try:
        check_span_within_window(
            f"annotation {annotation.key!r}", annotation.span, signals, owner, time_span, warn_when_outside=False
        )
    except TimeFValidationError:
        return False
    return True


def _hierarchy(record: Record) -> list[tuple[Record | Source, dict[str, tuple]]]:
    """Return the fields that an extension changes, for each container of a Record.

    Returns:
        Each container with its ``sources``, ``annotations``, and, for a Source, ``signals``.
    """
    return [
        (container, {name: getattr(container, name) for name in names})
        for container, names in (
            (record, ("sources", "annotations")),
            *((source, ("sources", "signals", "annotations")) for source in record.walk_sources()),
        )
    ]


class _Importer:
    """Copy parent Records into one child under one parent alias, extending the copies the child holds."""

    def __init__(self, child: "TimeFDataset", alias: ParentAlias, release: DatasetRef) -> None:
        self.child = child
        self.alias = alias
        self.release = release
        """The exact parent release."""
        self.records: dict[EntityId, Record] = {}
        """The child copy of each parent Record this call found or copied, by parent ID."""
        self.signals: dict[EntityId, Signal] = {}
        """The child copy of each parent Signal this call found or copied, by parent ID."""
        self.sources: dict[int, Source] = {}
        """The child copy of each parent Source this call found or copied, keyed by the ``id()`` of the parent Source."""
        self.annotations: dict[int, Annotation] = {}
        """The copy of each parent annotation this call copied, keyed by the ``id()`` of the parent annotation."""
        self.untold: set[int] = set()
        """The ``id()`` of each parent Source whose copy only its name, which a sibling shares, could identify."""

    def import_record(self, original: Record, wanted: frozenset[EntityId] | None) -> Record:
        """Copy a parent Record, or extend the child's copy of it, with the wanted Signals.

        A new copy gets its ID, and its Signals theirs, when the child registers it; a Signal that
        extends a copy gets its ID when the child indexes the copy again. If extending a copy fails,
        its Sources, Signals, and annotations are restored as they were, and the dataset's indexes
        and ID counters are unchanged.

        Args:
            original: The parent Record, as the parent's reader built it.
            wanted: The parent IDs of the Signals to bring, or ``None`` for the whole Record.

        Returns:
            The child's copy, registered and indexed.
        """
        child_id = self.child._imports.get((self.alias, ObjectKind.RECORD, original.id))
        target = None if child_id is None else self.child._records_by_id.get(child_id)
        if target is None:
            target = Record(
                # Origins are frozen and shared explicitly, like axes, so the copy keeps its parent's:
                # copies of Records that share a clock share it as well.
                start_time=original.start_time,
                time_span=original.time_span,
                metadata=deepcopy(original.metadata),
            )
            copied = _RecordCopy(self, original, target, wanted)
            copied.merge(new=True)
            self.child.add_record(record=target)
        else:
            saved = _hierarchy(target)
            try:
                copied = _RecordCopy(self, original, target, wanted)
                copied.merge(new=False)
                self.child._reindex_record(target)
            except BaseException:
                for container, fields in saved:
                    for name, value in fields.items():
                        setattr(container, name, value)
                raise
        self.records[original.id] = target
        self.signals.update(copied.held)
        self.sources.update(copied.pairs)
        imports = self.child._imports
        imports[self.alias, ObjectKind.RECORD, original.id] = target.id
        for parent_id, signal in copied.held.items():
            imports[self.alias, ObjectKind.SIGNAL, parent_id] = signal.id
        return target

    def copy_signal(self, original: Signal) -> Signal:
        """Copy a parent Signal without an ID, keeping its axis and the handles to its stored values.

        Returns:
            The copy, with copies of its annotations. The child gives it an ID when it indexes it.
        """
        return Signal.from_loader(
            spec=original.spec,
            time_axis=original.time_axis,
            name=original.name,
            loader=original._values,
            n_values=original.n_values,
            time_offsets_loader=original._time_offsets,
            annotations=tuple(self.copy_annotation(annotation) for annotation in original.annotations),
            metadata=deepcopy(original.metadata),
        )

    def copy_annotation(self, original: Annotation) -> Annotation:
        """Copy a parent annotation occurrence as a value, keeping a stored value's handle.

        The writer stores copies of equal content as one content row.

        Returns:
            The copy, which the call remembers for its parent annotation.
        """
        copy = replace(
            original,
            value=deepcopy(original.value),
            metadata=deepcopy(original.metadata),
            occurrence_metadata=deepcopy(original.occurrence_metadata),
        )
        self.annotations[id(original)] = copy
        return copy


class _RecordCopy:
    """Bring the wanted Sources and Signals of one parent Record into the child's copy of it.

    The copy's held Signals come from its own hierarchy, through the import map, so a Signal the
    child removed from the copy counts as not held, whatever the dataset's index says.
    """

    def __init__(
        self, importer: _Importer, original: Record, target: Record, wanted: frozenset[EntityId] | None
    ) -> None:
        self.importer = importer
        self.original = original
        self.target = target
        self.wanted = wanted
        """The parent IDs of the Signals to bring, or ``None`` for the whole Record."""
        imports = importer.child._imports
        keys = {signal.id: (importer.alias, ObjectKind.SIGNAL, signal.id) for signal in original.signals}
        # A new copy holds no Signal yet; a registered one holds Signals with child IDs.
        own = {signal.id: signal for signal in target.signals}
        self.held: dict[EntityId, Signal] = {
            parent_id: own[child_id]
            for parent_id, key in keys.items()
            if (child_id := imports.get(key)) is not None and child_id in own
        }
        """The copy of each parent Signal that the Record's copy holds, by parent ID."""
        self.held_before = frozenset(self.held)
        """The parent IDs of the Signals whose copy the Record's copy held when the call began."""
        self.imported = {id(signal) for signal in self.held.values()}
        """The ``id()`` of each held copy, which tells imported Signals from those the child added."""
        self.pairs = _pair_sources(original, target, self.held)
        """The copy of each parent Source the Record's copy holds, keyed by the ``id()`` of the parent Source.

        It starts with the copies found through the held Signals; the merge adds each copy it
        matches by name or makes.
        """

    def merge(self, *, new: bool) -> None:
        """Bring the wanted Sources and Signals into the Record's copy, then the annotations they place."""
        if self._merge_sources(self.original.sources, self.target) or new:
            self._attach_annotations(self.original, self.target, new=new)

    def _wants(self, signal: Signal) -> bool:
        """Return whether the call brings a parent Signal of which the copy holds no copy yet."""
        return signal.id not in self.held and (self.wanted is None or signal.id in self.wanted)

    def _brings(self, original: Source) -> bool:
        """Return whether a parent Source without a copy has anything for this call to bring.

        A Source that holds no Signal at any depth comes with its whole Record only.
        """
        signals = tuple(original.walk_signals())
        return any(self._wants(signal) for signal in signals) if signals else self.wanted is None

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> bool:
        """Bring the parent Sources below ``target``: extend the copies it holds, and copy the others.

        A copy is found through the copies of the Signals below it. A parent Source that none of
        them finds is matched by name among the copies below ``target`` that hold no imported
        Signal and are not matched yet; Signals the child added to a copy do not count. When
        several parent Sources, or several copies, share the name, the name can't tell them apart:
        the copies stay as they are if the call brings no Signal below the Source, and the call
        raises otherwise.

        Returns:
            Whether ``target`` gained a Signal at any depth.

        Raises:
            TimeFValidationError: If the call brings a Signal below a parent Source whose copy the
                name can't tell apart from a sibling's.
        """
        originals = tuple(originals)
        grew = False
        added: list[Source] = []
        unmatched = [
            source
            for source in target.sources
            if not any(id(signal) in self.imported for signal in source.walk_signals())
        ]
        unpaired = [original.name for original in originals if id(original) not in self.pairs]
        for original in originals:
            copy = self.pairs.get(id(original))
            if copy is None:
                candidates = [source for source in unmatched if source.name == original.name]
                if len(candidates) > 1 or (candidates and unpaired.count(original.name) > 1):
                    if not any(self._wants(signal) for signal in original.walk_signals()):
                        # Nothing comes below it, so its copy can stay unidentified, unless a task
                        # references one of its annotations.
                        self.importer.untold.add(id(original))
                        continue
                    raise TimeFValidationError(_untold(original.name, self.original.id, self.importer.release))
                copy = candidates[0] if candidates else None
                if copy is not None:
                    unmatched.remove(copy)
            if copy is not None:
                self.pairs[id(original)] = copy
                grew = self._merge_source(original, copy, new=False) or grew
            elif self._brings(original):
                copy = Source(name=original.name, metadata=deepcopy(original.metadata))
                self.pairs[id(original)] = copy
                grew = self._merge_source(original, copy, new=True) or grew
                added.append(copy)
        if added:
            target.sources = (*target.sources, *added)
        return grew

    def _merge_source(self, original: Source, target: Source, *, new: bool) -> bool:
        """Bring the wanted descendants of a parent Source into its copy, then the annotations they place.

        Returns:
            Whether ``target`` gained a Signal at any depth.
        """
        grew = self._merge_sources(original.sources, target)
        signals = []
        for signal in original.signals:
            if self._wants(signal):
                self.held[signal.id] = self.importer.copy_signal(signal)
                signals.append(self.held[signal.id])
        if signals:
            target.signals = (*target.signals, *signals)
            grew = True
        if new or grew:
            self._attach_annotations(original, target, new=new)
        return grew

    def _attach_annotations(self, original: Record | Source, target: Record | Source, *, new: bool) -> None:
        """Attach copies of the parent annotations that the call's new Signal copies place.

        Placement is judged on the parent's Signals below ``original``: the copy places an
        annotation when it holds a copy of each of them, or when the annotation's span lies inside
        the windows of those it holds a copy of. A new copy gets every annotation it places. An
        existing copy gets those it places now and did not place with the Signals it held when the
        call began, so its other annotations stay as the child left them, deleted ones included.

        Args:
            original: The parent Record or Source.
            target: Its copy.
            new: Whether the call created the copy.
        """
        signals = tuple(original.walk_signals())
        after = tuple(signal for signal in signals if signal.id in self.held)
        before = () if new else tuple(signal for signal in signals if signal.id in self.held_before)
        if isinstance(original, Record):
            time_span, owner = original.time_span, original._label
        else:
            time_span, owner = None, f"source {original.name!r}"

        def places(annotation: Annotation, held: tuple[Signal, ...]) -> bool:
            return len(held) == len(signals) or _fits(annotation, held, owner, time_span)

        placed = [
            annotation
            for annotation in original.annotations
            if places(annotation, after) and (new or not places(annotation, before))
        ]
        if placed:
            copies = (self.importer.copy_annotation(annotation) for annotation in placed)
            target.annotations = (*target.annotations, *copies)


def _task_fields(task: Task) -> Iterator[tuple[str, Any]]:
    """Yield the name and value of each field a task's constructor takes.

    Reading the dataclass fields, rather than naming them, keeps every field of every task class.

    Yields:
        Each field's name and the task's value for it.
    """
    for task_field in fields(task):
        if task_field.init:
            yield task_field.name, getattr(task, task_field.name)


def _leaves(value: object) -> Iterator[object]:
    """Yield the items of a task field's value, looking inside tuples.

    Yields:
        Each item that is not a tuple.
    """
    if isinstance(value, tuple):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _tasks_to_copy(
    child: "TimeFDataset",
    alias: ParentAlias,
    parent: "TimeFReader",
    task_ids: tuple[EntityId, ...],
    load: LoadPlan,
) -> tuple[dict[EntityId, Task], dict[EntityId, Task], dict[EntityId, Record], tuple[Annotation, ...]]:
    """Sort the selected parent tasks, and those they derive from, into tasks to copy and copies the child holds.

    A selected task the child holds a copy of is not read from the parent. A copy the child holds
    derives from copies it holds too, so the walk stops there. The read of the other tasks loads
    the stored values that ``load`` names into the tasks and the Records it builds.

    Returns:
        The parent tasks to copy and the child's copy of each other task, both by parent ID; every
        parent Record the read of the tasks built, by parent ID; and the parent dataset's own
        occurrences that the tasks reference.
    """

    def held_copy(identity: EntityId) -> Task | None:
        child_id = child._imports.get((alias, ObjectKind.TASK, identity))
        return None if child_id is None else child._tasks_by_id.get(child_id)

    held = {identity: copy for identity in task_ids if (copy := held_copy(identity)) is not None}
    unread = tuple(identity for identity in task_ids if identity not in held)
    read, built, on_dataset = parent._read_tasks_with_records(unread, load=load) if unread else ({}, {}, ())
    pending: dict[EntityId, Task] = {}
    queue = deque(read.values())
    while queue:
        task = queue.popleft()
        if task.id in pending or task.id in held:
            continue
        copy = held_copy(task.id)
        if copy is not None:
            held[task.id] = copy
        else:
            pending[task.id] = task
            queue.extend(task.from_tasks)
    return pending, held, built, on_dataset


def _check_candidates(tasks: Iterable[Task], release: DatasetRef) -> None:
    """Reject a correspondence task whose candidates are the whole parent.

    Raises:
        TimeFValidationError: If a ``TSCorrespondenceTask`` has no ``candidate_records``.
    """
    for task in tasks:
        if isinstance(task, TSCorrespondenceTask) and not task.candidate_records:
            raise TimeFValidationError(
                f"TSCorrespondenceTask {task.id!r} of {release} has no candidate_records, so its candidates are "
                "every Record of the parent; only a task with explicit candidate_records can be imported"
            )


def _task_dependencies(tasks: Iterable[Task]) -> tuple[dict[EntityId, Record], dict[EntityId, Signal]]:
    """Return the Records and Signals that tasks reference, apart from the tasks they derive from.

    Returns:
        The Records, which come whole, and the Signals, which come inside a partial copy of their
        Record, by parent ID.
    """
    records: dict[EntityId, Record] = {}
    signals: dict[EntityId, Signal] = {}
    for task in tasks:
        for name, value in _task_fields(task):
            if name == "from_tasks":
                continue
            for item in _leaves(value):
                if isinstance(item, Record):
                    records.setdefault(item.id, item)
                elif isinstance(item, Signal):
                    signals.setdefault(item.id, item)
    return records, signals


def _annotation_holders(
    tasks: Iterable[Task], built: dict[EntityId, Record], whole: Container[EntityId]
) -> tuple[EntityId, ...]:
    """Return the parent IDs of the Records that hold an annotation that a task references, beside the whole ones.

    A task references annotations on the parent dataset and on the Records that it references. The
    Records that it names come whole. A Record that it references only through a Signal target
    would come as a partial copy, which can lack the owner of the annotation. So such a Record
    comes whole when it holds an annotation that a task references.

    Args:
        tasks: The parent tasks to copy.
        built: The parent Records that the read of the tasks built, by parent ID.
        whole: The parent IDs of the Records that come whole already.

    Returns:
        The parent IDs, each once.
    """
    referenced = {
        id(annotation) for task in tasks for annotation in (*task.input_annotations, *task.target_annotations)
    }
    if not referenced:
        return ()
    return tuple(
        identity
        for identity, record in built.items()
        if identity not in whole and any(id(annotation) in referenced for annotation in record.walk_annotations())
    )


def _bucket(annotation: Annotation) -> tuple[str, TimePoint | TimeInterval | None, str | None]:
    """Return the fields that file an annotation among its owner's: its key, span, and source."""
    return annotation.key, annotation.span, annotation.source


def _same_occurrence(annotation: Annotation, original: Annotation) -> bool:
    """Return whether an annotation filed with a parent one, by bucket and content, holds the same occurrence.

    The filing compares everything but the fields of one attachment, which this compares.
    """
    return (
        annotation.confidence == original.confidence and annotation.occurrence_metadata == original.occurrence_metadata
    )


def _holds_reference(value: object) -> bool:
    """Return whether a value is, or holds at any depth, an object of a dataset or a span.

    Returns:
        Whether copying the value would carry an object that belongs to the parent.
    """
    if isinstance(value, Record | Source | Signal | Annotation | Task | Span):
        return True
    if isinstance(value, dict):
        return any(_holds_reference(item) for item in (*value.keys(), *value.values()))
    if isinstance(value, list | tuple | set | frozenset):
        return any(_holds_reference(item) for item in value)
    return False


_Group: TypeAlias = dict[tuple[Hashable, ...], list[Annotation]]
"""The annotations of one bucket of an owner, by their content key."""


def _take(group: _Group, original: Annotation) -> Annotation | None:
    """Remove and return the first annotation of a group that holds the same occurrence as ``original``.

    Returns:
        The annotation, or ``None`` if none does.
    """
    entries = group.get(original._content_key)
    index = next((index for index, held in enumerate(entries or ()) if _same_occurrence(held, original)), None)
    return None if entries is None or index is None else entries.pop(index)


def _set_annotations(owner: Record | Source | Signal, annotations: tuple[Annotation, ...]) -> None:
    """Replace the annotations of a Record, Source, or Signal."""
    if isinstance(owner, Signal):
        # A Signal is frozen; Signal.annotate changes its annotations the same way.
        object.__setattr__(owner, "annotations", annotations)  # noqa: PLC2801
    else:
        owner.annotations = annotations


class _TaskCopy:
    """Copy parent tasks into one child, pointing each reference at the child's copy of the object.

    A Record or Signal reference points at the copy the call found or made. An annotation reference
    points at the copy on its owner's copy, or on the child for an annotation of the parent dataset.
    """

    def __init__(self, importer: _Importer, originals: Iterable[Record], on_dataset: Iterable[Annotation]) -> None:
        self.importer = importer
        self.owners: dict[int, tuple[Record | Source | Signal, Record]] = {
            id(annotation): (owner, record)
            for record in originals
            for owner in (record, *record.walk_sources(), *record.walk_signals())
            for annotation in owner.annotations
        }
        """The parent Record, Source, or Signal that holds each annotation of the Records the call
        imports, with its Record, keyed by the ``id()`` of the annotation. They are the objects the
        read of the tasks built."""
        self.on_dataset = {id(annotation) for annotation in on_dataset}
        """The ``id()`` of each occurrence of the parent dataset itself that the read of the tasks built."""
        self.made = {id(copy) for copy in importer.annotations.values()}
        """The ``id()`` of each annotation copy this call made, which belongs to its own parent annotation."""
        self.resolved: dict[int, Annotation] = {}
        """The copy, from an earlier call, of each parent annotation a task references, keyed by the
        ``id()`` of the parent annotation."""
        self.filed: dict[int, dict[tuple[Hashable, ...], _Group]] = {}
        """The annotations of each owner copy that no reference took yet, by :func:`_bucket` and then
        by content key, keyed by the owner copy's ``id()``, or by the child's for its own."""
        self.saved: dict[int, tuple[Record | Source | Signal, tuple[Annotation, ...]]] = {}
        """Each owner copy that the call attached an annotation to, with the annotations it held before."""
        self.task: Task | None = None
        """The parent task being copied."""

    def copy_tasks(self, pending: dict[EntityId, Task], held: dict[EntityId, Task]) -> dict[EntityId, Task]:
        """Copy and register the pending tasks, then add each copy to the import map.

        The copies register together, all or none, and each gets the child's next task ID. If they
        fail, the annotations attached for them are removed again and the map is unchanged, so the
        child holds no task of this call.

        Args:
            pending: The parent tasks to copy, by parent ID.
            held: The child's copy of each other task, by parent ID.

        Returns:
            The child's copy of every task, pending or held, by parent ID.
        """
        child = self.importer.child
        dataset_annotations = list(child._annotations)
        try:
            copies = {identity: self._copy(task) for identity, task in pending.items()}
            found = held | copies
            for identity, copy in copies.items():
                copy.from_tasks = tuple(found[task.id] for task in pending[identity].from_tasks)
            if copies:
                child.add_tasks(tasks=copies.values())
        except BaseException:
            child._annotations = dataset_annotations
            for owner, annotations in self.saved.values():
                _set_annotations(owner, annotations)
            raise
        for identity, copy in copies.items():
            child._imports[self.importer.alias, ObjectKind.TASK, identity] = copy.id
        return found

    def _copy(self, original: Task) -> Task:
        """Copy a parent task field by field, with its ``from_tasks`` left for the caller to connect.

        Its own annotations are copied; every other reference points at the child's copy. The copy
        has no ID until the child registers it.

        Returns:
            The copy, of the parent task's class.
        """
        self.task = original
        values: dict[str, Any] = {}
        for name, value in _task_fields(original):
            if name == "from_tasks":
                values[name] = ()
            elif name == "annotations":
                values[name] = tuple(self.importer.copy_annotation(annotation) for annotation in value)
            else:
                values[name] = self._remap(name, value)
        return type(original)(**values)

    def _remap(self, name: str, value: object) -> object:
        """Return a task field's value with each Record, Signal, annotation, and span pointing at the child.

        Returns:
            The value for the copy. A value that references nothing is copied.

        Raises:
            TimeFValidationError: If the value holds a reference in a container other than a tuple,
                whose copy would keep the parent's object.
        """
        if isinstance(value, tuple):
            return tuple(self._remap(name, item) for item in value)
        if isinstance(value, Record):
            return self.importer.records[value.id]
        if isinstance(value, Signal):
            return self.importer.signals[value.id]
        if isinstance(value, Annotation):
            return self._annotation(value)
        if isinstance(value, Span):
            return self._span(value)
        if _holds_reference(value):
            raise TimeFValidationError(
                f"{type(self.task).__name__}.{name} holds a {type(value).__name__} that references objects; "
                "an import remaps references only inside tuples"
            )
        return deepcopy(value)

    def _span(self, span: Span) -> Span:
        """Return a task span with the Signals it names replaced by their copies.

        Returns:
            The span for the copy.
        """
        signals = self.importer.signals
        if isinstance(span, StepSpan):
            return replace(span, time_series=signals[span.time_series.id])
        if isinstance(span, TimeSpan) and span.time_series is not None:
            return replace(span, time_series=tuple(signals[signal.id] for signal in span.time_series))
        return span

    def _annotation(self, original: Annotation) -> Annotation:
        """Return the child's copy of a parent annotation that a task references.

        A copy this call made is the copy. Otherwise the copy is the annotation on the owner's copy
        that holds the same occurrence and that no other reference took, found by bucket and content
        key, so identical occurrences that one call references stay distinct; references in separate
        calls may share one copy. If the owner's copy holds none, because the child removed or
        changed it, a copy of the parent's is attached, beside a changed one. For an annotation of
        the parent dataset, a copy is attached unless one matches.

        Returns:
            The copy.

        Raises:
            TimeFValidationError: If the annotation lies neither on a Record the call imports nor on
                the parent dataset.
        """
        copy = self.importer.annotations.get(id(original))
        if copy is None:
            copy = self.resolved.get(id(original))
        if copy is not None:
            return copy
        owned = self.owners.get(id(original))
        if owned is None and id(original) not in self.on_dataset:
            raise TimeFValidationError(
                f"{type(self.task).__name__} references annotation {original.key!r}, which lies on no Record "
                "this import copies nor on the parent dataset"
            )
        owner = None if owned is None else self._owner_copy(*owned, original)
        copy = _take(self._filed(owner).get(_bucket(original), {}), original)
        if copy is not None:
            self.resolved[id(original)] = copy
            return copy
        return self._attach(original, owner)

    def _attach(self, original: Annotation, owner: Record | Source | Signal | None) -> Annotation:
        """Attach a copy of a parent annotation to its owner's copy, or to the child for the dataset's.

        Returns:
            The copy, an occurrence of its own beside any identical one the owner's copy holds.
        """
        copy = self.importer.copy_annotation(original)
        self.made.add(id(copy))
        if owner is None:
            self.importer.child._annotations.append(copy)
        else:
            self.saved.setdefault(id(owner), (owner, owner.annotations))
            _set_annotations(owner, (*owner.annotations, copy))
        return copy

    def _filed(self, owner: Record | Source | Signal | None) -> dict[tuple[Hashable, ...], _Group]:
        """Return the annotations of an owner's copy, filed at the call's first reference to it.

        A copy this call made is not filed for claims: it belongs to its own parent annotation.

        Args:
            owner: The owner's copy, or ``None`` for the child dataset.

        Returns:
            The annotations no reference took yet, by :func:`_bucket` and then by content key.
        """
        held = self.importer.child._annotations if owner is None else owner.annotations
        key = id(self.importer.child) if owner is None else id(owner)
        filed = self.filed.get(key)
        if filed is None:
            filed = self.filed[key] = {}
            for annotation in held:
                if id(annotation) not in self.made:
                    group = filed.setdefault(_bucket(annotation), {})
                    group.setdefault(annotation._content_key, []).append(annotation)
        return filed

    def _owner_copy(
        self, owner: Record | Source | Signal, record: Record, annotation: Annotation
    ) -> Record | Source | Signal:
        """Return the child's copy of the parent Record, Source, or Signal that holds an annotation.

        Returns:
            The copy.

        Raises:
            TimeFValidationError: If the copy of the annotation's Record holds no copy of its Source,
                or only the Source's name, which a sibling shares, could identify it.
        """
        if isinstance(owner, Record):
            return self.importer.records[owner.id]
        if isinstance(owner, Signal):
            return self.importer.signals[owner.id]
        copy = self.importer.sources.get(id(owner))
        if copy is None and id(owner) in self.importer.untold:
            raise TimeFValidationError(_untold(owner.name, record.id, self.importer.release))
        if copy is None:
            raise TimeFValidationError(
                f"a task references annotation {annotation.key!r} of the Source {owner.name!r} of Record "
                f"{record.id!r}, which the child's copy of that Record holds no copy of"
            )
        return copy


def _untold(name: str, record_id: EntityId, release: DatasetRef) -> str:
    """Return the error for Sources of one name whose copies nothing but that name could tell apart."""
    return (
        f"an import cannot tell apart the Sources named {name!r} of Record {record_id!r} of {release}: "
        "no imported Signal identifies their copies, and the name does not"
    )


def _parent_ids(selection: str, ids: Iterable[EntityId]) -> tuple[EntityId, ...]:
    """Return the parent IDs a selection names, in order, each once.

    Returns:
        The IDs.

    Raises:
        TimeFValidationError: If ``ids`` is a string, which would select one ID per character, or is
            not a collection of positive int64 IDs.
    """
    if isinstance(ids, str | bytes):
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, not the string {ids!r}")
    try:
        selected = tuple(ids)
    except TypeError as exc:
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, got {type(ids).__name__}") from exc
    return tuple(dict.fromkeys(check_id(identity, "parent id") for identity in selected))


def _resolve_alias(child: "TimeFDataset", parent: DatasetRef, alias: ParentAlias | None) -> ParentAlias:
    """Return the parent alias under which the child's card declares ``parent``.

    Returns:
        The alias.

    Raises:
        TimeFValidationError: If the child's ``metadata.parents`` does not declare ``parent``, ``alias``
            does not declare it, the card declares it under several aliases and ``alias``
            is omitted, the child already imports it under another alias, or ``alias`` already
            imports another release.
    """
    dataset_id = child.metadata.dataset_id
    declared = sorted(name for name, ref in child.metadata.parents.items() if ref == parent)
    if not declared:
        raise TimeFValidationError(
            f"{dataset_id} does not declare {parent} as a parent; declare it in metadata.parents to import from it"
        )
    if alias is None:
        if len(declared) > 1:
            raise TimeFValidationError(
                f"{dataset_id} declares {parent} under the parent aliases {declared}; pass alias= to choose one"
            )
        alias = declared[0]
    elif alias not in declared:
        raise TimeFValidationError(
            f"parent alias {alias!r} of {dataset_id} does not declare {parent}, which it declares as {declared}"
        )
    for other, card in child.parent_cards.items():
        if card.source == parent and other != alias:
            raise TimeFValidationError(f"{dataset_id} already imports {parent} under the parent alias {other!r}")
    card = child.parent_cards.get(alias)
    if card is not None and card.source != parent:
        raise TimeFValidationError(
            f"parent alias {alias!r} of {dataset_id} already imports {card.source}, not {parent}"
        )
    return alias


def _inherit_access(child: "TimeFDataset", parent: DatasetMetadata) -> None:
    """Raise the child's access requirement to the parent's when the parent's is stricter.

    The child has one ``access_url``: its own, or that of the first parent that made it stricter. A
    parent that is no stricter keeps its access requirement and page in its parent card.
    """
    if _STRICTNESS[parent.access] > _STRICTNESS[child.metadata.access]:
        child._metadata = child.metadata.model_copy(update={"access": parent.access, "access_url": parent.access_url})


def _record_parent(child: "TimeFDataset", alias: ParentAlias, parent: "TimeFReader") -> None:
    """Record ``parent`` in ``child``: its card under ``alias``, its lineage, and its stricter access."""
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    _inherit_access(child, parent.metadata)


def _read_records(
    parent: "TimeFReader",
    whole: Iterable[EntityId],
    signal_ids: Iterable[EntityId],
    built: dict[EntityId, Record],
    load: LoadPlan,
) -> list[tuple[Record, frozenset[EntityId] | None]]:
    """Read the parent Records to import, with the parent IDs of the Signals to bring from each.

    Args:
        parent: A reader of the parent release.
        whole: The parent IDs of the Records to bring whole.
        signal_ids: The parent IDs of the Signals to bring, inside a partial copy of their Record
            unless that comes whole.
        built: The parent Records that the import read already, by parent ID. Only the others are
            read, in one batch.
        load: Which kinds of stored values the batch loads.

    Returns:
        Each Record once, in the order selected, with the parent IDs of the Signals to bring from it,
        or ``None`` for the whole Record.
    """
    wanted: dict[EntityId, set[EntityId] | None] = dict.fromkeys(whole)
    signal_ids = tuple(dict.fromkeys(signal_ids))
    holders = parent._record_ids_of_signals(signal_ids) if signal_ids else {}
    for signal_id in signal_ids:
        partial = wanted.setdefault(holders[signal_id], set())
        if partial is not None:
            partial.add(signal_id)
    missing = tuple(identity for identity in wanted if identity not in built)
    originals = built | dict(zip(missing, parent.get_many(Record, missing, load=load) if missing else (), strict=True))
    return [
        (originals[identity], None if partial is None else frozenset(partial)) for identity, partial in wanted.items()
    ]


def import_from(  # noqa: PLR0913 - one argument per selection
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: ParentAlias | None = None,
    records: Iterable[EntityId] = (),
    signals: Iterable[EntityId] = (),
    tasks: Iterable[EntityId] = (),
    load: LoadPlan = LoadPlan.ALL_LAZY,
) -> ImportedDataset:
    """Copy the selected parent Records, Signals, and tasks into ``child``; see :meth:`TimeFDataset.import_from`.

    Returns:
        The child copies of the selected objects, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, a selection is not a collection of parent IDs,
            ``tasks`` is given for a child that streams its tasks, a task to copy is a
            ``TSCorrespondenceTask`` without ``candidate_records``, or ``add_tasks`` rejects the
            task copies.
        TimeFNotFoundError: If a selected ID names no parent object. The child is left unchanged.
    """  # noqa: DOC502 - raised by the helpers and the parent's reader
    record_ids = _parent_ids("records", records)
    signal_ids = _parent_ids("signals", signals)
    task_ids = _parent_ids("tasks", tasks)
    if task_ids and child.has_task_stream:
        raise TimeFValidationError(
            f"{child.metadata.dataset_id} streams its tasks, so tasks= cannot add to them; "
            "import tasks into a dataset that holds its tasks"
        )
    release = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    alias = _resolve_alias(child, release, alias)
    # The parent tasks to copy, with the tasks they derive from, the copies the child holds, and
    # the Records the read of the tasks built, which hold every annotation the tasks reference.
    pending, held, built, on_dataset = (
        _tasks_to_copy(child, alias, parent, task_ids, load) if task_ids else ({}, {}, {}, ())
    )
    _check_candidates(pending.values(), release)
    task_records, task_signals = _task_dependencies(pending.values())
    # A Record that holds a Signal target and an annotation that a task references comes whole, so
    # its copy holds the owner of the annotation.
    whole = (*record_ids, *task_records, *_annotation_holders(pending.values(), built, task_records))
    selected = _read_records(parent, whole, (*signal_ids, *task_signals), built, load)
    # Every read succeeded. Record the parent first, so the copies below always have their card,
    # even if a later one fails.
    _record_parent(child, alias, parent)
    importer = _Importer(child, alias, release)
    for original, wanted in selected:
        importer.import_record(original, wanted)
    copies = (
        _TaskCopy(importer, (original for original, _ in selected), on_dataset).copy_tasks(pending, held)
        if task_ids
        else {}
    )
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
        signals={identity: importer.signals[identity] for identity in signal_ids},
        tasks={identity: copies[identity] for identity in task_ids},
    )
