"""Copy selected parent Records, Signals, and tasks into a child dataset, sharing their stored values."""

from collections import deque
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import NAMESPACE_URL, uuid5

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
from timenet.types.ids import new_id
from timenet.types.object_kind import ObjectKind
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef
from timenet.types.text import InlineText, InlineTextList, TextHandle, TextListHandle


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.reader import TimeFReader


ImportMap: TypeAlias = dict[tuple[InputAlias, ObjectKind, str], str]
"""The child ID of each imported Record, Signal, and task, keyed by input alias, kind, and parent ID.

It is the one record of what a dataset imported. An import looks a parent object up here and then
finds the copy by its child ID among the dataset's own objects, so a copy the dataset no longer
holds is copied again. The map holds plain IDs and copies with ``dict(...)``. The writer stores the
entries whose copy the dataset still holds in the ``imported_objects`` table.
"""

_STRICTNESS = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the objects that one ``import_from`` call selected, by their parent IDs.

    Each copy is one the call made or one the child already held. The view lists only what the call
    selected: a Record the call copied to hold a selected Signal or for a selected task is in the
    dataset, not in ``records``, and so is a task that a selected task derives from.
    """

    alias: InputAlias
    """The input alias the objects were imported under."""
    parent: DatasetRef
    """The exact parent release."""
    records: dict[str, Record] = field(default_factory=dict)
    """The copy of each Record selected with ``records=``, keyed by its parent ID."""
    signals: dict[str, Signal] = field(default_factory=dict)
    """The copy of each Signal selected with ``signals=``, keyed by its parent ID."""
    tasks: dict[str, Task] = field(default_factory=dict)
    """The copy of each task selected with ``tasks=``, keyed by its parent ID."""


def _copy_id(parent: DatasetRef, kind: ObjectKind, parent_id: str) -> str:
    """Return the ID a copy of a parent object gets in the child.

    This stands in until datasets assign their own IDs. One parent object always gets the same ID,
    so rebuilding a child reproduces its catalog. Nothing derives this ID to find a copy; the import
    map does that.

    Returns:
        A UUID string derived from the exact parent release, the kind, and the parent ID.
    """
    return str(uuid5(NAMESPACE_URL, f"timenet:{parent}:{kind}:{parent_id}"))


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


def _pair_sources(original: Record, copy: Record, held: dict[str, Signal]) -> dict[int, Source]:
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


def _fits(annotation: Annotation, signals: tuple[Signal, ...], owner_id: str, time_span: TimeInterval | None) -> bool:
    """Return whether an annotation's span lies inside the windows of the given Signals.

    The rule is the one ``annotate`` applies, without its warning: an annotation without a span
    fits, and a Record's ``time_span``, when it has one, replaces the Signals' windows.

    Args:
        annotation: The annotation to place.
        signals: The Signals its owner holds.
        owner_id: The owner's ID, for the check's message.
        time_span: The owning Record's session span, or ``None`` for a Source.
    """
    if annotation.span is None:
        return True
    try:
        check_span_within_window(
            f"annotation {annotation.key!r}", annotation.span, signals, owner_id, time_span, warn_when_outside=False
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
    """Copy parent Records into one child under one input alias, extending the copies the child holds."""

    def __init__(self, child: "TimeFDataset", alias: InputAlias, release: DatasetRef) -> None:
        self.child = child
        self.alias = alias
        self.release = release
        """The exact parent release, which names the copies' IDs."""
        self.records: dict[str, Record] = {}
        """The child copy of each parent Record this call found or copied, by parent ID."""
        self.signals: dict[str, Signal] = {}
        """The child copy of each parent Signal this call found or copied, by parent ID."""
        self.sources: dict[int, Source] = {}
        """The child copy of each parent Source this call found or copied, keyed by the ``id()`` of the parent Source."""
        self.annotations: dict[int, Annotation] = {}
        """The copy of each parent annotation this call copied, keyed by the ``id()`` of the parent annotation."""

    def import_record(self, original: Record, wanted: frozenset[str] | None) -> Record:
        """Copy a parent Record, or extend the child's copy of it, with the wanted Signals.

        If extending a copy fails, its Sources, Signals, and annotations are restored as they were,
        and the dataset's indexes are unchanged.

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
                record_id=_copy_id(self.release, ObjectKind.RECORD, original.id),
                start_time=deepcopy(original.start_time),
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
                self.child._index_record(target, registered=True)
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
        """Copy a parent Signal with IDs of its own, keeping the handles to its stored values.

        Returns:
            The copy, with copies of its axis and annotations.
        """
        axis = original.time_axis
        return Signal.from_loader(
            spec=original.spec,
            time_axis=replace(axis, axis_id=_copy_id(self.release, ObjectKind.AXIS, axis.axis_id)),
            name=original.name,
            loader=original._values,
            n_values=original.n_values,
            id=_copy_id(self.release, ObjectKind.SIGNAL, original.id),
            time_offsets_loader=original._time_offsets,
            annotations=tuple(self.copy_annotation(annotation) for annotation in original.annotations),
            metadata=deepcopy(original.metadata),
        )

    def copy_annotation(self, original: Annotation) -> Annotation:
        """Copy a parent annotation occurrence with IDs of its own, keeping a stored value's handle.

        Returns:
            The copy, whose occurrences of one parent content share one content ID.
        """
        copy = replace(
            original,
            id=_copy_id(self.release, ObjectKind.ANNOTATION_CONTENT, original.id),
            # A parent's reader binds every occurrence it builds to an occurrence ID.
            occurrence_id=_copy_id(self.release, ObjectKind.ANNOTATION, original.occurrence_id or original.id),
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

    def __init__(self, importer: _Importer, original: Record, target: Record, wanted: frozenset[str] | None) -> None:
        self.importer = importer
        self.original = original
        self.target = target
        self.wanted = wanted
        """The parent IDs of the Signals to bring, or ``None`` for the whole Record."""
        imports = importer.child._imports
        keys = {signal.id: (importer.alias, ObjectKind.SIGNAL, signal.id) for signal in original.signals}
        own = {signal.id: signal for signal in target.signals}
        self.held: dict[str, Signal] = {
            parent_id: own[imports[key]] for parent_id, key in keys.items() if imports.get(key) in own
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
                        continue
                    raise TimeFValidationError(
                        f"an import cannot tell apart the Sources named {original.name!r} of Record "
                        f"{self.original.id!r} of {self.importer.release}: no imported Signal identifies "
                        "their copies, and the name does not"
                    )
                copy = candidates[0] if candidates else None
                if copy is not None:
                    unmatched.remove(copy)
            if copy is not None:
                self.pairs[id(original)] = copy
                grew = self._merge_source(original, copy, new=False) or grew
            elif self._brings(original):
                copy = Source(
                    id=_copy_id(self.importer.release, ObjectKind.SOURCE, original.id),
                    name=original.name,
                    metadata=deepcopy(original.metadata),
                )
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
        time_span = original.time_span if isinstance(original, Record) else None

        def places(annotation: Annotation, held: tuple[Signal, ...]) -> bool:
            return len(held) == len(signals) or _fits(annotation, held, original.id, time_span)

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
    child: "TimeFDataset", alias: InputAlias, parent: "TimeFReader", task_ids: tuple[str, ...]
) -> tuple[dict[str, Task], dict[str, Task]]:
    """Sort the selected parent tasks, and those they derive from, into tasks to copy and copies the child holds.

    A selected task the child holds a copy of is not read from the parent. A copy the child holds
    derives from copies it holds too, so the walk stops there.

    Returns:
        The parent tasks to copy, and the child's copy of each other task, both by parent ID.
    """
    by_id = {task.id: task for task in child.tasks}

    def held_copy(identity: str) -> Task | None:
        child_id = child._imports.get((alias, ObjectKind.TASK, identity))
        return None if child_id is None else by_id.get(child_id)

    held = {identity: copy for identity in task_ids if (copy := held_copy(identity)) is not None}
    unread = tuple(identity for identity in task_ids if identity not in held)
    pending: dict[str, Task] = {}
    queue = deque(parent.get_many(Task, unread) if unread else ())
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
    return pending, held


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


def _task_dependencies(tasks: Iterable[Task]) -> tuple[dict[str, Record], dict[str, Signal]]:
    """Return the Records and Signals that tasks reference, apart from the tasks they derive from.

    Returns:
        The Records, which come whole, and the Signals, which come inside a partial copy of their
        Record, by parent ID.
    """
    records: dict[str, Record] = {}
    signals: dict[str, Signal] = {}
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


def _placing_signals(record: Record, occurrence_id: str) -> frozenset[str] | None:
    """Return the fewest Signals of a parent Record whose partial copy holds an annotation occurrence.

    A Signal's annotation needs that Signal. A Record's or a Source's needs the Signals below its
    owner that place it, as an import places it: the first Signal whose window places it, or else
    every Signal below the owner. An annotation that needs no window, one without a span or one
    inside a Record's ``time_span``, still needs the first Signal below its owner: a Source is copied
    only with a Signal below it, and a written Record holds at least one Source. An annotation of
    a Source without Signals needs the whole Record.

    Returns:
        The parent IDs of the Signals, or ``None`` for the whole Record.

    Raises:
        TimeFValidationError: If no Record, Source, or Signal of the Record holds the occurrence.
    """
    for owner in (record, *record.walk_sources(), *record.walk_signals()):
        annotation = next((item for item in owner.annotations if item.occurrence_id == occurrence_id), None)
        if annotation is None:
            continue
        if isinstance(owner, Signal):
            return frozenset({owner.id})
        signals = tuple(owner.walk_signals())
        if not signals:
            return None
        time_span = owner.time_span if isinstance(owner, Record) else None
        placing = next((signal for signal in signals if _fits(annotation, (signal,), owner.id, time_span)), None)
        return frozenset({placing.id} if placing is not None else (signal.id for signal in signals))
    raise TimeFValidationError(f"Record {record.id!r} holds no annotation occurrence {occurrence_id!r}")


def _annotation_records(
    parent: "TimeFReader", tasks: Iterable[Task], named: dict[str, Record]
) -> tuple[dict[str, Record], dict[str, frozenset[str] | None]]:
    """Read the parent Records that hold the annotations tasks reference, apart from those they name.

    Args:
        parent: A reader of the parent release.
        tasks: The parent tasks to copy.
        named: The Records the tasks name, which come whole, by parent ID.

    Returns:
        The Records, by parent ID, and for each, the parent IDs of the Signals that a partial copy
        needs to hold the annotations, or ``None`` for the whole Record.
    """
    held = {annotation.occurrence_id for record in named.values() for annotation in record.walk_annotations()}
    referenced = tuple(
        dict.fromkeys(
            annotation.occurrence_id
            for task in tasks
            for annotation in (*task.input_annotations, *task.target_annotations)
            if annotation.occurrence_id is not None and annotation.occurrence_id not in held
        )
    )
    holders = parent._record_ids_of_occurrences(referenced) if referenced else {}
    ids = tuple(dict.fromkeys(holders.values()))
    records = dict(zip(ids, parent.get_many(Record, ids), strict=True)) if ids else {}
    wanted: dict[str, frozenset[str] | None] = dict.fromkeys(ids, frozenset())
    for occurrence_id, record_id in holders.items():
        signals = _placing_signals(records[record_id], occurrence_id)
        previous = wanted[record_id]
        wanted[record_id] = None if signals is None or previous is None else previous | signals
    return records, wanted


def _bucket(annotation: Annotation) -> tuple[str, TimePoint | TimeInterval | None, str | None]:
    """Return the fields that file an annotation among its owner's: its key, span, and source."""
    return annotation.key, annotation.span, annotation.source


def _same_occurrence(annotation: Annotation, original: Annotation) -> bool:
    """Return whether an annotation holds the same occurrence as a parent one, whatever their IDs.

    The value compares last, so that a differing field spares a read of stored text. A copy keeps
    its parent's handle, and handles from one reader at one stored location compare without a read.
    """
    return (
        _bucket(annotation) == _bucket(original)
        and annotation.confidence == original.confidence
        and annotation.occurrence_metadata == original.occurrence_metadata
        and annotation._has_same_content(original)
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


def _value_key(value: object) -> tuple[object, ...]:
    """Return a key that equal annotation values share, found without a read.

    Inline text keys by its content and a scalar by itself. Stored text keys by its length only, so
    a comparison, which may read it, tells stored values of one length apart. A value held inline
    and one held in storage never share a key.

    Returns:
        The key.
    """
    if isinstance(value, InlineText):
        return "text", value.value
    if isinstance(value, InlineTextList):
        return "texts", tuple(value.value)
    if isinstance(value, TextHandle):
        return "stored text", value.length
    if isinstance(value, TextListHandle):
        return "stored texts", value.length
    return "scalar", value


_Group: TypeAlias = dict[tuple[object, ...], list[Annotation]]
"""The annotations of one bucket of an owner, by :func:`_value_key`."""


def _take(group: _Group, original: Annotation) -> Annotation | None:
    """Remove and return the first annotation of a group that holds the same occurrence as ``original``.

    Returns:
        The annotation, or ``None`` if none does.
    """
    entries = group.get(_value_key(original.value))
    index = next((index for index, held in enumerate(entries or ()) if _same_occurrence(held, original)), None)
    return None if entries is None or index is None else entries.pop(index)


@dataclass
class _Filed:
    """The annotations that one owner's copy holds, filed once per call for references to claim."""

    groups: dict[tuple[object, ...], _Group] = field(default_factory=dict)
    """The annotations no reference took yet, by :func:`_bucket` and then by :func:`_value_key`."""
    occurrence_ids: set[str | None] = field(default_factory=set)
    """The occurrence ID of each annotation the copy holds, and of each copy the call attached."""
    contents: dict[str, Annotation] = field(default_factory=dict)
    """An annotation the copy holds for each content ID, and each copy the call attached."""


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

    def __init__(self, importer: _Importer, originals: Iterable[Record]) -> None:
        self.importer = importer
        originals = tuple(originals)
        self.owners: dict[int, tuple[Record | Source | Signal, Record]] = {
            id(annotation): (owner, record)
            for record in originals
            for owner in (record, *record.walk_sources(), *record.walk_signals())
            for annotation in owner.annotations
        }
        """The parent Record, Source, or Signal that holds each annotation of the Records the call
        imports, with its Record, keyed by the ``id()`` of the annotation. An annotation the tasks
        reference that none holds is one of the parent dataset."""
        self.originals: dict[str | None, Annotation] = {
            annotation.occurrence_id: annotation
            for record in originals
            for annotation in record.walk_annotations()
            if annotation.occurrence_id is not None
        }
        """The annotation of the imported Records that holds each occurrence. A task read may build
        a Record apart from the call's read of it, so a reference is looked up here first."""
        self.made = {id(copy) for copy in importer.annotations.values()}
        """The ``id()`` of each annotation copy this call made, which belongs to its own parent annotation."""
        self.resolved: dict[int, Annotation] = {}
        """The copy, from an earlier call, of each parent annotation a task references, keyed by the
        ``id()`` of the parent annotation."""
        self.filed: dict[int, _Filed] = {}
        """The annotations of each owner copy, keyed by its ``id()``, or by the child's for its own."""
        self.saved: dict[int, tuple[Record | Source | Signal, tuple[Annotation, ...]]] = {}
        """Each owner copy that the call attached an annotation to, with the annotations it held before."""
        self.task: Task | None = None
        """The parent task being copied."""

    def copy_tasks(self, pending: dict[str, Task], held: dict[str, Task]) -> dict[str, Task]:
        """Copy and register the pending tasks, then add each copy to the import map.

        The copies register together, all or none. If they fail, the annotations attached for them
        are removed again and the map is unchanged, so the child holds no task of this call.

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

        Its own annotations are copied; every other reference points at the child's copy.

        Returns:
            The copy, of the parent task's class.
        """
        self.task = original
        values: dict[str, Any] = {}
        for name, value in _task_fields(original):
            if name == "id":
                values[name] = _copy_id(self.importer.release, ObjectKind.TASK, original.id)
            elif name == "from_tasks":
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
            return replace(span, time_series_id=signals[span.time_series_id].id)
        if isinstance(span, TimeSpan) and span.time_series_ids is not None:
            return replace(span, time_series_ids=tuple(signals[identity].id for identity in span.time_series_ids))
        return span

    def _annotation(self, original: Annotation) -> Annotation:
        """Return the child's copy of a parent annotation that a task references.

        A copy this call made is the copy. Otherwise the copy is the annotation on the owner's copy
        that holds the same occurrence and that no other reference took, found by bucket and value,
        so identical occurrences that one call references stay distinct; references in separate
        calls may share one copy. If the owner's copy holds none, because the child removed or
        changed it, a copy of the parent's is attached, beside a changed one. For an annotation of
        the parent dataset, a copy is attached unless one matches.

        Returns:
            The copy.
        """
        original = self.originals.get(original.occurrence_id, original)
        copy = self.importer.annotations.get(id(original))
        if copy is None:
            copy = self.resolved.get(id(original))
        if copy is not None:
            return copy
        owned = self.owners.get(id(original))
        owner = None if owned is None else self._owner_copy(*owned, original)
        filed = self._filed(owner)
        copy = _take(filed.groups.get(_bucket(original), {}), original)
        if copy is not None:
            self.resolved[id(original)] = copy
            return copy
        return self._attach(original, owner, filed)

    def _attach(self, original: Annotation, owner: Record | Source | Signal | None, filed: _Filed) -> Annotation:
        """Attach a copy of a parent annotation to its owner's copy, or to the child for the dataset's.

        The copy gets the occurrence ID derived from the parent's, or a new one if the owner's copy
        holds that ID already: an identical occurrence took the copy that holds it, or the child
        changed that copy. Likewise, it gets a new content ID if the owner's copy holds the derived
        one with other content, because the child changed the value of a copy in place.

        Returns:
            The copy.
        """
        copy = self.importer.copy_annotation(original)
        if copy.occurrence_id in filed.occurrence_ids:
            copy = copy._new_occurrence()
        previous = filed.contents.get(copy.id)
        if previous is not None and not previous._has_same_content(copy):
            copy = replace(copy, id=new_id())
        self.importer.annotations[id(original)] = copy
        filed.occurrence_ids.add(copy.occurrence_id)
        filed.contents.setdefault(copy.id, copy)
        self.made.add(id(copy))
        if owner is None:
            self.importer.child._annotations.append(copy)
        else:
            self.saved.setdefault(id(owner), (owner, owner.annotations))
            _set_annotations(owner, (*owner.annotations, copy))
        return copy

    def _filed(self, owner: Record | Source | Signal | None) -> _Filed:
        """Return the annotations of an owner's copy, filed at the call's first reference to it.

        A copy this call made, and an annotation without an occurrence ID, which no task can
        reference, are not filed for claims.

        Args:
            owner: The owner's copy, or ``None`` for the child dataset.

        Returns:
            The filed annotations.
        """
        held = self.importer.child._annotations if owner is None else owner.annotations
        key = id(self.importer.child) if owner is None else id(owner)
        filed = self.filed.get(key)
        if filed is None:
            filed = self.filed[key] = _Filed()
            for annotation in held:
                if annotation.occurrence_id is None:
                    continue
                filed.occurrence_ids.add(annotation.occurrence_id)
                filed.contents.setdefault(annotation.id, annotation)
                if id(annotation) not in self.made:
                    group = filed.groups.setdefault(_bucket(annotation), {})
                    group.setdefault(_value_key(annotation.value), []).append(annotation)
        return filed

    def _owner_copy(
        self, owner: Record | Source | Signal, record: Record, annotation: Annotation
    ) -> Record | Source | Signal:
        """Return the child's copy of the parent Record, Source, or Signal that holds an annotation.

        Returns:
            The copy.

        Raises:
            TimeFValidationError: If the copy of the annotation's Record holds no copy of its Source.
        """
        if isinstance(owner, Record):
            return self.importer.records[owner.id]
        if isinstance(owner, Signal):
            return self.importer.signals[owner.id]
        copy = self.importer.sources.get(id(owner))
        if copy is None:
            raise TimeFValidationError(
                f"a task references annotation {annotation.key!r} of the Source {owner.name!r} of Record "
                f"{record.id!r}, which the child's copy of that Record holds no copy of"
            )
        return copy


def _parent_ids(selection: str, ids: Iterable[str]) -> tuple[str, ...]:
    """Return the parent IDs a selection names, in order, each once.

    Returns:
        The IDs.

    Raises:
        TimeFValidationError: If ``ids`` is a string, which would select one ID per character, or is
            not a collection of strings.
    """
    if isinstance(ids, str):
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, not the string {ids!r}")
    try:
        selected = tuple(ids)
    except TypeError as exc:
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, got {type(ids).__name__}") from exc
    for identity in selected:
        if not isinstance(identity, str):
            raise TimeFValidationError(f"parent IDs are strings, got {type(identity).__name__} {identity!r}")
    return tuple(dict.fromkeys(selected))


def _resolve_alias(child: "TimeFDataset", parent: DatasetRef, alias: InputAlias | None) -> InputAlias:
    """Return the input alias under which the child's card declares ``parent``.

    Returns:
        The alias.

    Raises:
        TimeFValidationError: If the child's ``metadata.inputs`` does not declare ``parent``, ``alias``
            names no input that declares it, the card declares it under several aliases and ``alias``
            is omitted, the child already imports it under another alias, or ``alias`` already
            imports another release.
    """
    dataset_id = child.metadata.dataset_id
    declared = sorted(name for name, ref in child.metadata.inputs.items() if ref == parent)
    if not declared:
        raise TimeFValidationError(
            f"{dataset_id} does not declare {parent} as an input; declare it in metadata.inputs to import from it"
        )
    if alias is None:
        if len(declared) > 1:
            raise TimeFValidationError(
                f"{dataset_id} declares {parent} under the input aliases {declared}; pass alias= to choose one"
            )
        alias = declared[0]
    elif alias not in declared:
        raise TimeFValidationError(
            f"input alias {alias!r} of {dataset_id} does not declare {parent}, which it declares as {declared}"
        )
    for other, card in child.parent_cards.items():
        if card.source == parent and other != alias:
            raise TimeFValidationError(f"{dataset_id} already imports {parent} under the input alias {other!r}")
    card = child.parent_cards.get(alias)
    if card is not None and card.source != parent:
        raise TimeFValidationError(f"input alias {alias!r} of {dataset_id} already imports {card.source}, not {parent}")
    return alias


def _inherit_access(child: "TimeFDataset", parent: DatasetMetadata) -> None:
    """Raise the child's access requirement to the parent's when the parent's is stricter.

    The child has one ``access_url``: its own, or that of the first parent that made it stricter. A
    parent that is no stricter keeps its access requirement and page in its parent card.
    """
    if _STRICTNESS[parent.access] > _STRICTNESS[child.metadata.access]:
        child._metadata = child.metadata.model_copy(update={"access": parent.access, "access_url": parent.access_url})


def _read_records(
    parent: "TimeFReader",
    whole: Iterable[str],
    signal_ids: Iterable[str],
    built: dict[str, Record],
    more: dict[str, frozenset[str] | None] | None = None,
) -> list[tuple[Record, frozenset[str] | None]]:
    """Read the parent Records to import, with the parent IDs of the Signals to bring from each.

    Args:
        parent: A reader of the parent release.
        whole: The parent IDs of the Records to bring whole.
        signal_ids: The parent IDs of the Signals to bring, inside a partial copy of their Record
            unless that comes whole.
        built: The parent Records that the import read already, by parent ID. Only the others are
            read, in one batch.
        more: More Records to bring, by parent ID, each with the parent IDs of the Signals to
            bring from it, or ``None`` for the whole Record.

    Returns:
        Each Record once, in the order selected, with the parent IDs of the Signals to bring from it,
        or ``None`` for the whole Record.
    """
    wanted: dict[str, set[str] | None] = dict.fromkeys(whole)
    for identity, signals in (more or {}).items():
        if identity not in wanted:
            wanted[identity] = None if signals is None else set(signals)
        elif signals is None:
            wanted[identity] = None
        elif (merged := wanted[identity]) is not None:
            merged.update(signals)
    signal_ids = tuple(dict.fromkeys(signal_ids))
    holders = parent._record_ids_of_signals(signal_ids) if signal_ids else {}
    for signal_id in signal_ids:
        partial = wanted.setdefault(holders[signal_id], set())
        if partial is not None:
            partial.add(signal_id)
    missing = tuple(identity for identity in wanted if identity not in built)
    originals = built | dict(zip(missing, parent.get_many(Record, missing) if missing else (), strict=True))
    return [
        (originals[identity], None if partial is None else frozenset(partial)) for identity, partial in wanted.items()
    ]


def import_from(  # noqa: PLR0913 - one argument per selection
    child: "TimeFDataset",
    parent: "TimeFReader",
    *,
    alias: InputAlias | None = None,
    records: Iterable[str] = (),
    signals: Iterable[str] = (),
    tasks: Iterable[str] = (),
) -> ImportedDataset:
    """Copy the selected parent Records, Signals, and tasks into ``child``; see :meth:`TimeFDataset.import_from`.

    Returns:
        The child copies of the selected objects, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared input under ``alias``, the alias
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
    # The parent tasks to copy, with the tasks they derive from, and the copies the child holds.
    pending, held = _tasks_to_copy(child, alias, parent, task_ids) if task_ids else ({}, {})
    _check_candidates(pending.values(), release)
    task_records, task_signals = _task_dependencies(pending.values())
    annotated, placing = _annotation_records(parent, pending.values(), task_records)
    selected = _read_records(
        parent, (*record_ids, *task_records), (*signal_ids, *task_signals), task_records | annotated, placing
    )
    # Every read succeeded. Record the parent first, so the copies below always have their card,
    # even if a later one fails.
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor
    _inherit_access(child, parent.metadata)
    importer = _Importer(child, alias, release)
    for original, wanted in selected:
        importer.import_record(original, wanted)
    copies = _TaskCopy(importer, (original for original, _ in selected)).copy_tasks(pending, held) if task_ids else {}
    return ImportedDataset(
        alias=alias,
        parent=release,
        records={identity: importer.records[identity] for identity in record_ids},
        signals={identity: importer.signals[identity] for identity in signal_ids},
        tasks={identity: copies[identity] for identity in task_ids},
    )
