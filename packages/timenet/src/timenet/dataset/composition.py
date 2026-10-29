"""Copy selected parent Records, Signals, and tasks into a child dataset, sharing their stored values."""

from collections import deque
from collections.abc import Container, Hashable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, TypeAlias

from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
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

The map is the only record of what a dataset imported. A copy gets the next ID of its kind in the
child, so no rule derives its ID from the parent ID. An import looks up a parent object here. Then
it finds the copy by its child ID among the objects of the dataset. If the dataset no longer holds
the copy, the import copies the object again. The map holds plain IDs, and ``dict(...)`` copies it.
The writer stores each entry whose copy the dataset holds in the ``imported_objects`` table. A
dataset that a reader reads back from storage gets these entries back. An edit that builds a new
dataset gives it a copy of the map, without the entries of the copies that the edit removes.
"""


@dataclass(frozen=True)
class ImportedDataset:
    """The child copies of the objects that one ``import_from`` call selected, by their parent IDs.

    Each copy is one that the call made or one that the child held already. The view lists only what
    the call selected. For example, a Record that the call copied to hold a selected Signal or for a
    selected task is in the dataset, but not in ``records``. The same is true for a task that a
    selected task derives from.
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
        held: The child copy of each parent Signal that the child holds, by parent ID.

    Returns:
        The copy of each parent Source that the child holds, keyed by the ``id()`` of the parent Source.
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
    """Return whether the span of an annotation lies inside the windows of the given Signals.

    The window of a Signal is the time range that its values cover. The rule is the rule of
    ``annotate``, without its warning. An annotation without a span fits. If a Record has a
    ``time_span``, this session span replaces the windows of the Signals.

    Args:
        annotation: The annotation to check.
        signals: The Signals that the owner of the annotation holds.
        owner: The owner, named in the message of the check.
        time_span: The session span of the Record that owns the annotation, or ``None`` for a Source.
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


def _span_on(span: Span, signals: Mapping[EntityId, Signal]) -> Span:
    """Return a span with each Signal that it names replaced by the Signal of the same ID in ``signals``.

    Args:
        span: A task's scope or span target.
        signals: The Signals to point at, by ID.

    Returns:
        The new span, or ``span`` itself if it names no Signal.
    """
    if isinstance(span, StepSpan):
        return replace(span, time_series=signals[span.time_series.id])
    if isinstance(span, TimeSpan) and span.time_series is not None:
        return replace(span, time_series=tuple(signals[signal.id] for signal in span.time_series))
    return span


def _hierarchy(record: Record) -> list[tuple[Record | Source, dict[str, tuple]]]:
    """Return the fields that an import changes when it adds to the copy of a Record.

    The import restores these fields when it fails.

    Returns:
        The Record and each of its Sources, with its ``sources``, its ``annotations``, and, for a
        Source, its ``signals``.
    """
    return [
        (container, {name: getattr(container, name) for name in names})
        for container, names in (
            (record, ("sources", "annotations")),
            *((source, ("sources", "signals", "annotations")) for source in record.walk_sources()),
        )
    ]


class _Importer:
    """Copy parent Records into one child under one parent alias, and add to the copies that the child holds."""

    def __init__(self, child: "TimeFDataset", alias: ParentAlias) -> None:
        self.child = child
        self.alias = alias
        self.records: dict[EntityId, Record] = {}
        """The child copy of each parent Record that this call found or copied, by parent ID."""
        self.signals: dict[EntityId, Signal] = {}
        """The child copy of each parent Signal that this call found or copied, by parent ID."""
        self.sources: dict[int, Source] = {}
        """The child copy of each parent Source that this call found or copied, keyed by the ``id()`` of
        the parent Source."""
        self.annotations: dict[int, Annotation] = {}
        """The copy of each parent annotation that this call copied, keyed by the ``id()`` of the parent
        annotation."""

    def import_record(self, original: Record, wanted: frozenset[EntityId] | None) -> Record:
        """Copy a parent Record with the wanted Signals, or add them to the copy that the child holds.

        A new copy and its Signals get their IDs when the child registers the copy. A Signal that the
        call adds to a copy gets its ID when the child indexes the copy again. If the call fails to
        add to a copy, it restores the Sources, Signals, and annotations of the copy. The indexes
        and the ID counters of the dataset then do not change.

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
                # Time origins are frozen, like axes, so the copy shares the time origin of the parent
                # Record. Thus copies of Records that share a time origin also share it in the child.
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
        """Copy a parent Signal without its ID, with its axis and the handles to its stored values.

        Returns:
            The copy, with copies of its annotations. The child gives the copy an ID when it indexes
            the copy.
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
        """Copy a parent annotation occurrence as a value, with the handle of a stored value.

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
    """Copy the wanted Sources and Signals of one parent Record into the copy of the Record in the child.

    The call finds the Signals that the copy holds in the hierarchy of the copy, through the import
    map. So a Signal that the child removed from the copy counts as missing, whatever the index of
    the dataset says.
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
        # A new copy holds no Signal yet. A registered copy holds Signals with child IDs.
        own = {signal.id: signal for signal in target.signals}
        self.held: dict[EntityId, Signal] = {
            parent_id: own[child_id]
            for parent_id, key in keys.items()
            if (child_id := imports.get(key)) is not None and child_id in own
        }
        """The copy of each parent Signal that the copy of the Record holds, by parent ID."""
        self.held_before = frozenset(self.held)
        """The parent IDs of the Signals that had a copy in the copy of the Record when the call began."""
        self.imported = {id(signal) for signal in self.held.values()}
        """The ``id()`` of each Signal copy in ``held``, which tells the imported Signals apart from the
        Signals that the child added."""
        self.pairs = _pair_sources(original, target, self.held)
        """The copy of each parent Source that the copy of the Record holds, keyed by the ``id()`` of the parent Source.

        It starts with the copies that the call finds through the Signals of the copy. The merge adds
        each copy that it finds by name or makes.
        """

    def merge(self, *, new: bool) -> None:
        """Copy the wanted Sources and Signals into the copy of the Record, then the annotations that they bring."""
        if self._merge_sources(self.original.sources, self.target) or new:
            self._attach_annotations(self.original, self.target, new=new)

    def _wants(self, signal: Signal) -> bool:
        """Return whether the call wants a parent Signal, and the copy of the Record lacks it."""
        return signal.id not in self.held and (self.wanted is None or signal.id in self.wanted)

    def _brings(self, original: Source) -> bool:
        """Return whether the call copies anything below a parent Source that has no copy yet.

        A Source without a Signal at any depth comes only with its whole Record.
        """
        signals = tuple(original.walk_signals())
        return any(self._wants(signal) for signal in signals) if signals else self.wanted is None

    def _merge_sources(self, originals: Iterable[Source], target: Record | Source) -> bool:
        """Copy the parent Sources below ``target``, or add to the copies of them that ``target`` holds.

        The call finds the copy of a parent Source through the copies of the Signals below it. If
        none of them finds a copy, the call looks for the name of the Source among the copies below
        ``target`` that hold no imported Signal. A Signal that the child added to a copy does not
        count. Sibling Sources have distinct names, so the name finds one copy at most.

        Returns:
            Whether ``target`` gained a Signal at any depth.
        """
        grew = False
        added: list[Source] = []
        unmatched = {
            source.name: source
            for source in target.sources
            if not any(id(signal) in self.imported for signal in source.walk_signals())
        }
        for original in originals:
            copy = self.pairs.get(id(original))
            if copy is None:
                copy = unmatched.pop(original.name, None)
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
        """Copy the wanted descendants of a parent Source into its copy, then the annotations that they bring.

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
        """Attach copies of the parent annotations that the new Signal copies of the call bring.

        The rules look at the Signals below ``original`` and at their copies. The copy of
        ``original`` keeps an annotation of ``original`` in these cases:

        - The copy holds a copy of every Signal below ``original``.
        - The annotation has no span.
        - The span of the annotation lies inside the windows of the Signals that have a copy. For a
          Record with a ``time_span``, the session span replaces the windows.

        A new copy gets every annotation that the rules keep. A copy that the child held before the
        call gets only the annotations that the rules keep now and did not keep when the call began.
        So its other annotations stay as the child left them. This includes the annotations that the
        child deleted.

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
    """Yield the name and the value of each field that the constructor of a task takes.

    The call reads the dataclass fields instead of a list of names, so it finds every field of every
    task class.

    Yields:
        Each field's name and the task's value for it.
    """
    for task_field in fields(task):
        if task_field.init:
            yield task_field.name, getattr(task, task_field.name)


def _leaves(value: object) -> Iterator[object]:
    """Yield the items of the value of a task field, and the items inside its tuples.

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
    """Sort the selected parent tasks and their ``from_tasks`` into tasks to copy and copies in the child.

    If the child holds a copy of a selected task, the call does not read that task from the parent.
    A copy in the child derives from copies in the child too, so the walk stops there. The read of
    the other tasks loads the stored values that ``load`` names into the tasks and into the Records
    that the read builds.

    Returns:
        The parent tasks to copy and the copy in the child of each other task, both by parent ID.
        Also every parent Record that the read of the tasks built, by parent ID, and the occurrences
        of the parent dataset itself that the tasks reference.
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
    """Return the Records and Signals that tasks reference, but not the tasks that they derive from.

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
    """Return the parent IDs of the other Records that hold an annotation that a task references.

    A task references annotations on the parent dataset and on the Records that it references. The
    Records that it names come whole. A Record that it references only through a Signal target comes
    as a partial copy, and that copy can lack the owner of the annotation. So such a Record comes
    whole when it holds an annotation that a task references.

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
    """Return the key, the span, and the source of an annotation, which group the annotations of an owner."""
    return annotation.key, annotation.span, annotation.source


def _same_occurrence(annotation: Annotation, original: Annotation) -> bool:
    """Return whether an annotation in the group of a parent annotation is the same occurrence.

    The group compares the key, the span, the source, and the content. This call compares the fields
    of one attachment: the confidence and the occurrence metadata.
    """
    return (
        annotation.confidence == original.confidence and annotation.occurrence_metadata == original.occurrence_metadata
    )


def _holds_reference(value: object) -> bool:
    """Return whether a value is, or holds at any depth, an object of a dataset or a span.

    Returns:
        Whether a copy of the value carries an object of the parent.
    """
    if isinstance(value, Record | Source | Signal | Annotation | Task | Span):
        return True
    if isinstance(value, dict):
        return any(_holds_reference(item) for item in (*value.keys(), *value.values()))
    if isinstance(value, list | tuple | set | frozenset):
        return any(_holds_reference(item) for item in value)
    return False


_Group: TypeAlias = dict[tuple[Hashable, ...], list[Annotation]]
"""The annotations of an owner with one key, span, and source, by their content key."""


def _take(group: _Group, original: Annotation) -> Annotation | None:
    """Remove and return the first annotation of a group that is the same occurrence as ``original``.

    Returns:
        The annotation, or ``None`` if the group holds no such annotation.
    """
    entries = group.get(original._content_key)
    index = next((index for index, held in enumerate(entries or ()) if _same_occurrence(held, original)), None)
    return None if entries is None or index is None else entries.pop(index)


def _set_annotations(owner: Record | Source | Signal, annotations: tuple[Annotation, ...]) -> None:
    """Replace the annotations of a Record, Source, or Signal."""
    if isinstance(owner, Signal):
        # A Signal is frozen. Signal.annotate changes its annotations the same way.
        object.__setattr__(owner, "annotations", annotations)  # noqa: PLC2801
    else:
        owner.annotations = annotations


class _TaskCopy:
    """Copy parent tasks into one child, and point each reference at the copy of the object in the child.

    A reference to a Record or Signal points at the copy that the call found or made. A reference to
    an annotation points at the copy on the copy of its owner. For an annotation of the parent
    dataset, it points at the copy on the child dataset.
    """

    def __init__(self, importer: _Importer, originals: Iterable[Record], on_dataset: Iterable[Annotation]) -> None:
        self.importer = importer
        self.owners: dict[int, tuple[Record | Source | Signal, Record]] = {
            id(annotation): (owner, record)
            for record in originals
            for owner in (record, *record.walk_sources(), *record.walk_signals())
            for annotation in owner.annotations
        }
        """The parent Record, Source, or Signal that holds each annotation of the Records that the call
        imports, with its Record, keyed by the ``id()`` of the annotation. The read of the tasks built
        these objects."""
        self.on_dataset = {id(annotation) for annotation in on_dataset}
        """The ``id()`` of each occurrence of the parent dataset itself that the read of the tasks built."""
        self.made = {id(copy) for copy in importer.annotations.values()}
        """The ``id()`` of each annotation copy that this call made. Each such copy belongs to its own
        parent annotation."""
        self.resolved: dict[int, Annotation] = {}
        """The copy from an earlier call of each parent annotation that a task references, keyed by the
        ``id()`` of the parent annotation."""
        self.filed: dict[int, dict[tuple[Hashable, ...], _Group]] = {}
        """The annotations of each owner copy that no reference took yet, grouped by :func:`_bucket` and
        then by content key. The key is the ``id()`` of the owner copy, or the ``id()`` of the child
        for the annotations of the child dataset itself."""
        self.saved: dict[int, tuple[Record | Source | Signal, tuple[Annotation, ...]]] = {}
        """Each owner copy that the call attached an annotation to, with the annotations that it held
        before."""
        self.task: Task | None = None
        """The parent task that the call copies now."""

    def copy_tasks(self, pending: dict[EntityId, Task], held: dict[EntityId, Task]) -> dict[EntityId, Task]:
        """Copy and register the pending tasks, then add each copy to the import map.

        The copies register together, all or none. Each copy gets the next task ID of the child. If
        the registration fails, the call removes the annotations that it attached for the copies,
        and the map does not change. So the child holds no task of this call.

        Args:
            pending: The parent tasks to copy, by parent ID.
            held: The copy in the child of each other task, by parent ID.

        Returns:
            The copy in the child of every task, pending or held, by parent ID.
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
        """Copy a parent task field by field, and leave its ``from_tasks`` for the caller to connect.

        The call copies the annotations of the task itself. Every other reference points at the copy
        in the child. The copy has no ID until the child registers it.

        Returns:
            The copy, of the class of the parent task.
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
        """Return the value of a task field, with each Record, Signal, annotation, and span pointed at the child.

        Returns:
            The value for the copy. The call copies a value that references nothing.

        Raises:
            TimeFValidationError: If the value holds a reference in a container other than a tuple.
                A copy of such a value keeps the object of the parent.
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
            return _span_on(value, self.importer.signals)
        if _holds_reference(value):
            raise TimeFValidationError(
                f"{type(self.task).__name__}.{name} holds a {type(value).__name__} that references objects; "
                "an import remaps references only inside tuples"
            )
        return deepcopy(value)

    def _annotation(self, original: Annotation) -> Annotation:
        """Return the child's copy of a parent annotation that a task references.

        The owner is the Record, Source, or Signal that holds the annotation. The lookup uses the first
        of these steps that finds a copy:

        1. The copy that this call already made or returned for this annotation.
        2. An annotation on the child's copy of the owner with the same key, span, source, content,
           confidence, and occurrence metadata. Each such match serves one reference only, so two
           identical annotations that one call references get two copies. Separate calls can return
           the same match.
        3. A new copy, attached to the owner's copy. This happens when the child removed or changed
           the annotation. A changed annotation stays next to the new copy.

        An annotation of the parent dataset itself goes to the child dataset, not to an owner.

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
        """Attach a copy of a parent annotation to the copy of its owner, or to the child dataset.

        An annotation of the parent dataset itself goes to the child dataset.

        Returns:
            The copy. It is an occurrence of its own, next to any identical occurrence on the copy of
            the owner.
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
        """Return the annotations of the copy of an owner, grouped at the first reference of the call to it.

        The groups leave out each copy that this call made. Such a copy belongs to its own parent
        annotation, so no other reference takes it.

        Args:
            owner: The copy of the owner, or ``None`` for the child dataset.

        Returns:
            The annotations that no reference took yet, grouped by :func:`_bucket` and then by content key.
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
            TimeFValidationError: If the copy of the Record of the annotation holds no copy of its
                Source.
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


def _parent_ids(selection: str, ids: Iterable[EntityId]) -> tuple[EntityId, ...]:
    """Return the parent IDs that a selection names, in order, each once.

    Returns:
        The IDs.

    Raises:
        TimeFValidationError: If ``ids`` is a string, because a string selects one ID per
            character, or if ``ids`` is not a collection of positive int64 IDs.
    """
    if isinstance(ids, str | bytes):
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, not the string {ids!r}")
    try:
        selected = tuple(ids)
    except TypeError as exc:
        raise TimeFValidationError(f"{selection}= takes a collection of parent IDs, got {type(ids).__name__}") from exc
    return tuple(dict.fromkeys(check_id(identity, "parent id") for identity in selected))


def _resolve_alias(child: "TimeFDataset", parent: DatasetRef, alias: ParentAlias | None) -> ParentAlias:
    """Return the parent alias under which the card of the child declares ``parent``.

    Returns:
        The alias.

    Raises:
        TimeFValidationError: If the ``metadata.parents`` of the child or ``alias`` does not declare
            ``parent``, if the card declares ``parent`` under several aliases and ``alias`` is not
            given, if the child imports ``parent`` under another alias already, or if ``alias``
            imports another release already.
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


def _record_parent(child: "TimeFDataset", alias: ParentAlias, parent: "TimeFReader") -> None:
    """Record ``parent`` in ``child``: its card under ``alias`` and its lineage."""
    card = ParentDatasetCard(metadata=parent.metadata)
    child.parent_cards[alias] = card
    for ancestor in (*parent._manifest.lineage, card):
        child.lineage[str(ancestor.source)] = ancestor


def _read_records(
    parent: "TimeFReader",
    whole: Iterable[EntityId],
    signal_ids: Iterable[EntityId],
    built: dict[EntityId, Record],
    load: LoadPlan,
) -> list[tuple[Record, frozenset[EntityId] | None]]:
    """Read the parent Records to import, with the parent IDs of the Signals to copy from each.

    Args:
        parent: A reader of the parent release.
        whole: The parent IDs of the Records to copy whole.
        signal_ids: The parent IDs of the Signals to copy. Each Signal comes in a partial copy of
            its Record, unless the Record comes whole.
        built: The parent Records that the import read already, by parent ID. The call reads only
            the others, in one batch.
        load: Which kinds of stored values the batch loads.

    Returns:
        Each Record once, in the order selected, with the parent IDs of the Signals to copy from it,
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
    """Copy the selected parent objects into ``child``, as :meth:`TimeFDataset.import_from` describes.

    Returns:
        The child copies of the selected objects, by their parent IDs.

    Raises:
        TimeFValidationError: If the parent is not a declared parent under ``alias``, the alias
            imports another release already, a selection is not a collection of parent IDs,
            ``tasks`` is given for a child that streams its tasks, a task to copy is a
            ``TSCorrespondenceTask`` without ``candidate_records``, or ``add_tasks`` rejects the
            task copies.
        TimeFNotFoundError: If a selected ID names no parent object. The child does not change.
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
    # The parent tasks to copy with the tasks that they derive from, the task copies that the child
    # holds, and the Records that the read of the tasks built. These Records hold every annotation
    # that the tasks reference.
    pending, held, built, on_dataset = (
        _tasks_to_copy(child, alias, parent, task_ids, load) if task_ids else ({}, {}, {}, ())
    )
    _check_candidates(pending.values(), release)
    task_records, task_signals = _task_dependencies(pending.values())
    # A Record that holds a Signal target and an annotation that a task references comes whole, so
    # its copy holds the owner of the annotation.
    whole = (*record_ids, *task_records, *_annotation_holders(pending.values(), built, task_records))
    selected = _read_records(parent, whole, (*signal_ids, *task_signals), built, load)
    # Every read succeeded. Record the parent first, so that the copies below always have its card,
    # also when a later copy fails.
    _record_parent(child, alias, parent)
    importer = _Importer(child, alias)
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
