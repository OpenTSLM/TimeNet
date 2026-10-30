"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable. To remove a row, you must write a new version without that row.
The cheap method is copy-on-write. This method reads the base version into memory. The values
stay lazy and come from the base shards. The method removes records or Signals, repairs each
cross-reference, and writes a new version with the normal atomic-commit writer. Ids are stable
and the system never reuses them, so surviving references stay valid without renumbering. On a
content-addressed or deduplicating backend, the rewrite stores only the chunks that changed.
"""

from collections.abc import Callable, Collection, Container, Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

from timenet.dataset.composition import _copy_annotation, _fits
from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFEditError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry._paths import release_dir, split_release_dir
from timenet.registry.dependencies import open_local_version
from timenet.types import Annotation, DatasetSchema, EntityId, Task, TimeInterval, TimeSpan, Version
from timenet.types.object_kind import ObjectKind
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def remove_records(dataset: TimeFDataset, record_ids: Iterable[EntityId], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset without the given records.

    The dataset does not contain the records in ``record_ids``. If ``cascade`` is set, the
    dataset also does not contain their dependents.

    The method repairs every surviving cross-reference. It removes deleted records from task
    inputs. A task references its inputs, candidate Records, and Record targets, and the Records
    that hold the Signals it references. It becomes invalid if the edit removes all of its inputs,
    an object that it uses as a target or candidate, or a Signal that its scope or a span names. It
    also becomes invalid if the edit removes an annotation that holds its answer or a task that it
    derives from. An input annotation on a removed Record is context, so the task keeps going
    without it. If ``cascade`` is set, the method removes an invalid task. If ``cascade`` is not
    set, the method rejects the edit, so a committed version never contains a dangling reference.

    The new dataset keeps copies of the parent cards and the lineage, and of the import map without
    the entries of the removed records, their Signals, and the removed tasks, so a later import
    into it extends the copies it holds and copies a removed one again. Its Records, Sources, and
    Signals are copies, with their own metadata and their own copies of the annotations. The
    annotations of the dataset and of the tasks are copies too, and the tasks point at the copies.
    So a change to a copy, or an import that extends it, leaves ``dataset`` as it is. Only the
    stored values stay shared.

    An edit refuses a dataset that streams its tasks, because it checks only the tasks in the list
    of the dataset.

    Args:
        dataset: The base dataset. This is typically read back from a committed version.
        record_ids: The record ids to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If ``dataset`` streams its tasks, ``record_ids`` is a string, an id is
            unknown, or a required reference dangles and ``cascade`` is off.
    """
    _refuse_task_stream(dataset)
    remove = _unique_ids_from_argument("record_ids", record_ids)
    known = {record.id for record in dataset.records}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown record ids: {sorted(unknown)}")

    copies: dict[int, Annotation] = {}
    annotations = _copied(dataset.annotations, copies)
    records = [_copy_record(record, copies) for record in dataset.records if record.id not in remove]
    reachable = _reachability(dataset, records, copies)
    removed_signal_ids = {signal.id for record in dataset.records if record.id in remove for signal in record.signals}
    invalid = {task.id for task in dataset.tasks if _task_invalidated(task, remove, removed_signal_ids, reachable)}
    removed_task_ids = _without_dependents(dataset, invalid, f"removing {sorted(remove)}", cascade=cascade)

    tasks = _rebuild_tasks(dataset, removed_task_ids, {record.id: record for record in records}, reachable, copies)
    removed = {ObjectKind.RECORD: remove, ObjectKind.SIGNAL: removed_signal_ids, ObjectKind.TASK: removed_task_ids}
    return _edited(dataset, records, annotations, tasks, removed)


def remove_signals(dataset: TimeFDataset, signal_ids: Iterable[EntityId], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset without the given Signals.

    Each Signal goes with its own annotations. Its Record and the Sources above it lose the
    annotations that the removed Signals placed and the Signals they keep do not, by the rule an
    import attaches annotations by: while a Record or Source keeps each of its Signals, all its
    annotations stay; otherwise an annotation stays if it has no span, or if its span lies inside
    the windows of the Signals it keeps or, for a Record with a ``time_span``, inside that session
    span. Records and Sources stay, even when they hold no Signal any more. This works on any
    dataset; on an imported copy, it undoes what importing the Signals attached, so a later import
    that copies them back attaches those annotations again, once.

    A task becomes invalid if it references a removed Signal, as a target or in a span, or an
    answer annotation that the edit drops; if an input Record is left without Signals; or if one
    of its spans that names no Signal no longer lies where the Signals of its input place it, by
    the same rule as an annotation. If ``cascade`` is set, the method removes such a task and the
    tasks derived from it. If ``cascade`` is not set, the method rejects the edit. An input
    annotation that the edit drops is context, so the task keeps going without it.

    The new dataset keeps copies of the parent cards and the lineage, and of the import map without
    the entries of the removed Signals and tasks. Its Records, Sources, the Signals it keeps, and
    the annotations are copies, as with :func:`remove_records`. An edit refuses a dataset that
    streams its tasks.

    Args:
        dataset: The base dataset.
        signal_ids: The IDs of the Signals to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If ``dataset`` streams its tasks, ``signal_ids`` is a string, an ID names no
            Signal, or the edit invalidates a task and ``cascade`` is off.
    """
    _refuse_task_stream(dataset)
    remove = _unique_ids_from_argument("signal_ids", signal_ids)
    known = {signal.id for record in dataset.records for signal in record.signals}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown signal ids: {sorted(unknown)}")

    copies: dict[int, Annotation] = {}
    annotations = _copied(dataset.annotations, copies)
    records = [_copy_record(record, copies, remove) for record in dataset.records]
    kept = {record.id: record.signals for record in records}
    reachable = _reachability(dataset, records, copies)
    invalid = {task.id for task in dataset.tasks if _loses_signals(task, remove, kept, reachable)}
    removed_task_ids = _without_dependents(dataset, invalid, f"removing signals {sorted(remove)}", cascade=cascade)

    tasks = _rebuild_tasks(dataset, removed_task_ids, {record.id: record for record in records}, reachable, copies)
    return _edited(dataset, records, annotations, tasks, {ObjectKind.SIGNAL: remove, ObjectKind.TASK: removed_task_ids})


def _refuse_task_stream(dataset: TimeFDataset) -> None:
    """Refuse a dataset that streams its tasks.

    An edit checks and rebuilds only the tasks in the list of a dataset. On a dataset that streams
    its tasks, that list is empty, so the edit would drop every streamed task without a message.

    Raises:
        TimeFEditError: If ``dataset`` streams its tasks.
    """
    if dataset.has_task_stream:
        raise TimeFEditError(
            "edits do not support a dataset with a task stream; edit before set_task_stream, or write the release "
            "and edit that"
        )


def _unique_ids_from_argument(argument_name: str, ids: Iterable[EntityId]) -> set[EntityId]:
    """Return the IDs that a caller passed as the argument ``argument_name``, each once.

    Raises:
        TimeFEditError: If ``ids`` is a string, which would name one ID per character.
    """
    if isinstance(ids, str | bytes):
        raise TimeFEditError(f"{argument_name} takes a collection of IDs, not the string {ids!r}")
    return set(ids)


def _edited(
    dataset: TimeFDataset,
    records: list[Record],
    annotations: tuple[Annotation, ...],
    tasks: list[Task],
    removed: Mapping[ObjectKind, Container[EntityId]],
) -> TimeFDataset:
    """Build an edit's dataset, with the import state of ``dataset`` and its schema re-derived.

    Args:
        dataset: The base dataset.
        records: The edit's Records.
        annotations: The edit's copies of the annotations of the dataset itself.
        tasks: The edit's tasks.
        removed: The IDs of the objects the edit removed, by kind, whose import map entries go.

    Returns:
        The new dataset.
    """
    # Use an empty schema, not the base dataset's schema. The derive_schema() call below
    # overwrites it anyway, and `dataset.schema or dataset.derive_schema()` mutated the input
    # dataset's cached schema as a side effect. Re-deriving the schema from the survivors also
    # drops spec, annotation, and task types that existed only on a removed record.
    edited = TimeFDataset.from_parts(
        metadata=dataset.metadata,
        records=records,
        tasks=tasks,
        schema=DatasetSchema(),
        annotations=annotations,
        last_ids=dataset.last_ids,
        import_state=dataset._import_state.without(removed),
    )
    edited.derive_schema()
    return edited


def _reachability(
    dataset: TimeFDataset, records: Iterable[Record], copies: Mapping[int, Annotation]
) -> Callable[[Task, Annotation], bool]:
    """Return a check of whether a task can still resolve an annotation that it references after an edit.

    The check builds no set per task. It finds the Record of an annotation in a map that the call
    builds once.

    Args:
        dataset: The base dataset. The check takes from it the occurrences on the dataset itself, and
            the Records that each task references (see ``TimeFDataset._referenced_records``).
        records: The edit's Records, with the copies of the annotations that the edit keeps.
        copies: The edit's copy of each annotation that it keeps, keyed by the ``id()`` of the original.

    Returns:
        The check. It takes a task and an annotation of ``dataset`` that the task references. It
        returns whether the annotation lies on the dataset itself, or whether its copy lies below an
        edit's Record that the task references.
    """
    record_ids = {id(annotation): record.id for record in records for annotation in record.walk_annotations()}
    on_dataset = frozenset(id(annotation) for annotation in dataset.annotations)

    def reachable(task: Task, annotation: Annotation) -> bool:
        if id(annotation) in on_dataset:
            return True
        copy = copies.get(id(annotation))
        record_id = None if copy is None else record_ids.get(id(copy))
        return record_id is not None and any(record.id == record_id for record in dataset._referenced_records(task))

    return reachable


def _copied(annotations: tuple[Annotation, ...], copies: dict[int, Annotation]) -> tuple[Annotation, ...]:
    """Return a copy of each annotation, and keep each copy in ``copies``.

    The copies have their own value and metadata (see ``composition._copy_annotation``). An
    annotation that ``copies`` already holds keeps that copy, so an annotation object that two owners
    share gets one copy.

    Args:
        annotations: The annotations to copy.
        copies: The edit's copy of each annotation so far, keyed by the ``id()`` of the original. The
            call adds its copies.

    Returns:
        The copies, in the order of ``annotations``.
    """
    for annotation in annotations:
        if id(annotation) not in copies:
            copies[id(annotation)] = _copy_annotation(annotation)
    return tuple(copies[id(annotation)] for annotation in annotations)


def _copy_record(record: Record, copies: dict[int, Annotation], removed: Container[EntityId] = frozenset()) -> Record:
    """Return a copy of a Record without the given Signals, and the annotations only those Signals placed.

    The copy keeps the Record's ID and has its own metadata. It is not validated again: it holds
    what the Record held, less what the edit removes.

    Args:
        record: The Record to copy.
        copies: The edit's copy of each annotation so far (see :func:`_copied`).
        removed: The IDs of the Signals to leave out.

    Returns:
        The copy, whose Sources, Signals, and annotations are copies too.
    """
    duplicate = copy(record)
    duplicate.sources = tuple(_copy_source(source, copies, removed) for source in record.sources)
    duplicate.annotations = _copied(_kept_annotations(record, removed, record.time_span), copies)
    duplicate.metadata = deepcopy(record.metadata)
    return duplicate


def _copy_source(source: Source, copies: dict[int, Annotation], removed: Container[EntityId] = frozenset()) -> Source:
    """Return a copy of a Source, and of the Sources below it, without the given Signals.

    The copy holds copies of the Signals it keeps (see :func:`_copy_signal`) and has its own
    metadata. It loses the annotations that only the removed Signals placed, and it holds copies of
    the others. An import extends a Source's copy in place, which leaves the original as it is.

    Returns:
        The copy.
    """
    duplicate = copy(source)
    duplicate.sources = tuple(_copy_source(child, copies, removed) for child in source.sources)
    duplicate.signals = tuple(_copy_signal(signal, copies) for signal in source.signals if signal.id not in removed)
    duplicate.annotations = _copied(_kept_annotations(source, removed, None), copies)
    duplicate.metadata = deepcopy(source.metadata)
    return duplicate


def _copy_signal(signal: Signal, copies: dict[int, Annotation]) -> Signal:
    """Return a copy of a Signal with its own metadata and copies of its annotations.

    The copy keeps the Signal's ID and its handles to the stored values, so both read the same
    values. An annotation attached to the copy, by ``annotate`` or by an import, changes the copy
    only.

    Returns:
        The copy.
    """
    duplicate = copy(signal)
    object.__setattr__(duplicate, "metadata", deepcopy(signal.metadata))  # noqa: PLC2801 (Signal is frozen)
    object.__setattr__(duplicate, "annotations", _copied(signal.annotations, copies))  # noqa: PLC2801
    return duplicate


def _kept_annotations(
    owner: Record | Source, removed: Container[EntityId], time_span: TimeInterval | None
) -> tuple[Annotation, ...]:
    """Return the annotations that a Record or Source keeps after the edit removes Signals.

    The rule is the one of a partial import. If the owner keeps all its Signals, it keeps every
    annotation. Otherwise it keeps an annotation without a span, and an annotation whose span lies
    inside the windows of the Signals it keeps, or inside the session span of a Record.

    Args:
        owner: The Record or Source, as it is before the edit.
        removed: The IDs of the Signals the edit removes.
        time_span: The session span of a Record, or ``None``.

    Returns:
        The annotations that stay.
    """
    signals = tuple(owner.walk_signals())
    kept = tuple(signal for signal in signals if signal.id not in removed)
    if len(kept) == len(signals):
        return owner.annotations
    label = owner._label if isinstance(owner, Record) else f"source {owner.name!r}"
    return tuple(annotation for annotation in owner.annotations if _fits(annotation, kept, label, time_span))


def _loses_signals(
    task: Task,
    removed: Collection[EntityId],
    kept: Mapping[EntityId, tuple[Signal, ...]],
    reachable: Callable[[Task, Annotation], bool],
) -> bool:
    """Return whether removing the Signals ``removed`` invalidates ``task``.

    It does if a Signal target is removed, or a span names a removed Signal; if an input Record is
    left without Signals; if a span that names no Signal no longer lies where the kept Signals of
    its input place it; or if an answer annotation is dropped.

    Args:
        task: The task.
        removed: The IDs of the removed Signals.
        kept: The Signals each Record keeps, by Record ID.
        reachable: Whether a task can still resolve an annotation that it references.
    """
    # The Signal targets, and the Signals that the scope and span targets name.
    if any(signal.id in removed for signal in task.referenced_signals()):
        return True
    for record in task.inputs:
        signals = kept[record.id]
        if len(signals) == len(record.signals):
            continue
        if not signals:
            return True
        for span in task.spans():
            if isinstance(span, TimeSpan) and span.time_series is None and not _places(span, signals, record):
                return True
    return not all(reachable(task, annotation) for annotation in task.target_annotations)


def _places(span: TimeSpan, signals: tuple[Signal, ...], record: Record) -> bool:
    """Return whether a span that names no Signal lies inside the windows of the Signals, or of a Record's time span."""
    try:
        check_span_within_window("task span", span, signals, record._label, record.time_span, warn_when_outside=False)
    except TimeFValidationError:
        return False
    return True


def _without_dependents(dataset: TimeFDataset, invalid: set[EntityId], edit: str, *, cascade: bool) -> set[EntityId]:
    """Return the invalid tasks and, with ``cascade``, the tasks derived from them transitively.

    Args:
        dataset: The base dataset.
        invalid: The IDs of the tasks the edit invalidates.
        edit: What the edit removes, for the error.
        cascade: If set, remove invalidated tasks instead of rejecting the edit.

    Returns:
        The IDs of the tasks to drop.

    Raises:
        TimeFEditError: If a task is invalid and ``cascade`` is off.
    """
    if invalid and not cascade:
        raise TimeFEditError(f"{edit} would dangle tasks {sorted(invalid)}; pass cascade=True to remove them")
    if not cascade:
        return invalid
    # Transitively drop tasks that derive from a removed task.
    changed = True
    while changed:
        changed = False
        for task in dataset.tasks:
            if task.id in invalid:
                continue
            if any(parent.id in invalid for parent in task.from_tasks):
                invalid.add(task.id)
                changed = True
    return invalid


def _task_invalidated(
    task: Task,
    remove: set[EntityId],
    removed_signal_ids: set[EntityId],
    reachable: Callable[[Task, Annotation], bool],
) -> bool:
    """Return whether removing ``remove`` strips a required reference from ``task``.

    A Record or Signal used as a target is required by construction, and so is a Signal that a
    span names. Candidate records are also required because silently changing the candidate pool
    changes the question. Target annotations are the answer and therefore cannot be stripped.
    Input annotations are context, so :func:`_rebuild_tasks` removes the ones that are no longer
    reachable.
    """
    targets = task.targets or ()
    target_record_ids = {target.id for target in targets if isinstance(target, Record)}
    candidate_record_ids = {record.id for record in getattr(task, "candidate_records", ())}
    if (target_record_ids | candidate_record_ids) & remove:
        return True
    # The Signal targets, and the Signals that the scope and span targets name.
    if any(signal.id in removed_signal_ids for signal in task.referenced_signals()):
        return True
    if task.inputs and all(record.id in remove for record in task.inputs):
        return True
    return not all(reachable(task, annotation) for annotation in task.target_annotations)


def _rebuild_tasks(
    dataset: TimeFDataset,
    removed_task_ids: set[EntityId],
    records_by_id: Mapping[EntityId, "Record"],
    reachable: Callable[[Task, Annotation], bool],
    copies: dict[int, Annotation],
) -> list[Task]:
    """Rebuild the surviving tasks. Strip out unreachable record, annotation, and from-task references.

    Each Record and Signal that a rebuilt task references, as an input, a candidate, a target, or
    in its scope or a span target, is the edit's copy (see
    :meth:`~timenet.types.Task.with_references`). Each annotation that it references is the edit's
    copy too, and the annotations of the task itself are copies.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks to drop.
        records_by_id: The edit's Record objects, keyed by id. They hold the edit's Signals.
        reachable: Whether a task can still resolve an annotation that it references.
        copies: The edit's copy of each annotation that it keeps (see :func:`_copied`). The call adds
            the copies of the annotations of the tasks.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to the rebuilt instances.
    """
    signals_by_id = {signal.id: signal for record in records_by_id.values() for signal in record.signals}
    rebuilt: dict[EntityId, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        # The edit's own rules: it drops the removed inputs and the input annotations that the task
        # cannot resolve any more, and it connects from_tasks below. A reachable annotation lies on
        # the dataset or on a Record that the edit keeps, so the edit holds a copy of it.
        input_annotations = (annotation for annotation in task.input_annotations if reachable(task, annotation))
        kept = replace(
            task,
            inputs=tuple(record for record in task.inputs if record.id in records_by_id),
            input_annotations=tuple(copies[id(annotation)] for annotation in input_annotations),
            target_annotations=tuple(copies[id(annotation)] for annotation in task.target_annotations),
            annotations=_copied(task.annotations, copies),
            from_tasks=(),
        )
        copy = kept.with_references(records_by_id, signals_by_id)
        copy._id = task.id
        rebuilt[task.id] = copy
    for task in dataset.tasks:
        if task.id in rebuilt:
            rebuilt[task.id].from_tasks = tuple(
                rebuilt[parent.id] for parent in task.from_tasks if parent.id in rebuilt
            )
    return list(rebuilt.values())


def edit_version(
    base_dir: Path,
    *,
    dataset_version: Version,
    remove_record_ids: Iterable[EntityId] = (),
    remove_signal_ids: Iterable[EntityId] = (),
    cascade: bool = False,
    **writer_kwargs: object,
) -> Path:
    """Read a committed version, remove records or Signals, and write the derived version next to it.

    The derived version reuses the base's payload files, so it is written into the base's registry
    as ``<root>/<id>/<dataset_version>/``, where readers find the base. To derive a version in
    another registry, copy the base there first, for example with ``download_version``, then edit
    the copy.

    The derived version keeps the base's parent cards, lineage, and import map, without the
    entries of what it removes, so ``get(..., parent=)`` finds the copies it holds and a later
    import extends them.

    Args:
        base_dir: The committed version directory ``<root>/<org>/<name>/<version>`` to derive from.
        dataset_version: The semantic version for the derived dataset. This must differ from
            the base version.
        remove_record_ids: The record ids to remove.
        remove_signal_ids: The IDs of the Signals to remove, with the annotations only they place;
            see :func:`remove_signals`. They are removed before the records, and a Signal of a
            removed record is left to that removal.
        cascade: If set, remove tasks that either removal invalidates instead of rejecting the edit.
        **writer_kwargs: Extra keyword arguments forwarded to :class:`~timenet.writer.TimeFWriter`.

    Returns:
        The new version directory.

    Raises:
        TimeFEditError: If ``dataset_version`` matches the base version, the IDs are a string or
            name nothing, or a required reference dangles and ``cascade`` is off.
    """
    # The whole write happens inside the reader's context. The edited dataset's series keep lazy
    # loaders that pull values from the base version's shards. writer.write() consumes those
    # values. If the reader's `with` block exits first, the write happens through a closed
    # reader. That only worked because close() silently reopened the shards and leaked the
    # handles.
    root, _ = split_release_dir(base_dir)
    record_ids = _unique_ids_from_argument("remove_record_ids", remove_record_ids)
    signal_ids = _unique_ids_from_argument("remove_signal_ids", remove_signal_ids)
    with TimeFReader(open_local_version(base_dir)) as reader:
        base_version = str(reader.metadata.dataset_version)
        if str(dataset_version) == base_version:
            raise TimeFEditError(f"derived version {dataset_version} must differ from the base version {base_version}")

        edited = reader.read()
        writer_kwargs.setdefault("values_backend", reader.values_backend)  # an edit keeps the base dataset's backend
        signal_ids -= {signal.id for record in edited.records if record.id in record_ids for signal in record.signals}
        if signal_ids:
            edited = remove_signals(edited, signal_ids, cascade=cascade)
        edited = remove_records(edited, record_ids, cascade=cascade)
        edited._metadata = edited.metadata.model_copy(update={"dataset_version": dataset_version})

        with TimeFWriter(root, edited, **writer_kwargs) as writer:  # ty: ignore[invalid-argument-type]
            writer.write()
    return release_dir(root, DatasetRef(dataset_id=edited.metadata.dataset_id, version=dataset_version))
