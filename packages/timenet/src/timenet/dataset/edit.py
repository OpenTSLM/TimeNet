"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable. To remove a row, you must write a new version without that row.
The cheap method is copy-on-write. This method reads the base version into memory. The values
stay lazy and come from the base shards. The method removes records or Signals, repairs each
cross-reference, and writes a new version with the normal atomic-commit writer. Ids are stable
and the system never reuses them, so surviving references stay valid without renumbering. On a
content-addressed or deduplicating backend, the rewrite stores only the chunks that changed.
"""

from collections.abc import Container, Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

from timenet.dataset.composition import _fits
from timenet.dataset.dataset import TimeFDataset, _TaskValidationIndex
from timenet.dataset.record import Record
from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFEditError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry._paths import release_dir, split_release_dir
from timenet.registry.dependencies import open_local_version
from timenet.types import Annotation, DatasetSchema, Span, StepSpan, Task, TimeInterval, TimeSpan, Version
from timenet.types.object_kind import ObjectKind
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def remove_records(dataset: TimeFDataset, record_ids: Iterable[str], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset without the given records.

    The dataset does not contain the records in ``record_ids``. If ``cascade`` is set, the
    dataset also does not contain their dependents.

    The method repairs every surviving cross-reference. It removes deleted records from task
    inputs and removes task IDs that no longer resolve from surviving records. A task references
    each of its Records: its inputs, candidate Records, and Record targets, and the Records that
    hold its Signal targets and the annotations it references. It becomes invalid if the edit
    removes all of its inputs, an object used as a target or candidate, a Signal that one of its
    spans names, an annotation that holds its answer, or a task from which it was derived. An
    input annotation on a removed Record is context, so the task keeps going without it. If
    ``cascade`` is set, the method removes an invalid task. If ``cascade`` is not set, the method
    rejects the edit, so a committed version never contains a dangling reference.

    The new dataset keeps copies of the parent cards and the lineage, and of the import map without
    the entries of the removed records, their Signals, and the removed tasks, so a later import
    into it extends the copies it holds and copies a removed one again. Its Records and Sources are
    copies, which such an import extends while ``dataset``'s stay as they are. Its Signals are
    ``dataset``'s own objects, though: annotating one changes it in both datasets, and so does a
    task import that attaches again an annotation removed from a Signal's copy.

    Args:
        dataset: The base dataset. This is typically read back from a committed version.
        record_ids: The record ids to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If ``record_ids`` is a string, an id is unknown, or a required reference
            dangles and ``cascade`` is off.
    """
    remove = _ids("record_ids", record_ids)
    known = {record.record_id for record in dataset.records}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown record ids: {sorted(unknown)}")

    records = [record for record in dataset.records if record.record_id not in remove]
    reachable = _reachable(dataset, records)
    removed_signal_ids = {
        signal.id for record in dataset.records if record.record_id in remove for signal in record.signals
    }
    invalid = {
        task.id for task in dataset.tasks if _task_invalidated(task, remove, removed_signal_ids, reachable[task.id])
    }
    removed_task_ids = _without_dependents(dataset, invalid, f"removing {sorted(remove)}", cascade=cascade)

    copies = [_copy_record(record, removed_task_ids) for record in records]
    tasks = _rebuild_tasks(dataset, removed_task_ids, {record.id: record for record in copies}, reachable)
    removed = {ObjectKind.RECORD: remove, ObjectKind.SIGNAL: removed_signal_ids, ObjectKind.TASK: removed_task_ids}
    return _edited(dataset, copies, tasks, removed)


def remove_signals(dataset: TimeFDataset, signal_ids: Iterable[str], *, cascade: bool = False) -> TimeFDataset:
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
    the entries of the removed Signals and tasks. Its Records and Sources are copies, and the
    Signals it keeps are ``dataset``'s own objects, as with :func:`remove_records`.

    Args:
        dataset: The base dataset.
        signal_ids: The IDs of the Signals to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If ``signal_ids`` is a string, an ID names no Signal, or the edit invalidates
            a task and ``cascade`` is off.
    """
    remove = _ids("signal_ids", signal_ids)
    known = {signal.id for record in dataset.records for signal in record.signals}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown signal ids: {sorted(unknown)}")

    copies = [_copy_record(record, set(), remove) for record in dataset.records]
    kept = {record.id: tuple(record.signals) for record in copies}
    reachable = _reachable(dataset, copies)
    invalid = {task.id for task in dataset.tasks if _loses_signals(task, remove, kept, reachable[task.id])}
    removed_task_ids = _without_dependents(dataset, invalid, f"removing signals {sorted(remove)}", cascade=cascade)

    for record in copies:
        record.task_ids = tuple(task_id for task_id in record.task_ids if task_id not in removed_task_ids)
    tasks = _rebuild_tasks(dataset, removed_task_ids, {record.id: record for record in copies}, reachable)
    return _edited(dataset, copies, tasks, {ObjectKind.SIGNAL: remove, ObjectKind.TASK: removed_task_ids})


def _ids(name: str, ids: Iterable[str]) -> set[str]:
    """Return the IDs an edit is given, each once.

    Raises:
        TimeFEditError: If ``ids`` is a string, which would name one ID per character.
    """
    if isinstance(ids, str):
        raise TimeFEditError(f"{name} takes a collection of IDs, not the string {ids!r}")
    return set(ids)


def _edited(
    dataset: TimeFDataset,
    records: list[Record],
    tasks: list[Task],
    removed: Mapping[ObjectKind, Container[str]],
) -> TimeFDataset:
    """Build an edit's dataset, with the import state of ``dataset`` and its schema re-derived.

    Args:
        dataset: The base dataset.
        records: The edit's Records.
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
        annotations=dataset.annotations,
    )
    edited.parent_cards = dict(dataset.parent_cards)
    edited.lineage = dict(dataset.lineage)
    edited._imports = {
        key: child_id for key, child_id in dataset._imports.items() if child_id not in removed.get(key[1], ())
    }
    edited.derive_schema()
    return edited


def _occurrence_ids(annotations: Iterable[Annotation]) -> frozenset[str]:
    """Return the occurrence IDs of the annotations that have one, which tasks reference them by."""
    return frozenset(annotation.occurrence_id for annotation in annotations if annotation.occurrence_id is not None)


def _reachable(dataset: TimeFDataset, records: Iterable[Record]) -> dict[str, set[str]]:
    """Return the annotation occurrences each task can still resolve after an edit, by task ID.

    These are the occurrences on the dataset, and those that the edit's Records hold among each
    task's Records (see ``TimeFDataset._task_records``), judged on ``dataset``.

    Args:
        dataset: The base dataset.
        records: The Records the edit keeps, with the annotations it keeps.

    Returns:
        The occurrence IDs, by task ID.
    """
    held = {record.id: _occurrence_ids(record.walk_annotations()) for record in records}
    on_dataset = _occurrence_ids(dataset.annotations)
    index = _TaskValidationIndex(dataset)
    reachable: dict[str, set[str]] = {}
    for task in dataset.tasks:
        found = set(on_dataset)
        for record in TimeFDataset._task_records(task, index):
            found |= held.get(record.id, frozenset())
        reachable[task.id] = found
    return reachable


def _copy_record(record: Record, removed_task_ids: Container[str], removed: Container[str] = frozenset()) -> Record:
    """Return a copy of a Record without the given Signals and tasks, and the annotations only those Signals placed.

    Args:
        record: The Record to copy.
        removed_task_ids: The IDs of the removed tasks, which leave ``task_ids``.
        removed: The IDs of the Signals to leave out.

    Returns:
        The copy, whose Sources are copies too.
    """
    return replace(
        record,
        sources=tuple(_copy_source(source, removed) for source in record.sources),
        annotations=_still_placed(record, removed, record.time_span),
        task_ids=tuple(task_id for task_id in record.task_ids if task_id not in removed_task_ids),
    )


def _copy_source(source: Source, removed: Container[str] = frozenset()) -> Source:
    """Return a copy of a Source, and of the Sources below it, without the given Signals.

    The copy holds the Signals it keeps, not copies of them, and has its own metadata. It loses
    the annotations that only the removed Signals placed. An import extends a Source's copy in
    place, which leaves the original as it is.

    Returns:
        The copy.
    """
    duplicate = copy(source)
    duplicate.sources = tuple(_copy_source(child, removed) for child in source.sources)
    duplicate.signals = tuple(signal for signal in source.signals if signal.id not in removed)
    duplicate.annotations = _still_placed(source, removed, None)
    duplicate.metadata = deepcopy(source.metadata)
    return duplicate


def _still_placed(
    owner: Record | Source, removed: Container[str], time_span: TimeInterval | None
) -> tuple[Annotation, ...]:
    """Return the annotations of a Record or Source that the Signals it keeps place.

    The rule is the one an import attaches annotations by. While the owner keeps each of its
    Signals, every annotation stays. Otherwise an annotation stays if it has no span, or if its
    span lies inside the windows of the Signals the owner keeps or inside ``time_span``.

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
    return tuple(annotation for annotation in owner.annotations if _fits(annotation, kept, owner.id, time_span))


def _names_one_of(span: Span, signal_ids: Container[str]) -> bool:
    """Return whether a span names one of the Signals."""
    if isinstance(span, StepSpan):
        return span.time_series_id in signal_ids
    return isinstance(span, TimeSpan) and any(identity in signal_ids for identity in span.time_series_ids or ())


def _loses_signals(task: Task, removed: set[str], kept: Mapping[str, tuple[Signal, ...]], reachable: set[str]) -> bool:
    """Return whether removing the Signals ``removed`` invalidates ``task``.

    It does if a Signal target is removed, or a span names a removed Signal; if an input Record is
    left without Signals; if a span that names no Signal no longer lies where the kept Signals of
    its input place it; or if an answer annotation is dropped.

    Args:
        task: The task.
        removed: The IDs of the removed Signals.
        kept: The Signals each Record keeps, by Record ID.
        reachable: The annotation occurrences the task can still resolve.
    """
    if any(signal.id in removed for signal in TimeFDataset._task_signals(task)):
        return True
    if any(_names_one_of(span, removed) for span in task.spans()):
        return True
    for record in task.inputs:
        signals = kept[record.id]
        if len(signals) == len(record.signals):
            continue
        if not signals:
            return True
        for span in task.spans():
            if isinstance(span, TimeSpan) and span.time_series_ids is None and not _places(span, signals, record):
                return True
    return any(annotation.occurrence_id not in reachable for annotation in task.target_annotations)


def _places(span: TimeSpan, signals: tuple[Signal, ...], record: Record) -> bool:
    """Return whether a span that names no Signal lies inside the windows of the Signals, or of a Record's time span."""
    try:
        check_span_within_window("task span", span, signals, record.id, record.time_span, warn_when_outside=False)
    except TimeFValidationError:
        return False
    return True


def _without_dependents(dataset: TimeFDataset, invalid: set[str], edit: str, *, cascade: bool) -> set[str]:
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


def _task_invalidated(task: Task, remove: set[str], removed_signal_ids: set[str], reachable: set[str]) -> bool:
    """Return whether removing ``remove`` strips a required reference from ``task``.

    A Record or Signal used as a target is required by construction, and so is a Signal that a
    span names. Candidate records are also required because silently changing the candidate pool
    changes the question. Target annotations are the answer and therefore cannot be stripped.
    Input annotations are context, so :func:`_rebuild_tasks` removes the ones that are no longer
    reachable.
    """
    targets = task.targets or ()
    target_record_ids = {target.id for target in targets if isinstance(target, Record)}
    target_signal_ids = {target.id for target in targets if isinstance(target, Signal)}
    candidate_record_ids = {record.id for record in getattr(task, "candidate_records", ())}
    if (target_record_ids | candidate_record_ids) & remove:
        return True
    if target_signal_ids & removed_signal_ids:
        return True
    if any(_names_one_of(span, removed_signal_ids) for span in task.spans()):
        return True
    if task.inputs and all(record.id in remove for record in task.inputs):
        return True
    return any(annotation.occurrence_id not in reachable for annotation in task.target_annotations)


def _rebuild_tasks(
    dataset: TimeFDataset,
    removed_task_ids: set[str],
    records_by_id: Mapping[str, "Record"],
    reachable: Mapping[str, set[str]],
) -> list[Task]:
    """Rebuild the surviving tasks. Strip out unreachable record, annotation, and from-task references.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks to drop.
        records_by_id: The replacement Record objects, keyed by ID.
        reachable: The annotation occurrence IDs each task can still resolve, by task ID.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to the rebuilt instances.
    """
    rebuilt: dict[str, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        inputs = tuple(records_by_id[record.id] for record in task.inputs if record.id in records_by_id)
        input_annotations = tuple(
            annotation for annotation in task.input_annotations if annotation.occurrence_id in reachable[task.id]
        )
        changes: dict[str, object] = {
            "inputs": inputs,
            "targets": tuple(
                records_by_id.get(target.id, target) if isinstance(target, Record) else target
                for target in task.targets or ()
            )
            if task.targets is not None
            else None,
            "input_annotations": input_annotations,
            "from_tasks": (),
        }
        if hasattr(task, "candidate_records"):
            candidates = cast("tuple[Record, ...]", task.candidate_records)
            changes["candidate_records"] = tuple(records_by_id[record.id] for record in candidates)
        rebuilt[task.id] = replace(
            task,
            **changes,
        )
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
    remove_record_ids: Iterable[str] = (),
    remove_signal_ids: Iterable[str] = (),
    cascade: bool = False,
    **writer_kwargs: object,
) -> Path:
    """Read a committed version, remove records or Signals, and write the derived version next to it.

    The derived version reuses the base's payload files, so it is written into the base's registry
    as ``<root>/<id>/<dataset_version>/``, where readers find the base. To derive a version in
    another registry, copy the base there first, for example with ``download_version``, then edit
    the copy.

    The derived version keeps the base's parent cards, lineage, and import map, without the
    entries of what it removes, so ``get(..., alias=)`` finds the copies it holds and a later
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
    record_ids = _ids("remove_record_ids", remove_record_ids)
    signal_ids = _ids("remove_signal_ids", remove_signal_ids)
    with TimeFReader(open_local_version(base_dir)) as reader:
        base_version = str(reader.metadata.dataset_version)
        if str(dataset_version) == base_version:
            raise TimeFEditError(f"derived version {dataset_version} must differ from the base version {base_version}")

        edited = reader.read()
        writer_kwargs.setdefault("values_backend", reader.values_backend)  # an edit keeps the base dataset's backend
        signal_ids -= {
            signal.id for record in edited.records if record.record_id in record_ids for signal in record.signals
        }
        if signal_ids:
            edited = remove_signals(edited, signal_ids, cascade=cascade)
        edited = remove_records(edited, record_ids, cascade=cascade)
        edited._metadata = edited.metadata.model_copy(update={"dataset_version": dataset_version})

        with TimeFWriter(root, edited, **writer_kwargs) as writer:  # ty: ignore[invalid-argument-type]
            writer.write()
    return release_dir(root, DatasetRef(dataset_id=edited.metadata.dataset_id, version=dataset_version))
