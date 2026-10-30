"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable. To remove a row, you must write a new version without that row.
The cheap method is copy-on-write. This method reads the base version into memory. The values
stay lazy and come from the base shards. The method removes the records, repairs each
cross-reference, and writes a new version with the normal atomic-commit writer. Ids are stable
and the system never reuses them, so surviving references stay valid without renumbering. On a
content-addressed or deduplicating backend, the rewrite stores only the chunks that changed.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from timenet.dataset.dataset import TimeFDataset
from timenet.dataset.record import Record
from timenet.errors import TimeFEditError
from timenet.reader import TimeFReader
from timenet.registry._paths import release_dir, split_release_dir
from timenet.registry.dependencies import open_local_version
from timenet.types import Annotation, DatasetSchema, EntityId, Task, Version
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

    Args:
        dataset: The base dataset. This is typically read back from a committed version.
        record_ids: The record ids to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If an id is unknown, or a required reference dangles and ``cascade`` is off.
    """
    remove = set(record_ids)
    known = {record.id for record in dataset.records}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown record ids: {sorted(unknown)}")

    reachable = _reachability(dataset, (record for record in dataset.records if record.id not in remove))
    removed_signal_ids = {signal.id for record in dataset.records if record.id in remove for signal in record.signals}
    removed_task_ids = _tasks_to_remove(dataset, remove, removed_signal_ids, reachable, cascade=cascade)

    records = [record for record in dataset.records if record.id not in remove]
    records_by_id = {record.id: record for record in records}
    tasks = _rebuild_tasks(dataset, removed_task_ids, records_by_id, reachable)

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
        last_ids=dataset.last_ids,
    )
    edited.derive_schema()
    return edited


def _tasks_to_remove(
    dataset: TimeFDataset,
    remove: set[EntityId],
    removed_signal_ids: set[EntityId],
    reachable: Callable[[Task, Annotation], bool],
    *,
    cascade: bool,
) -> set[EntityId]:
    """Return the ids of tasks that the removal invalidates.

    The method follows the reject-unless-cascade rule.

    Args:
        dataset: The base dataset.
        remove: The record IDs to remove.
        removed_signal_ids: The signal IDs owned by the removed records.
        reachable: Whether a task can still resolve an annotation that it references.
        cascade: If set, remove invalidated tasks (transitively) instead of rejecting the edit.

    Returns:
        The ids of the tasks to drop.

    Raises:
        TimeFEditError: If a task dangles and ``cascade`` is off.
    """
    invalid = {task.id for task in dataset.tasks if _task_invalidated(task, remove, removed_signal_ids, reachable)}
    if invalid and not cascade:
        raise TimeFEditError(
            f"removing {sorted(remove)} would dangle tasks {sorted(invalid)}; pass cascade=True to remove them"
        )
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


def _reachability(dataset: TimeFDataset, records: Iterable[Record]) -> Callable[[Task, Annotation], bool]:
    """Return a check of whether a task can still resolve an annotation that it references after an edit.

    The check builds no set per task. It finds the Record of an annotation in a map that the call
    builds once.

    Args:
        dataset: The base dataset. The check takes from it the occurrences on the dataset itself, and
            the Records that each task references (see ``TimeFDataset._referenced_records``).
        records: The Records that the edit keeps, with the annotations that it keeps.

    Returns:
        The check. It takes a task and an annotation that the task references. It returns whether
        the annotation lies on the dataset itself, or below a kept Record that the task references.
    """
    record_ids = {id(annotation): record.id for record in records for annotation in record.walk_annotations()}
    on_dataset = frozenset(id(annotation) for annotation in dataset.annotations)

    def reachable(task: Task, annotation: Annotation) -> bool:
        if id(annotation) in on_dataset:
            return True
        record_id = record_ids.get(id(annotation))
        return record_id is not None and any(record.id == record_id for record in dataset._referenced_records(task))

    return reachable


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
) -> list[Task]:
    """Rebuild the surviving tasks. Strip out unreachable record, annotation, and from-task references.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks to drop.
        records_by_id: The surviving Record objects, keyed by id.
        reachable: Whether a task can still resolve an annotation that it references.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to the rebuilt instances.
    """
    rebuilt: dict[EntityId, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        inputs = tuple(records_by_id[record.id] for record in task.inputs if record.id in records_by_id)
        input_annotations = tuple(annotation for annotation in task.input_annotations if reachable(task, annotation))
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
        copy = replace(task, **changes)
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
    cascade: bool = False,
    **writer_kwargs: object,
) -> Path:
    """Read a committed version, remove records, and write the derived version next to it.

    The derived version reuses the base's payload files, so it is written into the base's registry
    as ``<root>/<id>/<dataset_version>/``, where readers find the base. To derive a version in
    another registry, copy the base there first, for example with ``download_version``, then edit
    the copy.

    Args:
        base_dir: The committed version directory ``<root>/<org>/<name>/<version>`` to derive from.
        dataset_version: The semantic version for the derived dataset. This must differ from
            the base version.
        remove_record_ids: The record ids to remove.
        cascade: If set, remove tasks that the removal invalidates instead of rejecting the edit.
        **writer_kwargs: Extra keyword arguments forwarded to :class:`~timenet.writer.TimeFWriter`.

    Returns:
        The new version directory.

    Raises:
        TimeFEditError: If ``dataset_version`` matches the base version, or a required reference
            dangles and ``cascade`` is off.
    """
    # The whole write happens inside the reader's context. The edited dataset's series keep lazy
    # loaders that pull values from the base version's shards. writer.write() consumes those
    # values. If the reader's `with` block exits first, the write happens through a closed
    # reader. That only worked because close() silently reopened the shards and leaked the
    # handles.
    root, _ = split_release_dir(base_dir)
    with TimeFReader(open_local_version(base_dir)) as reader:
        base_version = str(reader.metadata.dataset_version)
        if str(dataset_version) == base_version:
            raise TimeFEditError(f"derived version {dataset_version} must differ from the base version {base_version}")

        base = reader.read()
        writer_kwargs.setdefault("values_backend", reader.values_backend)  # an edit keeps the base dataset's backend
        edited = remove_records(base, remove_record_ids, cascade=cascade)
        edited = TimeFDataset.from_parts(
            metadata=edited.metadata.model_copy(update={"dataset_version": dataset_version}),
            records=edited.records,
            tasks=edited.tasks,
            schema=DatasetSchema(),
            annotations=edited.annotations,
            last_ids=edited.last_ids,
        )
        edited.derive_schema()

        with TimeFWriter(root, edited, **writer_kwargs) as writer:  # ty: ignore[invalid-argument-type]
            writer.write()
    return release_dir(root, DatasetRef(dataset_id=edited.metadata.dataset_id, version=dataset_version))
