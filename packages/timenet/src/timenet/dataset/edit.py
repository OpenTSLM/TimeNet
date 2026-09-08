"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable. To remove a row, you must write a new version without that row.
The cheap method is copy-on-write. This method reads the base version into memory. The values
stay lazy and come from the base shards. The method removes the records, repairs each
cross-reference, and writes a new version with the normal atomic-commit writer. Ids are stable
and the system never reuses them, so surviving references stay valid without renumbering. On a
content-addressed or deduplicating backend, the rewrite stores only the chunks that changed.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from timenet.dataset.dataset import TimeFDataset
from timenet.errors import TimeFEditError
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetSchema, Task, Version
from timenet.writer import TimeFWriter


def remove_records(dataset: TimeFDataset, record_ids: Iterable[str], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset without the given records.

    The dataset does not contain the records in ``record_ids``. If ``cascade`` is set, the
    dataset also does not contain their dependents.

    The method repairs every surviving cross-reference. It strips each removed record id from
    every task's ``record_ids``. It removes a task id from each surviving record when the task
    id no longer resolves. It removes ``input_annotation_ids`` from each task when the task's
    surviving records no longer carry them. A task can lose a required reference: a forecasting
    ``target_record_id`` or ``context_record_ids``, its last surviving record, or an annotation
    that holds its answer. A task can also lose a required reference through a ``from_task``
    edge to a removed task. If ``cascade`` is set, the method removes such a task. If ``cascade``
    is not set, the method rejects the edit, so a committed version never dangles.

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
    known = {record.record_id for record in dataset.records}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown record ids: {sorted(unknown)}")

    surviving_ids = known - remove
    annotations_by_record = {
        record.record_id: frozenset(annotation.id for annotation in record.annotations)
        for record in dataset.records
        if record.record_id not in remove
    }
    # Registered annotations no record carries survive any record removal, and tasks reference them
    # regardless of which records remain, so they are always reachable.
    registered = dataset.registered_annotations
    registered_ids = frozenset(annotation.id for annotation in registered)
    removed_task_ids = _tasks_to_remove(dataset, remove, annotations_by_record, registered_ids, cascade=cascade)

    tasks = _rebuild_tasks(dataset, removed_task_ids, surviving_ids, annotations_by_record, registered_ids)
    surviving_task_ids = {task.id for task in tasks}
    records = [
        replace(record, task_ids=tuple(tid for tid in record.task_ids if tid in surviving_task_ids))
        for record in dataset.records
        if record.record_id not in remove
    ]

    # Use an empty schema, not the base dataset's schema. The derive_schema() call below
    # overwrites it anyway, and `dataset.schema or dataset.derive_schema()` mutated the input
    # dataset's cached schema as a side effect. Re-deriving the schema from the survivors also
    # drops spec, annotation, and task types that existed only on a removed record.
    edited = TimeFDataset.from_parts(
        metadata=dataset.metadata,
        records=records,
        tasks=tasks,
        schema=DatasetSchema(),
        registered_annotations=registered,
    )
    edited.derive_schema()
    return edited


def _tasks_to_remove(
    dataset: TimeFDataset,
    remove: set[str],
    annotations_by_record: Mapping[str, frozenset[str]],
    registered_ids: frozenset[str],
    *,
    cascade: bool,
) -> set[str]:
    """Return the ids of tasks that the removal invalidates.

    The method follows the reject-unless-cascade rule.

    Args:
        dataset: The base dataset.
        remove: The record ids to remove.
        annotations_by_record: Annotation ids carried by each surviving record.
        cascade: If set, remove invalidated tasks (transitively) instead of rejecting the edit.

    Returns:
        The ids of the tasks to drop.

    Raises:
        TimeFEditError: If a task dangles and ``cascade`` is off.
    """
    invalid = {
        task.id for task in dataset.tasks if _task_invalidated(task, remove, annotations_by_record, registered_ids)
    }
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


def _reachable_annotation_ids(
    task: Task, annotations_by_record: Mapping[str, frozenset[str]], registered_ids: frozenset[str]
) -> set[str]:
    """Return the annotation ids that ``task`` can still resolve.

    These are the ids on the records the task keeps, plus every registered annotation (which no record
    carries, so a record removal never strips it).

    :meth:`~timenet.dataset.TimeFDataset.add_task` validates a task's annotation references
    against its own records, not against the whole dataset. So an edit must repair the
    references in that same frame. A dataset-wide check is weaker: an annotation shared with a
    record outside the task survives the removal. But the task can no longer reach it through
    any of its own records.

    Args:
        task: The task whose references the method resolves.
        annotations_by_record: Annotation ids carried by each surviving record. A removed
            record is absent, so it contributes no ids.

    Returns:
        The annotation ids still reachable from the task's surviving records.
    """
    reachable: set[str] = set(registered_ids)
    for record_id in task.record_ids:
        reachable |= annotations_by_record.get(record_id, frozenset())
    return reachable


def _task_invalidated(
    task: Task, remove: set[str], annotations_by_record: Mapping[str, frozenset[str]], registered_ids: frozenset[str]
) -> bool:
    """Return whether removing ``remove`` strips a required reference from ``task``.

    A payload record reference is required by construction. For example, a forecast needs its
    horizon, an edit needs its source, and a correspondence needs its candidate pool. Without
    one of these, the object is no longer a task. So the method invalidates any task class that
    declares such a reference in :class:`~timenet.types.TaskRefs` when the referenced record is
    removed.

    ``target_annotation_ids`` is required for the same reason: it is the answer. An unreachable
    entry does more than dangle: it rewrites the ground truth. For example, a localization task
    that loses one of two target regions still reads back as a complete answer.
    ``input_annotation_ids`` is context given to the model, not the answer. So
    :func:`_rebuild_tasks` strips unreachable entries from it instead, the same way it already
    strips removed ``record_ids`` and ``from_tasks`` edges.
    """
    for name in type(task).refs.record_id_fields:
        value = getattr(task, name)
        referenced = (value,) if isinstance(value, str) else tuple(value or ())
        if any(record_id in remove for record_id in referenced):
            return True
    if task.record_ids and all(sid in remove for sid in task.record_ids):
        return True
    reachable = _reachable_annotation_ids(task, annotations_by_record, registered_ids)
    return any(annotation_id not in reachable for annotation_id in task.target_annotation_ids)


def _rebuild_tasks(
    dataset: TimeFDataset,
    removed_task_ids: set[str],
    surviving_ids: set[str],
    annotations_by_record: Mapping[str, frozenset[str]],
    registered_ids: frozenset[str],
) -> list[Task]:
    """Rebuild the surviving tasks. Strip out unreachable record, annotation, and from-task references.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks to drop.
        surviving_ids: The record ids that remain.
        annotations_by_record: Annotation ids carried by each surviving record.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to the rebuilt instances.
    """
    rebuilt: dict[str, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        reachable = _reachable_annotation_ids(task, annotations_by_record, registered_ids)
        rebuilt[task.id] = replace(
            task,
            record_ids=tuple(sid for sid in task.record_ids if sid in surviving_ids),
            input_annotation_ids=tuple(aid for aid in task.input_annotation_ids if aid in reachable),
            from_tasks=(),
        )
    for task in dataset.tasks:
        if task.id in rebuilt:
            rebuilt[task.id].from_tasks = tuple(
                rebuilt[parent.id] for parent in task.from_tasks if parent.id in rebuilt
            )
    return list(rebuilt.values())


def edit_version(
    base_dir: Path,
    out_root: Path,
    *,
    dataset_version: Version,
    remove_record_ids: Iterable[str] = (),
    cascade: bool = False,
    **writer_kwargs: object,
) -> Path:
    """Read a committed version, remove records, and publish a new version (copy-on-write).

    Args:
        base_dir: The committed version directory to derive from.
        out_root: The parent directory for the new version (``<out_root>/<id>/<version>/``).
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
    with TimeFReader(DatasetVersion.open_local(base_dir)) as reader:
        base_version = str(reader.metadata.dataset_version)
        if str(dataset_version) == base_version:
            raise TimeFEditError(f"derived version {dataset_version} must differ from the base version {base_version}")

        base = reader.read()
        writer_kwargs.setdefault("values_backend", reader.values_backend)  # an edit keeps the base dataset's backend
        edited = remove_records(base, remove_record_ids, cascade=cascade)
        edited = TimeFDataset.from_parts(
            metadata=replace(edited.metadata, dataset_version=dataset_version),
            records=edited.records,
            tasks=edited.tasks,
            schema=DatasetSchema(),
            registered_annotations=edited.registered_annotations,
        )
        edited.derive_schema()

        with TimeFWriter(out_root, edited, **writer_kwargs) as writer:  # ty: ignore[invalid-argument-type]
            writer.write()
    return out_root / edited.metadata.dataset_id / str(dataset_version)
