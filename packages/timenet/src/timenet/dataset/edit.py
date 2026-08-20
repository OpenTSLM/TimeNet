"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable. To remove a row, you must write a new version without that row.
The cheap method is copy-on-write. This method reads the base version into memory. The values
stay lazy and come from the base shards. The method removes the samples, repairs each
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


def remove_samples(dataset: TimeFDataset, sample_ids: Iterable[str], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset without the given samples.

    The dataset does not contain the samples in ``sample_ids``. If ``cascade`` is set, the
    dataset also does not contain their dependents.

    The method repairs every surviving cross-reference. It strips each removed sample id from
    every task's ``sample_ids``. It removes a task id from each surviving sample when the task
    id no longer resolves. It removes ``input_annotation_ids`` from each task when the task's
    surviving samples no longer carry them. A task can lose a required reference: a forecasting
    ``target_sample_id`` or ``context_sample_ids``, its last surviving sample, or an annotation
    that holds its answer. A task can also lose a required reference through a ``from_task``
    edge to a removed task. If ``cascade`` is set, the method removes such a task. If ``cascade``
    is not set, the method rejects the edit, so a committed version never dangles.

    Args:
        dataset: The base dataset. This is typically read back from a committed version.
        sample_ids: The sample ids to remove.
        cascade: If set, remove tasks that the removal invalidates (transitively) instead of
            rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If an id is unknown, or a required reference dangles and ``cascade`` is off.
    """
    remove = set(sample_ids)
    known = {sample.sample_id for sample in dataset.samples}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown sample ids: {sorted(unknown)}")

    surviving_ids = known - remove
    annotations_by_sample = {
        sample.sample_id: frozenset(annotation.id for annotation in sample.annotations)
        for sample in dataset.samples
        if sample.sample_id not in remove
    }
    removed_task_ids = _tasks_to_remove(dataset, remove, annotations_by_sample, cascade=cascade)

    tasks = _rebuild_tasks(dataset, removed_task_ids, surviving_ids, annotations_by_sample)
    surviving_task_ids = {task.id for task in tasks}
    samples = [
        replace(sample, task_ids=tuple(tid for tid in sample.task_ids if tid in surviving_task_ids))
        for sample in dataset.samples
        if sample.sample_id not in remove
    ]

    # Use an empty schema, not the base dataset's schema. The derive_schema() call below
    # overwrites it anyway, and `dataset.schema or dataset.derive_schema()` mutated the input
    # dataset's cached schema as a side effect. Re-deriving the schema from the survivors also
    # drops spec, annotation, and task types that existed only on a removed sample.
    edited = TimeFDataset.from_parts(metadata=dataset.metadata, samples=samples, tasks=tasks, schema=DatasetSchema())
    edited.derive_schema()
    return edited


def _tasks_to_remove(
    dataset: TimeFDataset,
    remove: set[str],
    annotations_by_sample: Mapping[str, frozenset[str]],
    *,
    cascade: bool,
) -> set[str]:
    """Return the ids of tasks that the removal invalidates.

    The method follows the reject-unless-cascade rule.

    Args:
        dataset: The base dataset.
        remove: The sample ids to remove.
        annotations_by_sample: Annotation ids carried by each surviving sample.
        cascade: If set, remove invalidated tasks (transitively) instead of rejecting the edit.

    Returns:
        The ids of the tasks to drop.

    Raises:
        TimeFEditError: If a task dangles and ``cascade`` is off.
    """
    invalid = {task.id for task in dataset.tasks if _task_invalidated(task, remove, annotations_by_sample)}
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


def _reachable_annotation_ids(task: Task, annotations_by_sample: Mapping[str, frozenset[str]]) -> set[str]:
    """Return the annotation ids that ``task`` can still resolve.

    These are the ids on the samples the task keeps.

    :meth:`~timenet.dataset.TimeFDataset.add_task` validates a task's annotation references
    against its own samples, not against the whole dataset. So an edit must repair the
    references in that same frame. A dataset-wide check is weaker: an annotation shared with a
    sample outside the task survives the removal. But the task can no longer reach it through
    any of its own samples.

    Args:
        task: The task whose references the method resolves.
        annotations_by_sample: Annotation ids carried by each surviving sample. A removed
            sample is absent, so it contributes no ids.

    Returns:
        The annotation ids still reachable from the task's surviving samples.
    """
    reachable: set[str] = set()
    for sample_id in task.sample_ids:
        reachable |= annotations_by_sample.get(sample_id, frozenset())
    return reachable


def _task_invalidated(task: Task, remove: set[str], annotations_by_sample: Mapping[str, frozenset[str]]) -> bool:
    """Return whether removing ``remove`` strips a required reference from ``task``.

    A payload sample reference is required by construction. For example, a forecast needs its
    horizon, an edit needs its source, and a correspondence needs its candidate pool. Without
    one of these, the object is no longer a task. So the method invalidates any task class that
    declares such a reference in :class:`~timenet.types.TaskRefs` when the referenced sample is
    removed.

    ``target_annotation_ids`` is required for the same reason: it is the answer. An unreachable
    entry does more than dangle: it rewrites the ground truth. For example, a localization task
    that loses one of two target regions still reads back as a complete answer.
    ``input_annotation_ids`` is context given to the model, not the answer. So
    :func:`_rebuild_tasks` strips unreachable entries from it instead, the same way it already
    strips removed ``sample_ids`` and ``from_tasks`` edges.
    """
    for name in type(task).refs.sample_id_fields:
        value = getattr(task, name)
        referenced = (value,) if isinstance(value, str) else tuple(value or ())
        if any(sample_id in remove for sample_id in referenced):
            return True
    if task.sample_ids and all(sid in remove for sid in task.sample_ids):
        return True
    reachable = _reachable_annotation_ids(task, annotations_by_sample)
    return any(annotation_id not in reachable for annotation_id in task.target_annotation_ids)


def _rebuild_tasks(
    dataset: TimeFDataset,
    removed_task_ids: set[str],
    surviving_ids: set[str],
    annotations_by_sample: Mapping[str, frozenset[str]],
) -> list[Task]:
    """Rebuild the surviving tasks. Strip out unreachable sample, annotation, and from-task references.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks to drop.
        surviving_ids: The sample ids that remain.
        annotations_by_sample: Annotation ids carried by each surviving sample.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to the rebuilt instances.
    """
    rebuilt: dict[str, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        reachable = _reachable_annotation_ids(task, annotations_by_sample)
        rebuilt[task.id] = replace(
            task,
            sample_ids=tuple(sid for sid in task.sample_ids if sid in surviving_ids),
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
    remove_sample_ids: Iterable[str] = (),
    cascade: bool = False,
    **writer_kwargs: object,
) -> Path:
    """Read a committed version, remove samples, and publish a new version (copy-on-write).

    Args:
        base_dir: The committed version directory to derive from.
        out_root: The parent directory for the new version (``<out_root>/<id>/<version>/``).
        dataset_version: The semantic version for the derived dataset. This must differ from
            the base version.
        remove_sample_ids: The sample ids to remove.
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
        edited = remove_samples(base, remove_sample_ids, cascade=cascade)
        edited = TimeFDataset.from_parts(
            metadata=replace(edited.metadata, dataset_version=dataset_version),
            samples=edited.samples,
            tasks=edited.tasks,
            schema=DatasetSchema(),
        )
        edited.derive_schema()

        derived_from = {"dataset_version": base_version, "op": "remove_samples"}
        with TimeFWriter(out_root, edited, derived_from=derived_from, **writer_kwargs) as writer:  # ty: ignore[invalid-argument-type]
            writer.write()
    return out_root / edited.metadata.dataset_id / str(dataset_version)
