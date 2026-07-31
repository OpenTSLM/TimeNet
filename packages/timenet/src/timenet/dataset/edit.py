"""Copy-on-write edits: derive a new immutable dataset version from an existing one.

TimeF versions are immutable, so "removing a row" means writing a *new* version with the row gone. The
cheap path is copy-on-write: read the base version into memory (values stay lazy, pulled from the base
shards), drop the samples, repair every cross-reference, and write a fresh version through the normal
atomic-commit writer. Ids are stable and never reused, so surviving references stay valid without any
renumbering. On a content-addressed / deduplicating backend the rewrite only re-stores the chunks that
actually changed.
"""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from timenet.dataset.dataset import TimeFDataset
from timenet.errors import TimeFEditError
from timenet.reader import TimeFReader
from timenet.types import DatasetSchema, Task, Version
from timenet.writer import TimeFWriter


def remove_samples(dataset: TimeFDataset, sample_ids: Iterable[str], *, cascade: bool = False) -> TimeFDataset:
    """Return a new in-memory dataset with ``sample_ids`` (and, with ``cascade``, their dependents) gone.

    Every surviving cross-reference is repaired: a deleted sample id is stripped from each task's
    ``sample_ids``, and each surviving sample drops task ids that no longer resolve. A task that would
    lose a *required* reference (a forecasting ``target_sample_id`` / ``context_sample_ids``, its last
    remaining sample, or a ``from_task`` edge to a removed task) is only removed when ``cascade`` is set;
    otherwise the edit is rejected so a committed version is never left dangling.

    Args:
        dataset: The base dataset (typically read back from a committed version).
        sample_ids: The sample ids to remove.
        cascade: Remove tasks invalidated by the removal (transitively) instead of rejecting the edit.

    Returns:
        A new :class:`TimeFDataset` with the removals applied and its schema re-derived.

    Raises:
        TimeFEditError: If an id is unknown, or a required reference would dangle and ``cascade`` is off.
    """
    remove = set(sample_ids)
    known = {sample.sample_id for sample in dataset.samples}
    unknown = remove - known
    if unknown:
        raise TimeFEditError(f"cannot remove unknown sample ids: {sorted(unknown)}")

    surviving_ids = known - remove
    removed_task_ids = _tasks_to_remove(dataset, remove, cascade=cascade)

    tasks = _rebuild_tasks(dataset, removed_task_ids, surviving_ids)
    surviving_task_ids = {task.id for task in tasks}
    samples = [
        replace(sample, task_ids=tuple(tid for tid in sample.task_ids if tid in surviving_task_ids))
        for sample in dataset.samples
        if sample.sample_id not in remove
    ]

    # An empty schema, not the base's: the derive_schema() below overwrites it anyway, and
    # `dataset.schema or dataset.derive_schema()` mutated the *input* dataset's cached schema as a
    # side effect. Re-deriving from the survivors also drops spec / annotation / task types that
    # existed only on a removed sample.
    edited = TimeFDataset.from_parts(metadata=dataset.metadata, samples=samples, tasks=tasks, schema=DatasetSchema())
    edited.derive_schema()
    return edited


def _tasks_to_remove(dataset: TimeFDataset, remove: set[str], *, cascade: bool) -> set[str]:
    """Return the ids of tasks invalidated by the removal, honoring the reject-unless-cascade rule.

    Args:
        dataset: The base dataset.
        remove: The sample ids being removed.
        cascade: Remove invalidated tasks (transitively) instead of rejecting.

    Returns:
        The ids of the tasks to drop.

    Raises:
        TimeFEditError: If a task would dangle and ``cascade`` is off.
    """
    invalid = {task.id for task in dataset.tasks if _task_invalidated(task, remove)}
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


def _task_invalidated(task: Task, remove: set[str]) -> bool:
    """Return whether removing ``remove`` strips a required reference from ``task``.

    A payload sample reference is required by construction — a forecast without its horizon, an edit
    without its source, a correspondence without its candidate pool is not a task any more — so any task
    class declaring one in :class:`~timenet.types.TaskRefs` is invalidated when that sample goes.
    """
    for name in type(task).refs.sample_id_fields:
        value = getattr(task, name)
        referenced = (value,) if isinstance(value, str) else tuple(value or ())
        if any(sample_id in remove for sample_id in referenced):
            return True
    return bool(task.sample_ids) and all(sid in remove for sid in task.sample_ids)


def _rebuild_tasks(dataset: TimeFDataset, removed_task_ids: set[str], surviving_ids: set[str]) -> list[Task]:
    """Rebuild the surviving tasks with removed sample ids and from-task edges stripped out.

    Args:
        dataset: The base dataset.
        removed_task_ids: The ids of the tasks being dropped.
        surviving_ids: The sample ids that remain.

    Returns:
        The rebuilt surviving tasks, with ``from_tasks`` rewired to rebuilt instances.
    """
    rebuilt: dict[str, Task] = {}
    for task in dataset.tasks:
        if task.id in removed_task_ids:
            continue
        rebuilt[task.id] = replace(
            task,
            sample_ids=tuple(sid for sid in task.sample_ids if sid in surviving_ids),
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
        out_root: Parent directory the new version is written under (``<out_root>/<id>/<version>/``).
        dataset_version: The semantic version for the derived dataset (must differ from the base's).
        remove_sample_ids: Sample ids to remove.
        cascade: Remove tasks invalidated by the removal instead of rejecting the edit.
        **writer_kwargs: Extra keyword arguments forwarded to :class:`~timenet.writer.TimeFWriter`.

    Returns:
        The new version directory.

    Raises:
        TimeFEditError: If ``dataset_version`` matches the base version, or a required reference would
            dangle and ``cascade`` is off.
    """
    # The whole write happens inside the reader's context. The edited dataset's series keep lazy
    # loaders that pull values from the base version's shards, and those are consumed during
    # writer.write(); letting the reader's `with` exit first meant writing through a closed reader,
    # which only worked because close() silently reopened the shards and leaked the handles.
    with TimeFReader(base_dir) as reader:
        base_version = str(reader.metadata.dataset_version)
        if str(dataset_version) == base_version:
            raise TimeFEditError(f"derived version {dataset_version} must differ from the base version {base_version}")

        base = reader.read()
        writer_kwargs.setdefault("values_backend", reader.values_backend)  # an edit keeps the base's backend
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
