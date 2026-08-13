"""The :class:`TimeFDataset` in-memory model a connector populates during ``convert()``."""

from collections.abc import Iterable
from datetime import datetime
import sys
from typing import Literal, TextIO, TypeVar, cast, overload

import numpy as np
import pyarrow as pa

from timenet.dataset.describe import describe_text
from timenet.dataset.sample import Sample, check_span_within_window
from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import (
    AnnotationDescriptor,
    DatasetMetadata,
    DatasetSchema,
    Task,
    TimeInterval,
    annotation_type_of,
    value_type_of,
)


T = TypeVar("T")
TTask = TypeVar("TTask", bound=Task)


class TimeFDataset:
    """Holds samples and their tasks as Python objects. No I/O: persistence is the writer's concern."""

    def __init__(self, *, metadata: DatasetMetadata) -> None:
        """Create an empty dataset.

        Args:
            metadata: The dataset's descriptive identity.
        """
        self._metadata = metadata
        self._samples: list[Sample] = []
        self._tasks: list[Task] = []
        self._schema: DatasetSchema | None = None

    def add_sample(
        self,
        *,
        time_series: tuple[TimeSeries, ...],
        subject_ids: tuple[str, ...] = (),
        sample_id: str | None = None,
        start_time: datetime | int | None = None,
        time_span: TimeInterval | None = None,
    ) -> Sample:
        """Create a sample, register it, and return it.

        Args:
            time_series: The logical :class:`TimeSeries` streams the sample uses.
            subject_ids: Subjects this sample belongs to (empty for subject-less domains).
            sample_id: An explicit id (default: an auto-generated uuid4). Pass one for deterministic
                output, e.g. when generating golden fixtures.
            start_time: Wall-clock timestamp that the sample's relative zero refers to: a
                timezone-aware datetime or whole Unix microseconds, or ``None`` when no wall-clock
                reference exists.
            time_span: The session's overall span, if the series have gaps an unscoped span may fall in
                (see :attr:`Sample.time_span`). Must be a whole-sample :class:`~timenet.types.TimeInterval`
                containing every series' window.

        Returns:
            The newly created :class:`Sample`.

        Raises:
            TimeFValidationError: If ``time_series`` is empty, or two series share a
                ``time_series_id``. Ids must be distinct: the writer keys shards by them, and
                :meth:`Sample.add_annotation` and :meth:`add_task` both resolve references against
                them, so a repeat silently collapses two channels into one.
        """
        if not time_series:
            raise TimeFValidationError("add_sample requires a non-empty time_series")
        series_ids = [ts.time_series_id for ts in time_series]
        if len(set(series_ids)) != len(series_ids):
            duplicates = sorted({sid for sid in series_ids if series_ids.count(sid) > 1})
            raise TimeFValidationError(f"add_sample requires distinct time_series_ids, got duplicates {duplicates}")
        if sample_id is None:
            sample = Sample(
                time_series=tuple(time_series),
                subject_ids=tuple(subject_ids),
                start_time=start_time,
                time_span=time_span,
            )
        else:
            sample = Sample(
                sample_id=sample_id,
                time_series=tuple(time_series),
                subject_ids=tuple(subject_ids),
                start_time=start_time,
                time_span=time_span,
            )
        self._samples.append(sample)
        return sample

    def add_task(self, samples: Sample | Iterable[Sample], task: Task) -> Task:
        """Register a task and link it to its samples.

        Every span the task carries — its ``scope`` and, for a
        :class:`~timenet.types.TemporalLocalizationTask`, its target regions — is checked against the
        samples here, where the samples are available to check against. So are the sample and annotation
        ids the task references, and the source tasks in its ``from_tasks``.

        Set ``scope`` and ``from_tasks`` on the task itself: they describe that one task, not the call.

        Args:
            samples: The sample, or samples, the task is attached to.
            task: The task instance (payload, ``scope``, and ``from_tasks`` already set by the caller).

        Returns:
            The registered task (same instance, with ``sample_ids`` populated).

        Raises:
            TimeFValidationError: If ``samples`` is empty; if the task's id is already registered; if a
                task in ``from_tasks`` is neither registered nor the task itself; if a scope-dependent
                payload rule fails (a :class:`~timenet.types.ForecastingTask` ``target_span`` with no
                scope, a frame mismatch, or a context that leaks the target); if the task sets both
                ``target`` and ``target_annotation_ids`` or, when its answer is not a produced series,
                neither; if a span's ``time_series_ids`` does not resolve to a series on every target
                sample or the span falls outside a sample's covered span; or if a referenced sample or
                annotation is not registered in this dataset.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise TimeFValidationError("add_task requires at least one sample")
        return self._register_batch((task,), targets)[0]

    def add_tasks(self, samples: Sample | Iterable[Sample], tasks: Iterable[Task]) -> tuple[Task, ...]:
        """Register several tasks against the same samples, all together or not at all.

        The whole batch is validated before any of it is attached: if one task fails a check, the call
        raises and leaves the dataset and every task in the batch untouched. To keep the tasks before a
        failure attached, loop :meth:`add_task` instead.

        A task may derive from another in the same batch — list it in the deriving task's ``from_tasks`` —
        because the batch is checked as a unit, so the order within ``tasks`` does not matter.

        Args:
            samples: The sample, or samples, the tasks are attached to.
            tasks: The task instances to register. Pass a single one to :meth:`add_task`.

        Returns:
            The registered tasks (the same instances, with ``sample_ids`` populated), in the order given.

        Raises:
            TimeFValidationError: If ``samples`` is empty; if two tasks in the batch share an id or one
                reuses a registered id; if a ``from_tasks`` parent is neither registered nor in the batch,
                or the derivation is cyclic; or if any task fails the checks :meth:`add_task` documents.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise TimeFValidationError("add_tasks requires at least one sample")
        # Drain `tasks` before validating: a connector generator may attach an annotation and then yield
        # a task referencing it, so the annotation must already be on the sample when the refs are checked.
        batch = tuple(tasks)
        return self._register_batch(batch, targets)

    def _register_batch(self, batch: tuple[Task, ...], targets: tuple[Sample, ...]) -> tuple[Task, ...]:
        """Validate a whole batch of tasks and attach it, or attach none of it.

        Shared by :meth:`add_task` and :meth:`add_tasks` so the singular and plural forms cannot drift;
        the singular is the batch of one. Validation is side-effect-free, so the attachment below runs
        only once the whole batch is known good.

        Args:
            batch: The tasks to register together, already drained from the caller's iterable.
            targets: The samples the tasks attach to.

        Returns:
            The registered tasks (the same instances, with ``sample_ids`` populated), in order.

        Raises:
            TimeFValidationError: as documented on :meth:`add_tasks`.
        """  # noqa: DOC502 (raised by _validate_task_batch, not directly here)
        self._validate_task_batch(batch, targets)
        for task in batch:
            task.sample_ids = tuple(sample.sample_id for sample in targets)
            for sample in targets:
                sample.task_ids = (*sample.task_ids, task.id)
            self._tasks.append(task)
        return batch

    def _validate_task_batch(self, batch: tuple[Task, ...], targets: tuple[Sample, ...]) -> None:
        """Run every check the batch must pass, without attaching anything.

        Covers the cross-task invariants a batch makes possible, on top of the per-task checks: ids stay
        unique against the batch and the dataset, each ``from_tasks`` parent is already registered or in
        the batch, and no task derives from itself or closes a cycle.

        Args:
            batch: The tasks being registered together.
            targets: The samples the tasks attach to.

        Raises:
            TimeFValidationError: as documented on :meth:`add_tasks`.
        """
        registered_ids = {task.id for task in self._tasks}
        batch_ids = [task.id for task in batch]
        duplicated = sorted({task_id for task_id in batch_ids if batch_ids.count(task_id) > 1})
        if duplicated:
            raise TimeFValidationError(f"tasks in the batch share an id: {duplicated}")
        reused = sorted(set(batch_ids) & registered_ids)
        if reused:
            raise TimeFValidationError(f"task id(s) already registered in this dataset: {reused}")
        known_task_ids = registered_ids | set(batch_ids)
        for task in batch:
            for parent_id in task.from_task_ids:
                if parent_id == task.id:
                    raise TimeFValidationError(f"{type(task).__name__} {task.id!r} lists itself in from_tasks")
                if parent_id not in known_task_ids:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {task.id!r} derives from task {parent_id!r}, which is not "
                        f"registered in this dataset or part of the batch"
                    )
        self._check_no_derivation_cycle(batch)
        for task in batch:
            task.check_against_scope()
            self._check_task_answer(task)
            self._check_sample_refs(task)
            for sample in targets:
                for span in task.spans():
                    check_span_within_window(
                        f"{type(task).__name__} span", span, sample.time_series, sample.sample_id, sample.time_span
                    )
            self._check_annotation_refs(task, targets)

    @staticmethod
    def _check_no_derivation_cycle(batch: tuple[Task, ...]) -> None:
        """Reject a ``from_tasks`` cycle formed among the batch's own tasks.

        Only batch tasks can close a cycle: a task already in the dataset passed this check when it was
        added and cannot derive from one that did not exist yet. So the walk stays inside the batch,
        following the parents each task shares with it.

        Args:
            batch: The tasks being registered together.

        Raises:
            TimeFValidationError: If the batch's derivations contain a cycle.
        """
        batch_ids = {task.id for task in batch}
        parents = {task.id: {p for p in task.from_task_ids if p in batch_ids} for task in batch}
        # Kahn's algorithm: peel off tasks whose in-batch parents are all resolved; whatever is left
        # after no more can be peeled sits on a cycle.
        resolved: set[str] = set()
        progressed = True
        while progressed:
            progressed = False
            for task_id, deps in parents.items():
                if task_id not in resolved and deps <= resolved:
                    resolved.add(task_id)
                    progressed = True
        unresolved = sorted(set(parents) - resolved)
        if unresolved:
            raise TimeFValidationError(f"tasks in the batch form a cyclic from_tasks derivation: {unresolved}")

    def derive_schema(self) -> DatasetSchema:
        """Walk the dataset's instances and build its :class:`DatasetSchema`.

        Collects the distinct spec, annotation, and task types, stores the result on the
        dataset, and returns it.

        Returns:
            The derived :class:`DatasetSchema`.

        Raises:
            TimeFValidationError: If one spec type or annotation key yields conflicting descriptors
                across samples.
        """
        specs = self._ordered_unique(ts.spec for sample in self._samples for ts in sample.time_series)
        by_spec_type: dict[str, object] = {}
        for spec in specs:
            existing = by_spec_type.get(spec.spec_type)
            if existing is not None:
                raise TimeFValidationError(
                    f"spec_type {spec.spec_type!r} has conflicting TimeSeriesSpec contracts: {existing!r} and {spec!r}"
                )
            by_spec_type[spec.spec_type] = spec
        annotations = self._ordered_unique(
            AnnotationDescriptor(
                key=annotation.key,
                annotation_type=annotation_type_of(annotation),
                value_type=value_type_of(annotation.value),
                # __post_init__ normalizes unit to a plain string (or None), so this is always str | None.
                unit=cast("str | None", annotation.unit),
                description=annotation.description,
            )
            for sample in self._samples
            for annotation in sample.annotations
        )
        by_key: dict[str, AnnotationDescriptor] = {}
        for descriptor in annotations:
            existing = by_key.get(descriptor.key)
            if existing is not None:
                raise TimeFValidationError(
                    f"annotation {descriptor.key!r} has conflicting descriptors across samples: "
                    f"{existing!r} and {descriptor!r}"
                )
            by_key[descriptor.key] = descriptor
        tasks = self._ordered_unique(type(task) for task in self._tasks)
        self._schema = DatasetSchema(
            time_series_specs=tuple(specs),
            annotations=tuple(annotations),
            tasks=tuple(tasks),
        )
        return self._schema

    @classmethod
    def from_parts(
        cls,
        *,
        metadata: DatasetMetadata,
        samples: Iterable[Sample],
        tasks: Iterable[Task],
        schema: DatasetSchema,
    ) -> "TimeFDataset":
        """Build a dataset from already-constructed parts (used by the reader on read-back).

        Args:
            metadata: The dataset's descriptive identity.
            samples: Fully-built samples (their loaders pull from disk).
            tasks: Fully-built tasks with resolved ``from_tasks``.
            schema: The schema reconstructed from the manifest.

        Returns:
            The hydrated dataset.
        """
        dataset = cls(metadata=metadata)
        dataset._samples = list(samples)
        dataset._tasks = list(tasks)
        dataset._schema = schema
        return dataset

    @staticmethod
    def _check_task_answer(task: Task) -> None:
        """Reject a task whose answer is both inline and by reference, or missing entirely.

        Args:
            task: The task being registered.

        Raises:
            TimeFValidationError: If ``target`` and ``target_annotation_ids`` are both set, or both are
                unset on a task whose answer is not a produced series.
        """
        name = type(task).__name__
        if task.target is not None and task.target_annotation_ids:
            raise TimeFValidationError(
                f"{name} sets both target={task.target!r} and target_annotation_ids "
                f"{list(task.target_annotation_ids)}; the answer is either inline or by reference, not both"
            )
        if not type(task).answer_is_sample and task.target is None and not task.target_annotation_ids:
            raise TimeFValidationError(
                f"{name} needs an answer: pass target=, or target_annotation_ids= to point at stored annotations"
            )

    @staticmethod
    def _check_annotation_refs(task: Task, samples: tuple[Sample, ...]) -> None:
        """Reject an input or target annotation id that none of the task's samples carries.

        Args:
            task: The task being registered.
            samples: The samples the task is attached to.

        Raises:
            TimeFValidationError: If a referenced annotation id is not attached to any target sample.
        """
        known = {annotation.id for sample in samples for annotation in sample.annotations}
        for field_name in ("input_annotation_ids", "target_annotation_ids"):
            for annotation_id in getattr(task, field_name):
                if annotation_id not in known:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {field_name} references annotation {annotation_id!r}, "
                        f"which is not attached to any of samples {[s.sample_id for s in samples]}"
                    )

    def _check_sample_refs(self, task: Task) -> None:
        """Reject a payload sample id that is not registered in this dataset.

        Args:
            task: The task being registered.

        Raises:
            TimeFValidationError: If a payload reference names an unknown sample.
        """
        known = {sample.sample_id for sample in self._samples}
        for field_name in type(task).refs.sample_id_fields:
            value = getattr(task, field_name)
            sample_ids = (value,) if isinstance(value, str) else value or ()
            for sample_id in sample_ids:
                if sample_id not in known:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {field_name} references unknown sample {sample_id!r}"
                    )

    @staticmethod
    def _ordered_unique(items: Iterable[T]) -> list[T]:
        """Return items with duplicates removed, preserving first-seen order."""
        return list(dict.fromkeys(items))

    @property
    def metadata(self) -> DatasetMetadata:
        """The dataset's descriptive identity."""
        return self._metadata

    @property
    def samples(self) -> tuple[Sample, ...]:
        """All samples in insertion order."""
        return tuple(self._samples)

    @property
    def tasks(self) -> tuple[Task, ...]:
        """All tasks in insertion order."""
        return tuple(self._tasks)

    @property
    def schema(self) -> DatasetSchema | None:
        """The derived schema, or ``None`` until :meth:`derive_schema` is called."""
        return self._schema

    def tasks_of(self, task_type: type[TTask]) -> tuple[TTask, ...]:
        """Return every task of a given type, in insertion order.

        Args:
            task_type: The task subclass to keep (e.g. :class:`~timenet.types.ClassificationTask`).

        Returns:
            The matching tasks.
        """
        return tuple(task for task in self._tasks if isinstance(task, task_type))

    @overload
    def tasks_for(self, sample: Sample) -> tuple[Task, ...]: ...
    @overload
    def tasks_for(self, sample: Sample, task_type: type[TTask]) -> tuple[TTask, ...]: ...
    def tasks_for(self, sample: Sample, task_type: type[Task] = Task) -> tuple[Task, ...]:
        """Return the tasks attached to a sample, optionally filtered by type.

        The inverse of the stored direction: tasks reference their samples, so this resolves a
        sample's ``task_ids`` back to the task objects.

        Args:
            sample: The sample whose tasks to resolve.
            task_type: Keep only tasks of this subclass (defaults to every task on the sample).

        Returns:
            The sample's tasks of ``task_type``, in the sample's task order.

        Raises:
            TimeFValidationError: If ``sample`` is not registered in this dataset, or if one of its
                ``task_ids`` does not resolve to a registered task that links back to the sample.
        """
        registered_ids = {registered.sample_id for registered in self._samples}
        if sample.sample_id not in registered_ids:
            raise TimeFValidationError(f"sample {sample.sample_id!r} is not registered in this dataset")
        by_id = {task.id: task for task in self._tasks}
        resolved: list[Task] = []
        for task_id in sample.task_ids:
            task = by_id.get(task_id)
            if task is None or sample.sample_id not in task.sample_ids:
                raise TimeFValidationError(
                    f"sample {sample.sample_id!r} links task {task_id!r}, but the task is missing or does "
                    f"not link back to the sample"
                )
            resolved.append(task)
        return tuple(task for task in resolved if isinstance(task, task_type))

    @overload
    def to_features_and_targets(
        self,
        *,
        task: type[Task] | None = ...,
        output: Literal["arrow"] = ...,
        features: Literal["timestep", "series"] = ...,
    ) -> tuple[pa.Array, pa.Array]: ...
    @overload
    def to_features_and_targets(
        self,
        *,
        task: type[Task] | None = ...,
        output: Literal["numpy"],
        features: Literal["timestep", "series"] = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def to_features_and_targets(
        self,
        *,
        task: type[Task] | None = None,
        output: Literal["arrow", "numpy"] = "arrow",
        features: Literal["timestep", "series"] = "timestep",
    ) -> tuple[pa.Array, pa.Array] | tuple[np.ndarray, np.ndarray]:
        """Build an ``(X, y)`` training pair, deferring materialization by default.

        Requires every sample to carry exactly one task of ``task`` and pairs its sole channel's values
        with that task's ``target``. ``features`` chooses the shape of ``X``:

        - ``"timestep"`` (default): one feature per point, a rectangular matrix. Needs equal-length
          samples. Arrow ``FixedSizeListArray[T]``; NumPy ``(n, T)`` ``float32``.
        - ``"series"``: one sequence feature per sample, so variable-length series are fine. Arrow
          ``ListArray``; NumPy ``(n,)`` object array of 1-D arrays.

        ``output="arrow"`` (the default) builds those straight from the series loaders with no NumPy copy
        in between; ``output="numpy"`` materializes them. ``y`` is always the targets (an Arrow string
        array or a 1-D NumPy array).

        Args:
            task: The task type to read targets from (e.g.
                :class:`~timenet.types.ClassificationTask`). Omit it to infer the type when the dataset
                has exactly one task type carrying an inline target.
            output: ``"arrow"`` to keep the deferred Arrow arrays, or ``"numpy"`` to materialize them.
            features: ``"timestep"`` for a rectangular per-point matrix, or ``"series"`` for one
                variable-length sequence per sample.

        Returns:
            ``(X, y)`` as two Arrow arrays (``output="arrow"``) or two NumPy arrays (``output="numpy"``).

        Raises:
            TimeFValidationError: If ``output``/``features`` is invalid; if ``task`` is omitted and the
                dataset has zero or several task types with inline targets; if a matched task carries no inline target
                (its answer is a produced series, or stored as ``target_annotation_ids``); if a matched
                sample is not single-channel; or ``features="timestep"`` is asked of samples that are not
                all the same length; or if the dataset has no samples or any sample does not carry exactly
                one task of ``task``.
        """
        if output not in {"arrow", "numpy"}:
            raise TimeFValidationError(f"output must be 'arrow' or 'numpy', got {output!r}")
        if features not in {"timestep", "series"}:
            raise TimeFValidationError(f"features must be 'timestep' or 'series', got {features!r}")
        resolved = task if task is not None else self._infer_target_task()
        rows, targets = self._rows_and_targets(resolved)
        # Build only the representation asked for: no Arrow list array on the NumPy path, and no
        # concat of every point on the series+NumPy path.
        try:
            y = pa.array(targets)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise TimeFValidationError(
                f"{resolved.__name__} targets cannot be represented as a scalar Arrow target array"
            ) from exc
        if features == "series":  # one variable-length sequence per sample
            if output == "numpy":
                x_obj = np.empty(len(rows), dtype=object)
                x_obj[:] = [row.to_numpy(zero_copy_only=False) for row in rows]
                return x_obj, y.to_numpy(zero_copy_only=False)
            offsets = [0]
            for row in rows:
                offsets.append(offsets[-1] + len(row))
            return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.concat_arrays(rows)), y
        # features == "timestep": a rectangular matrix, so every sample must share one length
        length = len(rows[0])
        if any(len(row) != length for row in rows):
            raise TimeFValidationError(
                f"features='timestep' needs equal-length samples (got {sorted({len(r) for r in rows})}); "
                "use features='series'"
            )
        values = pa.concat_arrays(rows)
        if output == "numpy":
            return values.to_numpy(zero_copy_only=False).reshape(len(rows), length), y.to_numpy(zero_copy_only=False)
        return pa.FixedSizeListArray.from_arrays(values, length), y

    def _rows_and_targets(self, resolved: type[Task]) -> tuple[list[pa.Array], list[object]]:
        """Pair each matched sample's values with its task's target, for :meth:`to_features_and_targets`.

        Args:
            resolved: The task type to read targets from.

        Returns:
            The per-sample Arrow value arrays and their targets, in sample order.

        Raises:
            TimeFValidationError: If a matched task carries no inline target, if the dataset has no
                samples, or any sample does not carry exactly one task of ``resolved``.
        """
        if not resolved.target_is_scalar:
            raise TimeFValidationError(
                f"{resolved.__name__} does not carry scalar targets supported by to_features_and_targets"
            )
        matched_by_sample: dict[str, list[Task]] = {}
        for candidate in self.tasks_of(resolved):
            for sample_id in candidate.sample_ids:
                matched_by_sample.setdefault(sample_id, []).append(candidate)
        rows: list[pa.Array] = []
        targets: list[object] = []
        for sample in self._samples:
            matched = matched_by_sample.get(sample.sample_id) or []
            if len(matched) != 1:
                raise TimeFValidationError(
                    f"sample {sample.sample_id!r} carries {len(matched)} {resolved.__name__} tasks; "
                    "to_features_and_targets needs exactly one per sample"
                )
            if matched[0].target is None:
                raise TimeFValidationError(
                    f"{resolved.__name__} {matched[0].id!r} has no inline target to use as y; its answer is "
                    f"a produced series or stored as target_annotation_ids"
                )
            rows.append(sample.to_arrow())  # Arrow straight from the loader, no NumPy copy
            targets.append(matched[0].target)
        if not rows:
            raise TimeFValidationError(f"dataset has no samples to build {resolved.__name__} features from")
        return rows, targets

    def _infer_target_task(self) -> type[Task]:
        """Infer the dataset's sole task type carrying an inline target, for :meth:`to_features_and_targets`.

        Returns:
            The single task class whose instances carry a ``target``.

        Raises:
            TimeFValidationError: If the dataset has zero or several such task types (pass ``task=``).
        """
        kinds = {type(task) for task in self._tasks if task.target is not None and type(task).target_is_scalar}
        if len(kinds) == 1:
            return kinds.pop()
        names = ", ".join(sorted(kind.__name__ for kind in kinds)) or "(none)"
        raise TimeFValidationError(f"pass task= to to_features_and_targets; dataset has target task types: {names}")

    def describe(self, *, rows: int = 5, file: TextIO | None = None) -> None:
        """Print a plain-text summary: identity, counts, specs/columns, and a sample preview.

        Like pandas' ``describe``/``info``. The preview reads only span metadata (no series values);
        value dtypes are sampled from one series per spec. Works before :meth:`derive_schema` since
        everything is computed from the samples.

        Args:
            rows: Number of samples to show in the preview.
            file: Where to write (defaults to ``sys.stdout``).
        """
        print(describe_text(self, rows=rows), file=file or sys.stdout)
