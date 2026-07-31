"""The :class:`TimeFDataset` in-memory model a connector populates during ``convert()``."""

from collections.abc import Iterable
import sys
from typing import Literal, TextIO, TypeVar, cast, overload

import numpy as np
import pyarrow as pa

from timenet.dataset.describe import describe_text
from timenet.dataset.sample import Sample
from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import (
    AnnotationDescriptor,
    DatasetMetadata,
    DatasetSchema,
    LabelingTask,
    TargetTask,
    Task,
    View,
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
        view: View = View.FULL,
        subject_ids: tuple[str, ...] = (),
        sample_id: str | None = None,
        t0_unix_ns: int | None = None,
    ) -> Sample:
        """Create a sample, register it, and return it.

        Args:
            time_series: The logical :class:`TimeSeries` streams the sample uses.
            view: Which slice of the source this sample represents (defaults to the full recording).
            subject_ids: Subjects this sample belongs to (empty for subject-less domains).
            sample_id: An explicit id (default: an auto-generated uuid4). Pass one for deterministic
                output, e.g. when generating golden fixtures.
            t0_unix_ns: Wall-clock anchor for the sample's relative timeline (Unix time, UTC, integer
                nanoseconds), or ``None`` when no wall-clock reference exists.

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
                view=view,
                subject_ids=tuple(subject_ids),
                t0_unix_ns=t0_unix_ns,
            )
        else:
            sample = Sample(
                sample_id=sample_id,
                time_series=tuple(time_series),
                view=view,
                subject_ids=tuple(subject_ids),
                t0_unix_ns=t0_unix_ns,
            )
        self._samples.append(sample)
        return sample

    def add_task(
        self,
        samples: Sample | Iterable[Sample],
        task: Task,
        *,
        from_tasks: tuple[Task, ...] = (),
    ) -> Task:
        """Register a task and link it to its samples.

        Args:
            samples: The sample, or samples, the task is attached to.
            task: The task instance (payload already set by the caller).
            from_tasks: Source tasks this task derives from. Overrides the task's own ``from_tasks``
                only when non-empty, so a task constructed with ``from_tasks=`` is not clobbered.

        Returns:
            The registered task (same instance, with ``sample_ids`` populated).

        Raises:
            TimeFValidationError: If ``samples`` is empty, if a ``LabelingTask``'s ``time_series_ids``
                does not resolve to a series on every target sample, or if one of its ``windows_s``
                falls outside a target sample's span.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise TimeFValidationError("add_task requires at least one sample")

        if isinstance(task, LabelingTask):
            for sample in targets:
                if task.time_series_ids is not None:
                    series_ids = {ts.time_series_id for ts in sample.time_series}
                    for series_id in task.time_series_ids:
                        if series_id not in series_ids:
                            raise TimeFValidationError(
                                f"LabelingTask references unknown time_series_id {series_id!r} "
                                f"on sample {sample.sample_id!r}"
                            )
                self._check_windows_within_sample(task, sample)

        if from_tasks:
            task.from_tasks = tuple(from_tasks)
        task.sample_ids = tuple(sample.sample_id for sample in targets)
        for sample in targets:
            sample.task_ids = (*sample.task_ids, task.id)
        self._tasks.append(task)
        return task

    def derive_schema(self) -> DatasetSchema:
        """Walk the dataset's instances and build its :class:`DatasetSchema`.

        Collects the distinct spec, data-source, annotation, and task types, stores the result on the
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
        data_sources = self._ordered_unique(spec.data_source for spec in specs if spec.data_source is not None)
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
            data_sources=tuple(data_sources),
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
    def _check_windows_within_sample(task: LabelingTask, sample: Sample) -> None:
        """Reject a labeling window that falls outside the sample's span.

        ``windows_s`` is in the source recording timeline, the same frame as ``TimeSeries.t_start_s``,
        so a window is checked against the union of the targeted series' spans. A series with an open
        ``t_end_s`` imposes no upper bound.

        Args:
            task: The labeling task whose ``windows_s`` to check.
            sample: The sample the task is being attached to.

        Raises:
            TimeFValidationError: If a window lies outside the covered span.
        """
        if task.windows_s is None:
            return
        covered = [
            ts for ts in sample.time_series if task.time_series_ids is None or ts.time_series_id in task.time_series_ids
        ]
        if not covered:
            return
        span_start = min(ts.t_start_s for ts in covered)
        ends = [ts.t_end_s for ts in covered]
        span_end = None if any(end is None for end in ends) else max(end for end in ends if end is not None)
        for start_s, end_s in task.windows_s:
            if start_s < span_start or (span_end is not None and end_s > span_end):
                raise TimeFValidationError(
                    f"LabelingTask window ({start_s}, {end_s}) falls outside sample "
                    f"{sample.sample_id!r} span ({span_start}, {span_end}); windows_s is in the "
                    f"source recording timeline"
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
        task: type[TargetTask] | None = ...,
        output: Literal["arrow"] = ...,
        features: Literal["timestep", "series"] = ...,
    ) -> tuple[pa.Array, pa.Array]: ...
    @overload
    def to_features_and_targets(
        self,
        *,
        task: type[TargetTask] | None = ...,
        output: Literal["numpy"],
        features: Literal["timestep", "series"] = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def to_features_and_targets(
        self,
        *,
        task: type[TargetTask] | None = None,
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
            task: The target-bearing task type to read labels from (e.g.
                :class:`~timenet.types.ClassificationTask`). Omit it to infer the type when the dataset
                has exactly one target-bearing task type; ``ForecastingTask`` has no target.
            output: ``"arrow"`` to keep the deferred Arrow arrays, or ``"numpy"`` to materialize them.
            features: ``"timestep"`` for a rectangular per-point matrix, or ``"series"`` for one
                variable-length sequence per sample.

        Returns:
            ``(X, y)`` as two Arrow arrays (``output="arrow"``) or two NumPy arrays (``output="numpy"``).

        Raises:
            ValueError: If ``output``/``features`` is invalid; if ``task`` is omitted and the dataset has
                zero or several target-bearing task types; if a sample is not single-channel; or
                ``features="timestep"`` is asked of samples that are not all the same length.
            TimeFValidationError: If the dataset has no samples, or any sample does not carry exactly one
                task of ``task``.
        """
        if output not in {"arrow", "numpy"}:
            raise ValueError(f"output must be 'arrow' or 'numpy', got {output!r}")
        if features not in {"timestep", "series"}:
            raise ValueError(f"features must be 'timestep' or 'series', got {features!r}")
        resolved = task if task is not None else self._infer_target_task()
        matched_by_sample: dict[str, list[TargetTask]] = {}
        for candidate in self.tasks_of(resolved):
            for sample_id in candidate.sample_ids:
                matched_by_sample.setdefault(sample_id, []).append(candidate)
        rows: list[pa.Array] = []
        targets: list[str] = []
        for sample in self._samples:
            matched = matched_by_sample.get(sample.sample_id) or []
            if len(matched) != 1:
                raise TimeFValidationError(
                    f"sample {sample.sample_id!r} carries {len(matched)} {resolved.__name__} tasks; "
                    f"to_features_and_targets needs exactly one per sample"
                )
            rows.append(sample.to_arrow())  # Arrow straight from the loader, no NumPy copy
            targets.append(matched[0].target)
        if not rows:
            raise TimeFValidationError(f"dataset has no samples to build {resolved.__name__} features from")
        # Build only the representation asked for: no Arrow list array on the NumPy path, and no
        # concat of every point on the series+NumPy path.
        y = pa.array(targets)
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
            raise ValueError(
                f"features='timestep' needs equal-length samples (got {sorted({len(r) for r in rows})}); "
                "use features='series'"
            )
        values = pa.concat_arrays(rows)
        if output == "numpy":
            return values.to_numpy(zero_copy_only=False).reshape(len(rows), length), y.to_numpy(zero_copy_only=False)
        return pa.FixedSizeListArray.from_arrays(values, length), y

    def _infer_target_task(self) -> type[TargetTask]:
        """Infer the dataset's sole target-bearing task type, for :meth:`to_features_and_targets`.

        Returns:
            The single task class carrying a ``target``.

        Raises:
            ValueError: If the dataset has zero or several target-bearing task types (pass ``task=``).
        """
        kinds = {type(task) for task in self.tasks_of(TargetTask)}
        if len(kinds) == 1:
            return kinds.pop()
        names = ", ".join(sorted(kind.__name__ for kind in kinds)) or "(none)"
        raise ValueError(f"pass task= to to_features_and_targets; dataset has target task types: {names}")

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
