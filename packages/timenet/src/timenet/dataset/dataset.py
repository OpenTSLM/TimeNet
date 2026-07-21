"""The :class:`TimeFDataset` in-memory model a connector populates during ``convert()``."""

from collections.abc import Iterable
import sys
from typing import TextIO, TypeVar

from timenet.dataset.describe import describe_text
from timenet.dataset.sample import Sample
from timenet.dataset.time_series import TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import (
    AnnotationDescriptor,
    DatasetMetadata,
    DatasetSchema,
    LabelingTask,
    Task,
    View,
    annotation_type_of,
    value_type_of,
)


T = TypeVar("T")


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
        view: View,
        subject_ids: tuple[str, ...] = (),
        sample_id: str | None = None,
    ) -> Sample:
        """Create a sample, register it, and return it.

        Args:
            time_series: One :class:`TimeSeries` per channel the sample uses.
            view: Which slice of the source this sample represents.
            subject_ids: Subjects this sample belongs to (empty for subject-less domains).
            sample_id: An explicit id (default: an auto-generated uuid4). Pass one for deterministic
                output, e.g. when generating golden fixtures.

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
            sample = Sample(time_series=tuple(time_series), view=view, subject_ids=tuple(subject_ids))
        else:
            sample = Sample(
                sample_id=sample_id,
                time_series=tuple(time_series),
                view=view,
                subject_ids=tuple(subject_ids),
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
            ValueError: If one annotation key yields conflicting descriptors across samples (e.g. the
                same key seen with different value types or units).
        """
        specs = self._ordered_unique(ts.spec for sample in self._samples for ts in sample.time_series)
        data_sources = self._ordered_unique(spec.data_source for spec in specs if spec.data_source is not None)
        annotations = self._ordered_unique(
            AnnotationDescriptor(
                key=annotation.key,
                annotation_type=annotation_type_of(annotation),
                value_type=value_type_of(annotation.value),
                unit=annotation.unit,
                description=annotation.description,
            )
            for sample in self._samples
            for annotation in sample.annotations
        )
        by_key: dict[str, AnnotationDescriptor] = {}
        for descriptor in annotations:
            existing = by_key.get(descriptor.key)
            if existing is not None:
                raise ValueError(
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
