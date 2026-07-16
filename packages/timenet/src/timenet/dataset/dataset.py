"""The :class:`TimeFDataset` in-memory model a connector populates during ``convert()``."""

from collections.abc import Iterable
import sys
from typing import TextIO, TypeVar

from timenet.dataset.describe import describe_text
from timenet.dataset.sample import Sample
from timenet.dataset.time_series import TimeSeries
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
            ValueError: If ``time_series`` is empty.
        """
        if not time_series:
            raise ValueError("add_sample requires a non-empty time_series")
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
            ValueError: If ``samples`` is empty, or a ``LabelingTask``'s ``time_series_ids`` does not
                resolve to a series on every target sample.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise ValueError("add_task requires at least one sample")

        if isinstance(task, LabelingTask) and task.time_series_ids is not None:
            for sample in targets:
                series_ids = {ts.time_series_id for ts in sample.time_series}
                for series_id in task.time_series_ids:
                    if series_id not in series_ids:
                        raise ValueError(
                            f"LabelingTask references unknown time_series_id {series_id!r} "
                            f"on sample {sample.sample_id!r}"
                        )

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
