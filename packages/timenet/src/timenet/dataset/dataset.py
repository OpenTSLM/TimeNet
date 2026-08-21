"""The :class:`TimeFDataset` class is the in-memory model that a connector populates during ``convert()``."""

from collections.abc import Callable, Iterable, Iterator, Sequence
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
    Annotation,
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
    """Holds samples and their tasks as Python objects. It does not do I/O. The writer handles persistence."""

    def __init__(self, *, metadata: DatasetMetadata) -> None:
        """Create an empty dataset.

        Args:
            metadata: The dataset's descriptive identity.
        """
        self._metadata = metadata
        self._samples: list[Sample] = []
        self._tasks: list[Task] = []
        # Annotations that tasks reference but no sample carries, deduped by id. A task's metadata
        # (for example a question's answer options) lives here once, referenced by input_annotation_ids,
        # instead of being copied onto every sample the tasks are about.
        self._registered_annotations: dict[str, Annotation] = {}
        # An optional re-iterable task source. When set, tasks stream past the dataset instead of
        # accumulating in _tasks, so a dataset with millions of tasks over few samples still fits.
        self._task_stream: Callable[[], Iterator[Task]] | None = None
        self._streamed_task_types: tuple[type[Task], ...] = ()
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
            time_series: The logical :class:`TimeSeries` streams that the sample uses.
            subject_ids: The subjects that this sample belongs to. The tuple is empty for
                domains that have no subjects.
            sample_id: An explicit ID. The default is an automatically generated uuid4 value.
                Pass an explicit ID for deterministic output, for example for golden test fixtures.
            start_time: The wall-clock timestamp for the sample's relative zero point. This value
                can be a timezone-aware datetime or a whole number of Unix microseconds. Use
                ``None`` when no wall-clock reference exists.
            time_span: The overall span of the session. Use this if the series have gaps that an
                unscoped span can fall into (see :attr:`Sample.time_span`). The value must be a
                whole-sample :class:`~timenet.types.TimeInterval` object that contains every
                series window.

        Returns:
            The newly created :class:`Sample`.

        Raises:
            TimeFValidationError: This error occurs if ``time_series`` is empty, or if two series
                share the same ``time_series_id`` value. The ids must be distinct. The writer uses
                the ids as shard keys, and :meth:`Sample.add_annotation` and :meth:`add_task` both
                resolve references by these ids. So a repeated id silently merges two channels
                into one.
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

        This method checks every span that the task carries against the samples given here. This
        includes the task's ``scope`` and, for a :class:`~timenet.types.TemporalLocalizationTask`,
        its target regions. The samples are available here, so the method checks them at this
        point. The method also checks the sample and annotation ids that the task references, and
        the source tasks listed in its ``from_tasks``.

        Set ``scope`` and ``from_tasks`` on the task itself. These fields describe that one task,
        not the call to this method.

        Args:
            samples: The sample, or samples, that the task attaches to.
            task: The task instance. The caller must already set its payload, ``scope``, and
                ``from_tasks`` fields.

        Returns:
            The registered task. This is the same instance, with the ``sample_ids`` field
            populated.

        Raises:
            TimeFValidationError: This error occurs if ``samples`` is empty. It also occurs if the
                task's id is already registered, or if a task in ``from_tasks`` is neither
                registered nor the task itself. It also occurs if a scope-dependent payload rule
                fails. Examples include a :class:`~timenet.types.ForecastingTask` ``target_span``
                with no scope, a frame mismatch, or a context that leaks the target. It also occurs if
                the task sets both ``target`` and ``target_annotation_ids``, or if its answer is
                not a produced series and it sets neither field. It also occurs if a span's
                ``time_series_ids`` does not resolve to a series on every target sample. It also
                occurs if the span falls outside a sample's covered span. It also occurs if a
                referenced sample or annotation is not registered in this dataset.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise TimeFValidationError("add_task requires at least one sample")
        return self._register_batch((task,), targets)[0]

    def add_tasks(self, samples: Sample | Iterable[Sample], tasks: Iterable[Task]) -> tuple[Task, ...]:
        """Register several tasks against the same samples, all together or not at all.

        The method validates the whole batch before it attaches any task. If one task fails a
        check, the call raises an error, and the dataset and every task in the batch stay
        unchanged. To keep the tasks that passed before a failure, call :meth:`add_task` in a
        loop instead.

        A task can derive from another task in the same batch. To do this, list the parent task
        in the deriving task's ``from_tasks`` field. The batch is checked as a unit, so the order
        of tasks within ``tasks`` does not matter.

        Args:
            samples: The sample, or samples, that the tasks attach to.
            tasks: The task instances to register. For a single task, use :meth:`add_task` instead.

        Returns:
            The registered tasks, in the order given. These are the same instances, with the
            ``sample_ids`` field populated.

        Raises:
            TimeFValidationError: This error occurs if ``samples`` is empty. It also occurs if two
                tasks in the batch share an id, or if one task reuses an id that is already
                registered. It also occurs if a ``from_tasks`` parent is neither registered nor in
                the batch, or if the derivation forms a cycle. It also occurs if any task fails a
                check that :meth:`add_task` documents.
        """
        targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
        if not targets:
            raise TimeFValidationError("add_tasks requires at least one sample")
        # Drain `tasks` before validation. A connector generator can attach an annotation and then
        # yield a task that references it, so the annotation must already be on the sample when
        # the method checks the references.
        batch = tuple(tasks)
        return self._register_batch(batch, targets)

    def register_annotations(self, annotations: Iterable[Annotation]) -> None:
        """Register annotations that tasks reference but no sample carries.

        Deduped by id, so many tasks can share one annotation without copying it. The writer persists
        these alongside the sample annotations, so a task's ``input_annotation_ids`` /
        ``target_annotation_ids`` resolve without the annotation being attached to a sample. Register
        an annotation before the task that references it (:meth:`add_task` checks the reference).

        Args:
            annotations: The annotations to register. A repeated id must map to an equal annotation.

        Raises:
            TimeFValidationError: If two annotations share an id but are not equal.
        """
        for annotation in annotations:
            existing = self._registered_annotations.get(annotation.id)
            if existing is not None and existing != annotation:
                raise TimeFValidationError(
                    f"annotation id {annotation.id!r} is registered twice with different values: "
                    f"{existing!r} and {annotation!r}"
                )
            self._registered_annotations[annotation.id] = annotation

    def set_task_stream(self, task_types: Sequence[type[Task]], source: Callable[[], Iterator[Task]]) -> None:
        """Provide tasks as a re-iterable stream instead of materializing them in the dataset.

        For a dataset with far more tasks than samples (many questions over few recordings), holding
        every task in memory is the scaling wall. A streaming connector builds the bounded samples and
        registered annotations, then hands the tasks over through ``source``; the writer streams them to
        disk without a list. Streamed tasks are trusted, not validated the way :meth:`add_task` validates
        them: each must already have its ``sample_ids`` set and reference only registered annotations and
        existing samples. Streamed tasks do not populate ``Sample.task_ids``.

        Args:
            task_types: The task classes the stream yields, so :meth:`derive_schema` records them.
                Every yielded task must be one of these types.
            source: A callable returning a fresh iterator over the tasks each time it is called. The
                writer calls it more than once (a peek for id storage, then the write), so it must
                re-read its source rather than exhaust a one-shot generator.

        Raises:
            TimeFValidationError: If tasks were already added with :meth:`add_task`. A dataset either
                streams its tasks or materializes them, never both, or the writer would drop one set.
        """
        if self._tasks:
            raise TimeFValidationError(
                "set_task_stream cannot follow add_task/add_tasks: a dataset either streams its tasks or "
                "materializes them, not both"
            )
        self._streamed_task_types = tuple(task_types)
        self._task_stream = source

    def iter_tasks(self) -> Iterator[Task]:
        """Yield the dataset's tasks, from the stream when one is set, else the materialized list.

        Yields:
            Each task. A streamed dataset re-reads its source on every call.
        """
        if self._task_stream is not None:
            yield from self._task_stream()
        else:
            yield from self._tasks

    def _register_batch(self, batch: tuple[Task, ...], targets: tuple[Sample, ...]) -> tuple[Task, ...]:
        """Validate a whole batch of tasks, then attach all of it or none of it.

        :meth:`add_task` and :meth:`add_tasks` both call this method. This keeps the singular and
        plural forms consistent. The singular form is just a batch of one. Validation has no side
        effects, so the attachment step below runs only after the method confirms that the whole
        batch is good.

        Args:
            batch: The tasks to register together, already drained from the caller's iterable.
            targets: The samples that the tasks attach to.

        Returns:
            The registered tasks, in order. These are the same instances, with the ``sample_ids``
            field populated.

        Raises:
            TimeFValidationError: This error occurs under the conditions documented on
                :meth:`add_tasks`.
        """
        if self._task_stream is not None:
            raise TimeFValidationError(
                "add_task/add_tasks cannot be used on a streamed dataset: set_task_stream already provides its tasks"
            )
        self._validate_task_batch(batch, targets)
        for task in batch:
            task.sample_ids = tuple(sample.sample_id for sample in targets)
            for sample in targets:
                sample.task_ids = (*sample.task_ids, task.id)
            self._tasks.append(task)
        return batch

    def _validate_task_batch(self, batch: tuple[Task, ...], targets: tuple[Sample, ...]) -> None:
        """Run every check that the batch must pass, without attaching anything.

        This method checks the cross-task rules that only a batch makes possible, in addition to
        the per-task checks. The ids must stay unique against the batch and the dataset. Each
        ``from_tasks`` parent must already be registered or be in the batch. No task can derive
        from itself or close a cycle.

        Args:
            batch: The tasks to register together.
            targets: The samples that the tasks attach to.

        Raises:
            TimeFValidationError: This error occurs under the conditions documented on
                :meth:`add_tasks`.
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
        """Reject a ``from_tasks`` cycle that forms among the batch's own tasks.

        Only tasks in the batch can close a cycle. A task already in the dataset passed this
        check when the dataset added it. So it cannot derive from a task that did not exist yet.
        For this reason, the walk stays inside the batch. It follows the parents that each task
        shares with the batch.

        Args:
            batch: The tasks to register together.

        Raises:
            TimeFValidationError: If the batch's derivations contain a cycle.
        """
        batch_ids = {task.id for task in batch}
        parents = {task.id: {p for p in task.from_task_ids if p in batch_ids} for task in batch}
        # Kahn's algorithm: the loop peels off tasks whose parents in the batch are all resolved.
        # A task that remains when the loop can peel off no more tasks sits on a cycle.
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

        This method collects the distinct spec, annotation, and task types. It stores the result
        on the dataset, and it returns the result.

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
        sample_annotations = (annotation for sample in self._samples for annotation in sample.annotations)
        annotations = self._ordered_unique(
            AnnotationDescriptor(
                key=annotation.key,
                annotation_type=annotation_type_of(annotation),
                value_type=value_type_of(annotation.value),
                # The __post_init__ method normalizes unit to a plain string or None, so this
                # value is always str | None.
                unit=cast("str | None", annotation.unit),
                description=annotation.description,
            )
            # Registered (task-referenced) annotations carry descriptors too, so their key/type reach
            # the schema even though no sample carries them.
            for annotation in (*sample_annotations, *self._registered_annotations.values())
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
        # A streamed dataset declares its task types up front, so the schema records them without
        # draining the stream just to collect them.
        streamed = self._ordered_unique(self._streamed_task_types)
        tasks = streamed if self._task_stream is not None else self._ordered_unique(type(task) for task in self._tasks)
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
        registered_annotations: Iterable[Annotation] = (),
    ) -> "TimeFDataset":
        """Build a dataset from parts that are already constructed.

        The reader uses this method when it reads a dataset back from disk.

        Args:
            metadata: The dataset's descriptive identity.
            samples: Fully built samples. Their loaders pull data from disk.
            tasks: Fully built tasks, with their ``from_tasks`` references resolved.
            schema: The schema reconstructed from the manifest.
            registered_annotations: Annotations that tasks reference but no sample carries (see
                :meth:`register_annotations`).

        Returns:
            The dataset, built from these parts.
        """
        dataset = cls(metadata=metadata)
        dataset._samples = list(samples)
        dataset._tasks = list(tasks)
        dataset._registered_annotations = {annotation.id: annotation for annotation in registered_annotations}
        dataset._schema = schema
        return dataset

    @staticmethod
    def _check_task_answer(task: Task) -> None:
        """Reject a task whose answer is both inline and by reference, or whose answer is missing.

        Args:
            task: The task to register.

        Raises:
            TimeFValidationError: This error occurs if ``target`` and ``target_annotation_ids``
                are both set. It also occurs if both fields are unset on a task whose answer is
                not a produced series.
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

    def _check_annotation_refs(self, task: Task, samples: tuple[Sample, ...]) -> None:
        """Reject an input or target annotation id that no target sample carries and none is registered.

        Args:
            task: The task to register.
            samples: The samples that the task attaches to.

        Raises:
            TimeFValidationError: If a referenced annotation id is neither attached to a target
                sample nor registered with :meth:`register_annotations`.
        """
        known = {annotation.id for sample in samples for annotation in sample.annotations}
        known |= self._registered_annotations.keys()
        for field_name in ("input_annotation_ids", "target_annotation_ids"):
            for annotation_id in getattr(task, field_name):
                if annotation_id not in known:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {field_name} references annotation {annotation_id!r}, "
                        f"which no target sample {[s.sample_id for s in samples]} carries and which is not "
                        f"registered with register_annotations"
                    )

    def _check_sample_refs(self, task: Task) -> None:
        """Reject a payload sample id that is not registered in this dataset.

        Args:
            task: The task to register.

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
    def registered_annotations(self) -> tuple[Annotation, ...]:
        """Annotations registered for tasks to reference, which no sample carries (registration order)."""
        return tuple(self._registered_annotations.values())

    @property
    def has_task_stream(self) -> bool:
        """Whether tasks stream from a source (see :meth:`set_task_stream`) rather than the ``_tasks`` list."""
        return self._task_stream is not None

    @property
    def schema(self) -> DatasetSchema | None:
        """The derived schema, or ``None`` until :meth:`derive_schema` is called."""
        return self._schema

    def tasks_of(self, task_type: type[TTask]) -> tuple[TTask, ...]:
        """Return every task of a given type, in insertion order.

        Args:
            task_type: The task subclass to keep, for example :class:`~timenet.types.ClassificationTask`.

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

        This method reverses the stored direction. Tasks reference their samples, so this method
        resolves a sample's ``task_ids`` back to the task objects.

        Args:
            sample: The sample whose tasks to resolve.
            task_type: Keep only tasks of this subclass. The default keeps every task on the
                sample.

        Returns:
            The sample's tasks of ``task_type``, in the sample's task order.

        Raises:
            TimeFValidationError: This error occurs if ``sample`` is not registered in this
                dataset. It also occurs if one of the sample's ``task_ids`` does not resolve to a
                registered task that links back to the sample.
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
        """Build an ``(X, y)`` training pair. By default, this method defers materialization.

        This method requires every sample to carry exactly one task of ``task``. It pairs the
        values of that task's sole channel with the task's ``target``. The ``features`` argument
        chooses the shape of ``X``:

        - ``"timestep"`` (the default): one feature per point, in a rectangular matrix. This needs
          equal-length samples. The result is an Arrow ``FixedSizeListArray[T]``, or a NumPy
          ``(n, T)`` array of ``float32`` values.
        - ``"series"``: one sequence feature per sample, so variable-length series work too. The
          result is an Arrow ``ListArray``, or a NumPy ``(n,)`` object array of 1-D arrays.

        With ``output="arrow"`` (the default), the method builds these arrays straight from the
        series loaders, with no NumPy copy in between. With ``output="numpy"``, the method
        materializes them. ``y`` is always the targets, as an Arrow string array or a 1-D NumPy
        array.

        Args:
            task: The task type to read targets from, for example
                :class:`~timenet.types.ClassificationTask`. Omit this argument to infer the type
                when the dataset has exactly one task type that carries an inline target.
            output: Use ``"arrow"`` to keep the deferred Arrow arrays, or ``"numpy"`` to
                materialize them.
            features: Use ``"timestep"`` for a rectangular per-point matrix, or ``"series"`` for
                one variable-length sequence per sample.

        Returns:
            ``(X, y)`` as two Arrow arrays when ``output="arrow"``, or two NumPy arrays when
            ``output="numpy"``.

        Raises:
            TimeFValidationError: This error occurs if ``output`` or ``features`` is invalid. It
                also occurs if ``task`` is omitted and the dataset has zero or several task types
                with inline targets. It also occurs if a matched task carries no inline target,
                for example if its answer is a produced series or is stored as
                ``target_annotation_ids``. It also occurs if a matched sample is not single
                channel, or if ``features="timestep"`` is asked of samples that are not all the
                same length. It also occurs if the dataset has no samples, or if any sample does
                not carry exactly one task of ``task``.
        """
        if output not in {"arrow", "numpy"}:
            raise TimeFValidationError(f"output must be 'arrow' or 'numpy', got {output!r}")
        if features not in {"timestep", "series"}:
            raise TimeFValidationError(f"features must be 'timestep' or 'series', got {features!r}")
        resolved = task if task is not None else self._infer_target_task()
        rows, targets = self._rows_and_targets(resolved)
        # Build only the representation that the caller asked for. This skips the Arrow list
        # array on the NumPy path, and skips the concat of every point on the series+NumPy path.
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
        # features == "timestep" builds a rectangular matrix, so every sample must share one length
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
            TimeFValidationError: This error occurs if a matched task carries no inline target. It
                also occurs if the dataset has no samples, or if any sample does not carry exactly
                one task of ``resolved``.
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
        """Infer the dataset's sole task type that carries an inline target, for :meth:`to_features_and_targets`.

        Returns:
            The single task class whose instances carry a ``target``.

        Raises:
            TimeFValidationError: If the dataset has zero or several such task types. In that
                case, pass ``task=`` instead.
        """
        kinds = {type(task) for task in self._tasks if task.target is not None and type(task).target_is_scalar}
        if len(kinds) == 1:
            return kinds.pop()
        names = ", ".join(sorted(kind.__name__ for kind in kinds)) or "(none)"
        raise TimeFValidationError(f"pass task= to to_features_and_targets; dataset has target task types: {names}")

    def describe(self, *, rows: int = 5, file: TextIO | None = None) -> None:
        """Print a plain-text summary of the dataset: its identity, counts, specs and columns, and a sample preview.

        This method works like pandas' ``describe`` and ``info`` methods. The preview reads only
        span metadata, not series values. The method samples value dtypes from one series per
        spec. This method works even before :meth:`derive_schema` runs, because it computes
        everything from the samples.

        Args:
            rows: The number of samples to show in the preview.
            file: Where to write the output. The default is ``sys.stdout``.
        """
        print(describe_text(self, rows=rows), file=file or sys.stdout)
