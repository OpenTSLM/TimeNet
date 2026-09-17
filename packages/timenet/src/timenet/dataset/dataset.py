"""The :class:`TimeFDataset` class is the in-memory model that a connector populates during ``convert()``."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
import sys
from typing import Literal, TextIO, TypeVar, cast, overload

import numpy as np
import pyarrow as pa

from timenet.dataset.describe import describe_text
from timenet.dataset.record import Record, check_span_within_window
from timenet.dataset.source import Source
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
    new_id,
    value_type_of,
)
from timenet.values_backends import ValuesBackend


T = TypeVar("T")
TTask = TypeVar("TTask", bound=Task)


class TimeFDataset:  # noqa: PLR0904
    """Holds records and their tasks as Python objects. It does not do I/O. The writer handles persistence."""

    def __init__(self, *, metadata: DatasetMetadata) -> None:
        """Create an empty dataset.

        Args:
            metadata: The dataset's descriptive identity.
        """
        self._metadata = metadata
        self._records: list[Record] = []
        self._records_by_id: dict[str, Record] = {}
        self._sources_by_id: dict[str, Source] = {}
        self._signals_by_id: dict[str, TimeSeries] = {}
        self._signals_by_record_id: dict[str, tuple[TimeSeries, ...]] = {}
        self._tasks: list[Task] = []
        self._annotations: list[Annotation] = []
        # Reusable annotation content that no hierarchy object carries, deduplicated by content ID.
        self._registered_annotations: dict[str, Annotation] = {}
        # An optional re-iterable task source. When set, tasks stream past the dataset instead of
        # accumulating in _tasks, so a dataset with millions of tasks over few records still fits.
        self._task_stream: Callable[[], Iterator[Task]] | None = None
        self._streamed_task_types: tuple[type[Task], ...] = ()
        self._schema: DatasetSchema | None = None

    def add_record(  # noqa: PLR0913 - transitional constructor and complete-object registration
        self,
        *,
        record: Record | None = None,
        sources: tuple[Source, ...] = (),
        time_series: tuple[TimeSeries, ...] = (),
        subject_ids: tuple[str, ...] = (),
        record_id: str | None = None,
        start_time: datetime | int | None = None,
        time_span: TimeInterval | None = None,
    ) -> Record:
        """Register a complete record, or create one from sources or legacy flat series.

        Args:
            record: A complete record to register without rebuilding it.
            sources: Root sources used to construct a new hierarchical record.
            time_series: Legacy flat streams used to construct a record during migration.
            subject_ids: The subjects that this record belongs to. The tuple is empty for
                domains that have no subjects.
            record_id: An explicit ID. The default is an automatically generated uuid4 value.
                Pass an explicit ID for deterministic output, for example for golden test fixtures.
            start_time: The wall-clock timestamp for the record's relative zero point. This value
                can be a timezone-aware datetime or a whole number of Unix microseconds. Use
                ``None`` when no wall-clock reference exists.
            time_span: The overall span of the session. Use this if the series have gaps that an
                unscoped span can fall into (see :attr:`Record.time_span`). The value must be a
                whole-record :class:`~timenet.types.TimeInterval` object that contains every
                series window.

        Returns:
            The registered :class:`Record`. When ``record`` is given, this is the same instance.

        Raises:
            TimeFValidationError: If the call mixes ``record`` with construction arguments, creates
                an empty record, repeats a signal ID, or registers a duplicate record ID.
        """
        construction_requested = bool(sources or time_series or subject_ids or record_id or start_time or time_span)
        if record is not None and construction_requested:
            raise TimeFValidationError("add_record accepts record= or construction fields, not both")
        if record is None and not sources and not time_series:
            raise TimeFValidationError("add_record requires record=, sources=, or legacy time_series=")
        duplicates = self._duplicate_ids(ts.id for ts in time_series)
        if record is None and duplicates:
            raise TimeFValidationError(
                f"add_record requires distinct time_series_ids (signal IDs), got duplicates {duplicates}"
            )
        if record is not None:
            registered = record
        else:
            resolved_record_id = record_id or new_id()
            resolved_sources = tuple(sources)
            if time_series and not resolved_sources:
                resolved_sources = (
                    Source(
                        id=f"{resolved_record_id}-source",
                        name="Source",
                        signals=tuple(time_series),
                    ),
                )
            registered = Record(
                record_id=resolved_record_id,
                sources=resolved_sources,
                time_series=tuple(time_series),
                subject_ids=tuple(subject_ids),
                start_time=start_time,
                time_span=time_span,
            )
        if registered.record_id in self._records_by_id:
            raise TimeFValidationError(f"record id {registered.record_id!r} is already registered")
        record_sources, record_signals = self._index_record_hierarchy(registered)
        repeated_sources = sorted(self._sources_by_id.keys() & record_sources.keys())
        if repeated_sources:
            raise TimeFValidationError(
                f"source id(s) {repeated_sources} already belong to another record; each Source has one owner"
            )
        repeated_signals = sorted(self._signals_by_id.keys() & record_signals.keys())
        if repeated_signals:
            raise TimeFValidationError(
                f"signal id(s) {repeated_signals} already belong to another Source; each Signal has one owner"
            )
        self._records.append(registered)
        self._records_by_id[registered.record_id] = registered
        self._sources_by_id.update(record_sources)
        self._signals_by_id.update(record_signals)
        self._signals_by_record_id[registered.record_id] = tuple(record_signals.values())
        return registered

    @staticmethod
    def _duplicate_ids(ids: Iterable[str]) -> list[str]:
        """Return repeated IDs in sorted order without repeatedly scanning the input."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item_id in ids:
            if item_id in seen:
                duplicates.add(item_id)
            else:
                seen.add(item_id)
        return sorted(duplicates)

    @classmethod
    def _index_record_hierarchy(cls, record: Record) -> tuple[dict[str, Source], dict[str, TimeSeries]]:
        """Collect one record's hierarchy and reject IDs repeated inside it.

        Returns:
            Sources and Signals keyed by their stable public IDs.

        Raises:
            TimeFValidationError: If two Sources or Signals in the record share an ID.
        """
        sources = tuple(record.walk_sources())
        source_duplicates = cls._duplicate_ids(source.id for source in sources)
        if source_duplicates:
            raise TimeFValidationError(f"record {record.record_id!r} contains duplicate source IDs {source_duplicates}")
        signals = (
            tuple(
                signal for source in sources for signal in sorted(source.signals, key=lambda item: (item.name, item.id))
            )
            if sources
            else tuple(record.time_series)
        )
        signal_duplicates = cls._duplicate_ids(signal.id for signal in signals)
        if signal_duplicates:
            raise TimeFValidationError(f"record {record.record_id!r} contains duplicate signal IDs {signal_duplicates}")
        return (
            {source.id: source for source in sources},
            {signal.id: signal for signal in signals},
        )

    def write(
        self,
        *,
        path: str | Path,
        values_backend: ValuesBackend = ValuesBackend.PARQUET,
    ) -> Path:
        """Write this dataset as one immutable TimeF version.

        Args:
            path: Registry root under which the dataset ID and version directories are created.
            values_backend: Typed values-plane backend selection.

        Returns:
            The committed version directory.
        """
        from timenet.writer import TimeFWriter  # noqa: PLC0415

        root = Path(path)
        if self.schema is None:
            self.derive_schema()
        with TimeFWriter(root, self, values_backend=values_backend) as writer:
            writer.write()
        return root / self.metadata.dataset_id / str(self.metadata.dataset_version)

    @classmethod
    def open(cls, *, path: str | Path) -> "TimeFDataset":
        """Open one local TimeF version directory.

        Args:
            path: Directory containing ``manifest.json`` and ``control.duckdb``.

        Returns:
            The hydrated dataset with lazy Signal values.
        """
        from timenet.reader import TimeFReader  # noqa: PLC0415
        from timenet.registry.version import DatasetVersion  # noqa: PLC0415

        with TimeFReader(DatasetVersion.open_local(path)) as reader:
            return reader.read()

    def add_task(self, *, task: Task) -> Task:
        """Validate and register one fully constructed task.

        The Task owns its input and target object references. Registration verifies those objects
        against this dataset before changing either the dataset or its Records.

        Args:
            task: The concrete task to register.

        Returns:
            The same task instance.

        """
        return self._register_batch((task,))[0]

    def add_tasks(self, *, tasks: Iterable[Task]) -> tuple[Task, ...]:
        """Register several complete tasks, all together or not at all.

        The method validates the whole batch before it attaches any task. If one task fails a
        check, the call raises an error, and the dataset and every task in the batch stay
        unchanged. To keep the tasks that passed before a failure, call :meth:`add_task` in a
        loop instead.

        A task can derive from another task in the same batch. To do this, list the parent task
        in the deriving task's ``from_tasks`` field. The batch is checked as a unit, so the order
        of tasks within ``tasks`` does not matter.

        Args:
            tasks: The task instances to register. For a single task, use :meth:`add_task` instead.

        Returns:
            The registered tasks, in the order given.

        """
        # Drain tasks before validation. A connector generator can attach an annotation and then
        # yield a task that references it, so the annotation must already be on the record when
        # the method checks the references.
        batch = tuple(tasks)
        return self._register_batch(batch)

    def register_annotations(self, annotations: Iterable[Annotation]) -> None:
        """Register annotations that tasks reference but no record carries.

        The content is deduplicated by ID. This supports shared vocabularies and other reusable
        metadata without attaching an occurrence to a hierarchy object.

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

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach a static annotation to the dataset.

        Returns:
            The attached occurrence.

        Raises:
            TimeFValidationError: If the annotation has a time placement, which is ambiguous across records.
        """
        if annotation.span is not None:
            raise TimeFValidationError("dataset annotations cannot have a time span")
        attached = annotation._new_occurrence()
        self._annotations.append(attached)
        return attached

    def set_task_stream(self, task_types: Sequence[type[Task]], source: Callable[[], Iterator[Task]]) -> None:
        """Provide tasks as a re-iterable stream instead of materializing them in the dataset.

        For a dataset with far more tasks than records (many questions over few recordings), holding
        every task in memory is the scaling wall. A streaming connector builds the bounded records and
        registered annotations, then hands the tasks over through ``source``; the writer streams them to
        disk without a list. Streamed tasks do not populate ``Record.task_ids``.

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

    def iter_streamed_tasks_validated(self) -> Iterator[Task]:
        """Yield the streamed tasks, validating each against the dataset before it is written.

        Streamed tasks skip :meth:`add_task`'s checks, so validate each here as it passes through: an
        undeclared type, an attachment to an unknown record, a dangling reference, a bad answer, or an
        out-of-window span raises before the task reaches disk. The dataset holds no task list, so the
        cross-task checks (duplicate ids, ``from_tasks`` derivations) that need every task at once do
        not run for a stream.

        Yields:
            Each validated task, in the source's order.

        Raises:
            TimeFValidationError: If a streamed task fails one of the per-task checks.
        """  # noqa: DOC502 (raised by _validate_streamed_task, not directly here)
        by_id = {record.record_id: record for record in self._records}
        declared = set(self._streamed_task_types)
        for task in self.iter_tasks():
            self._validate_streamed_task(task, by_id, declared)
            yield task

    def _validate_streamed_task(self, task: Task, by_id: dict[str, Record], declared: set[type[Task]]) -> None:
        """Run the per-task checks a streamed task must pass, without cross-task state.

        Args:
            task: The streamed task.
            by_id: The dataset's records, keyed by ``record_id``.
            declared: The task types :meth:`set_task_stream` declared.

        Raises:
            TimeFValidationError: If the task's type, record attachment, references, answer, or spans
                are invalid.
        """
        if type(task) not in declared:
            raise TimeFValidationError(
                f"streamed task {task.id!r} has type {type(task).__name__}, not one of the declared "
                f"{sorted(t.__name__ for t in declared)}"
            )
        referenced = self._task_records(task)
        for record in referenced:
            registered = by_id.get(record.id)
            if registered is not record:
                raise TimeFValidationError(
                    f"{type(task).__name__} {task.id!r} refers to Record {record.id!r}, which is not "
                    "registered in this dataset"
                )
        task.check_against_scope()
        self._check_task_answer(task)
        self._check_signal_refs(task)
        for record in task.inputs:
            for span in task.spans():
                check_span_within_window(
                    f"{type(task).__name__} span", span, record.signals, record.record_id, record.time_span
                )
        self._check_annotation_refs(task, task.inputs)

    def _register_batch(self, batch: tuple[Task, ...]) -> tuple[Task, ...]:
        """Validate a whole batch of tasks, then attach all of it or none of it.

        :meth:`add_task` and :meth:`add_tasks` both call this method. This keeps the singular and
        plural forms consistent. The singular form is just a batch of one. Validation has no side
        effects, so the attachment step below runs only after the method confirms that the whole
        batch is good.

        Args:
            batch: The tasks to register together, already drained from the caller's iterable.

        Returns:
            The registered tasks, in order.

        Raises:
            TimeFValidationError: This error occurs under the conditions documented on
                :meth:`add_tasks`.
        """
        if self._task_stream is not None:
            raise TimeFValidationError(
                "add_task/add_tasks cannot be used on a streamed dataset: set_task_stream already provides its tasks"
            )
        self._validate_task_batch(batch)
        for task in batch:
            for record in task.inputs:
                record.task_ids = (*record.task_ids, task.id)
            self._tasks.append(task)
        return batch

    def _validate_task_batch(self, batch: tuple[Task, ...]) -> None:
        """Run every check that the batch must pass, without attaching anything.

        This method checks the cross-task rules that only a batch makes possible, in addition to
        the per-task checks. The ids must stay unique against the batch and the dataset. Each
        ``from_tasks`` parent must already be registered or be in the batch. No task can derive
        from itself or close a cycle.

        Args:
            batch: The tasks to register together.

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
            for parent in task.from_tasks:
                parent_id = parent.id
                if parent_id == task.id:
                    raise TimeFValidationError(f"{type(task).__name__} {task.id!r} lists itself in from_tasks")
                if parent_id not in known_task_ids:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {task.id!r} derives from task {parent_id!r}, which is not "
                        f"registered in this dataset or part of the batch"
                    )
        self._check_no_derivation_cycle(batch)
        for task in batch:
            self._check_record_refs(task)
            task.check_against_scope()
            self._check_task_answer(task)
            self._check_signal_refs(task)
            for record in task.inputs:
                for span in task.spans():
                    check_span_within_window(
                        f"{type(task).__name__} span", span, record.signals, record.record_id, record.time_span
                    )
            self._check_annotation_refs(task, task.inputs)

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
        parents = {task.id: {parent.id for parent in task.from_tasks if parent.id in batch_ids} for task in batch}
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
                across records.
        """
        specs = self._ordered_unique(signal.spec for record in self._records for signal in record.signals)
        by_spec_type: dict[str, object] = {}
        for spec in specs:
            existing = by_spec_type.get(spec.spec_type)
            if existing is not None:
                raise TimeFValidationError(
                    f"spec_type {spec.spec_type!r} has conflicting TimeSeriesSpec contracts: {existing!r} and {spec!r}"
                )
            by_spec_type[spec.spec_type] = spec
        record_annotations = (annotation for record in self._records for annotation in record.annotations)
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
            # the schema even though no record carries them.
            for annotation in (*record_annotations, *self._registered_annotations.values())
        )
        by_key: dict[str, AnnotationDescriptor] = {}
        for descriptor in annotations:
            existing = by_key.get(descriptor.key)
            if existing is not None:
                raise TimeFValidationError(
                    f"annotation {descriptor.key!r} has conflicting descriptors across records: "
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
    def from_parts(  # noqa: PLR0913 - reader reconstruction names each stored component
        cls,
        *,
        metadata: DatasetMetadata,
        records: Iterable[Record],
        tasks: Iterable[Task],
        schema: DatasetSchema,
        registered_annotations: Iterable[Annotation] = (),
        annotations: Iterable[Annotation] = (),
    ) -> "TimeFDataset":
        """Build a dataset from parts that are already constructed.

        The reader uses this method when it reads a dataset back from disk.

        Args:
            metadata: The dataset's descriptive identity.
            records: Fully built records. Their loaders pull data from disk.
            tasks: Fully built tasks, with their ``from_tasks`` references resolved.
            schema: The schema reconstructed from the manifest.
            registered_annotations: Annotations that tasks reference but no record carries (see
                :meth:`register_annotations`).
            annotations: Annotations attached directly to the dataset.

        Returns:
            The dataset, built from these parts.
        """
        dataset = cls(metadata=metadata)
        for record in records:
            dataset.add_record(record=record)
        dataset._tasks = list(tasks)
        dataset._registered_annotations = {annotation.id: annotation for annotation in registered_annotations}
        dataset._annotations = list(annotations)
        dataset._schema = schema
        return dataset

    @staticmethod
    def _check_task_answer(task: Task) -> None:
        """Reject a task whose answer is ambiguous or missing.

        Args:
            task: The task to register.

        Raises:
            TimeFValidationError: If inline targets and target annotations are both present, or if
                neither representation is present.
        """
        name = type(task).__name__
        if task.targets is not None and task.target_annotations:
            raise TimeFValidationError(
                f"{name} sets inline targets and target_annotations; use one answer representation"
            )
        if task.targets is None and not task.target_annotations:
            raise TimeFValidationError(f"{name} needs an answer: set targets= or target_annotations=")

    def _check_annotation_refs(self, task: Task, records: tuple[Record, ...]) -> None:
        """Reject an input or target annotation id that no target record carries and none is registered.

        Args:
            task: The task to register.
            records: The records that the task attaches to.

        Raises:
            TimeFValidationError: If a referenced annotation id is neither attached to a target
                record nor registered with :meth:`register_annotations`.
        """
        attached = [*self._annotations]
        for record in records:
            attached.extend(record.annotations)
            for source in record.walk_sources():
                attached.extend(source.annotations)
                for signal in source.signals:
                    attached.extend(signal.annotations)
        known_occurrences = {
            annotation.occurrence_id for annotation in attached if annotation.occurrence_id is not None
        }
        for field_name in ("input_annotations", "target_annotations"):
            for annotation in getattr(task, field_name):
                if annotation.occurrence_id not in known_occurrences:
                    raise TimeFValidationError(
                        f"{type(task).__name__} {field_name} refers to annotation occurrence "
                        f"{annotation.occurrence_id!r}, which is not attached to its dataset or input hierarchy"
                    )

    def _check_record_refs(self, task: Task) -> None:
        """Reject a task Record object that is not registered in this dataset.

        Args:
            task: The task to register.

        Raises:
            TimeFValidationError: If a task references an unknown or replacement Record object.
        """
        known = {record.id: record for record in self._records}
        for record in self._task_records(task):
            if known.get(record.id) is not record:
                raise TimeFValidationError(
                    f"{type(task).__name__} references Record {record.id!r}, which is not registered in this dataset"
                )

    def _check_signal_refs(self, task: Task) -> None:
        """Reject a Signal target that is not owned by a registered Record.

        Args:
            task: The task to register.

        Raises:
            TimeFValidationError: If a Signal target is unknown or is a replacement object.
        """
        known = {signal.id: signal for record in self._records for signal in record.signals}
        for signal in self._task_signals(task):
            if known.get(signal.id) is not signal:
                raise TimeFValidationError(
                    f"{type(task).__name__} references Signal {signal.id!r}, which is not registered in this dataset"
                )

    @staticmethod
    def _task_records(task: Task) -> tuple[Record, ...]:
        """Return all Records referenced by a task, without duplicate objects."""
        candidates = cast("tuple[Record, ...]", getattr(task, "candidate_records", ()))
        target_records = tuple(target for target in task.targets or () if isinstance(target, Record))
        found: list[Record] = []
        seen: set[int] = set()
        for record in (*task.inputs, *candidates, *target_records):
            identity = id(record)
            if identity not in seen:
                seen.add(identity)
                found.append(record)
        return tuple(found)

    @staticmethod
    def _task_signals(task: Task) -> tuple[TimeSeries, ...]:
        """Return all Signals referenced directly as target items."""
        return tuple(target for target in task.targets or () if isinstance(target, TimeSeries))

    @staticmethod
    def _ordered_unique(items: Iterable[T]) -> list[T]:
        """Return items with duplicates removed, preserving first-seen order."""
        return list(dict.fromkeys(items))

    @property
    def metadata(self) -> DatasetMetadata:
        """The dataset's descriptive identity."""
        return self._metadata

    @property
    def records(self) -> tuple[Record, ...]:
        """All records in insertion order."""
        return tuple(self._records)

    @property
    def tasks(self) -> tuple[Task, ...]:
        """All tasks in insertion order."""
        return tuple(self._tasks)

    @property
    def registered_annotations(self) -> tuple[Annotation, ...]:
        """Annotations registered for tasks to reference, which no record carries (registration order)."""
        return tuple(self._registered_annotations.values())

    @property
    def annotations(self) -> tuple[Annotation, ...]:
        """Return annotations attached to the dataset itself."""
        return tuple(self._annotations)

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
    def tasks_for(self, record: Record) -> tuple[Task, ...]: ...
    @overload
    def tasks_for(self, record: Record, task_type: type[TTask]) -> tuple[TTask, ...]: ...
    def tasks_for(self, record: Record, task_type: type[Task] = Task) -> tuple[Task, ...]:
        """Return the tasks attached to a record, optionally filtered by type.

        This method reverses the stored direction. Tasks reference their records, so this method
        resolves a record's ``task_ids`` back to the task objects.

        Args:
            record: The record whose tasks to resolve.
            task_type: Keep only tasks of this subclass. The default keeps every task on the
                record.

        Returns:
            The record's tasks of ``task_type``, in the record's task order.

        Raises:
            TimeFValidationError: This error occurs if ``record`` is not registered in this
                dataset. It also occurs if one of the record's ``task_ids`` does not resolve to a
                registered task that links back to the record.
        """
        registered_ids = {registered.record_id for registered in self._records}
        if record.record_id not in registered_ids:
            raise TimeFValidationError(f"record {record.record_id!r} is not registered in this dataset")
        by_id = {task.id: task for task in self._tasks}
        resolved: list[Task] = []
        for task_id in record.task_ids:
            task = by_id.get(task_id)
            if task is None or all(linked is not record for linked in task.inputs):
                raise TimeFValidationError(
                    f"record {record.record_id!r} links task {task_id!r}, but the task is missing or does "
                    f"not link back to the record"
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

        This method requires every record to carry exactly one task of ``task``. It pairs the
        values of that task's sole signal with the task's only scalar target. The ``features`` argument
        chooses the shape of ``X``:

        - ``"timestep"`` (the default): one feature per point, in a rectangular matrix. This needs
          equal-length records. The result is an Arrow ``FixedSizeListArray[T]``, or a NumPy
          ``(n, T)`` array of ``float32`` values.
        - ``"series"``: one sequence feature per record, so variable-length series work too. The
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
                one variable-length sequence per record.

        Returns:
            ``(X, y)`` as two Arrow arrays when ``output="arrow"``, or two NumPy arrays when
            ``output="numpy"``.

        Raises:
            TimeFValidationError: This error occurs if ``output`` or ``features`` is invalid. It
                also occurs if ``task`` is omitted and the dataset has zero or several task types
                with inline targets. It also occurs if a matched task carries no inline target,
                for example if its answer is a produced series or is stored as target annotations.
                It also occurs if a matched record is not single
                signal, or if ``features="timestep"`` is asked of records that are not all the
                same length. It also occurs if the dataset has no records, or if any record does
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
        if features == "series":  # one variable-length sequence per record
            if output == "numpy":
                x_obj = np.empty(len(rows), dtype=object)
                x_obj[:] = [row.to_numpy(zero_copy_only=False) for row in rows]
                return x_obj, y.to_numpy(zero_copy_only=False)
            offsets = [0]
            for row in rows:
                offsets.append(offsets[-1] + len(row))
            return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.concat_arrays(rows)), y
        # features == "timestep" builds a rectangular matrix, so every record must share one length
        length = len(rows[0])
        if any(len(row) != length for row in rows):
            raise TimeFValidationError(
                f"features='timestep' needs equal-length records (got {sorted({len(r) for r in rows})}); "
                "use features='series'"
            )
        values = pa.concat_arrays(rows)
        if output == "numpy":
            return values.to_numpy(zero_copy_only=False).reshape(len(rows), length), y.to_numpy(zero_copy_only=False)
        return pa.FixedSizeListArray.from_arrays(values, length), y

    def _rows_and_targets(self, resolved: type[Task]) -> tuple[list[pa.Array], list[object]]:
        """Pair each matched record's values with its task's target, for :meth:`to_features_and_targets`.

        Args:
            resolved: The task type to read targets from.

        Returns:
            The per-record Arrow value arrays and their targets, in record order.

        Raises:
            TimeFValidationError: This error occurs if a matched task carries no inline target. It
                also occurs if the dataset has no records, or if any record does not carry exactly
                one task of ``resolved``.
        """
        matched_by_record: dict[str, list[Task]] = {}
        for candidate in self.tasks_of(resolved):
            for record in candidate.inputs:
                matched_by_record.setdefault(record.id, []).append(candidate)
        rows: list[pa.Array] = []
        targets: list[object] = []
        for record in self._records:
            matched = matched_by_record.get(record.record_id) or []
            if len(matched) != 1:
                raise TimeFValidationError(
                    f"record {record.record_id!r} carries {len(matched)} {resolved.__name__} tasks; "
                    "to_features_and_targets needs exactly one per record"
                )
            inline = matched[0].targets
            if inline is None or len(inline) != 1 or not isinstance(inline[0], (str, int, float, bool)):
                raise TimeFValidationError(
                    f"{resolved.__name__} {matched[0].id!r} needs exactly one scalar target to use as y"
                )
            rows.append(record.to_arrow())  # Arrow straight from the loader, no NumPy copy
            targets.append(inline[0])
        if not rows:
            raise TimeFValidationError(f"dataset has no records to build {resolved.__name__} features from")
        return rows, targets

    def _infer_target_task(self) -> type[Task]:
        """Infer the dataset's sole task type that carries an inline target, for :meth:`to_features_and_targets`.

        Returns:
            The single task class whose instances carry inline targets.

        Raises:
            TimeFValidationError: If the dataset has zero or several such task types. In that
                case, pass ``task=`` instead.
        """
        kinds = {
            type(task)
            for task in self._tasks
            if task.targets is not None
            and len(task.targets) == 1
            and isinstance(task.targets[0], (str, int, float, bool))
        }
        if len(kinds) == 1:
            return kinds.pop()
        names = ", ".join(sorted(kind.__name__ for kind in kinds)) or "(none)"
        raise TimeFValidationError(f"pass task= to to_features_and_targets; dataset has target task types: {names}")

    def describe(self, *, rows: int = 5, file: TextIO | None = None) -> None:
        """Print a plain-text summary of the dataset: its identity, counts, specs and columns, and a record preview.

        This method works like pandas' ``describe`` and ``info`` methods. The preview reads only
        span metadata, not series values. The method checks value dtypes from one series per
        spec. This method works even before :meth:`derive_schema` runs, because it computes
        everything from the records.

        Args:
            rows: The number of records to show in the preview.
            file: Where to write the output. The default is ``sys.stdout``.
        """
        print(describe_text(self, rows=rows), file=file or sys.stdout)
