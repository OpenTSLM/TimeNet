"""The declarative objects a builder writes: ``Task -> Record -> Source -> Signal`` plus annotations.

These are the in-memory objects of the proposed hierarchy. They hold no storage detail: no ids for
annotation content, no occurrence rows, no reverse indexes. The writer derives all of that when it
compiles the hierarchy into the control-plane database.

``Source`` nests, so one record can describe a bedside monitor holding an ECG and a thermometer, or
an IMU holding three sensor packages. A ``Signal`` is a leaf: one sequence of values, one time axis,
one spec.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from timenet.dataset.axis import TimeAxis
from timenet.errors import TimeFValidationError
from timenet.types import DatasetMetadata, TimeSeriesSpec, new_id


SpanType = str
"""How an annotation places itself in time: ``static``, ``point``, or ``interval``."""


@dataclass(frozen=True)
class Annotation:
    """One statement about an object, and where in time it applies.

    The same annotation object can be attached to many objects. The writer stores its payload once
    and records one occurrence for each attachment, so ``patient_sex=male`` on 100,000 records costs
    one payload and 100,000 short rows.
    """

    name: str
    """What the annotation says something about, for example ``patient_sex`` or ``lead_status``."""
    value: str
    """The statement itself, for example ``male`` or ``Lead fell off``."""
    unit: str | None = None
    """The unit of ``value``, when it is a quantity."""
    span_type: SpanType = "static"
    """``static`` for a whole-object statement, ``point`` for an instant, ``interval`` for a range."""
    start_us: int | None = None
    """The instant, or the start of the interval, in microseconds from the record's relative zero."""
    end_us: int | None = None
    """The end of the interval, in microseconds. It is ``None`` unless ``span_type`` is ``interval``."""
    provenance: str | None = None
    """Who or what made this statement."""
    confidence: float | None = None
    """How sure the source of the statement is, between 0 and 1."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Anything else the occurrence carries."""

    @classmethod
    def static(cls, *, name: str, value: Any, unit: str | None = None, **kwargs: Any) -> Self:
        """Build an annotation that applies to a whole object, with no place in time.

        Args:
            name: The annotation's name.
            value: The annotation's value. It is stored as text.
            unit: The unit of ``value``, when it is a quantity.
            kwargs: Any other field, such as ``provenance`` or ``confidence``.

        Returns:
            The annotation.
        """
        return cls(name=name, value=str(value), unit=unit, span_type="static", **kwargs)

    @classmethod
    def point(cls, *, name: str, value: Any, at_us: int, unit: str | None = None, **kwargs: Any) -> Self:
        """Build an annotation that applies at one instant.

        Args:
            name: The annotation's name.
            value: The annotation's value. It is stored as text.
            at_us: The instant, in microseconds from the record's relative zero.
            unit: The unit of ``value``, when it is a quantity.
            kwargs: Any other field, such as ``provenance`` or ``confidence``.

        Returns:
            The annotation.
        """
        return cls(name=name, value=str(value), unit=unit, span_type="point", start_us=at_us, **kwargs)

    @classmethod
    def interval(
        cls, *, name: str, value: Any, start_us: int, end_us: int, unit: str | None = None, **kwargs: Any
    ) -> Self:
        """Build an annotation that applies over a time range.

        Args:
            name: The annotation's name.
            value: The annotation's value. It is stored as text.
            start_us: The start of the range, in microseconds from the record's relative zero.
            end_us: The end of the range, in microseconds.
            unit: The unit of ``value``, when it is a quantity.
            kwargs: Any other field, such as ``provenance`` or ``confidence``.

        Returns:
            The annotation.

        Raises:
            TimeFValidationError: If ``end_us`` is before ``start_us``.
        """
        if end_us < start_us:
            raise TimeFValidationError(f"interval annotation ends at {end_us} before it starts at {start_us}")
        return cls(
            name=name, value=str(value), unit=unit, span_type="interval", start_us=start_us, end_us=end_us, **kwargs
        )

    def content_key(self) -> tuple[str, str, str | None]:
        """Return what makes this annotation's payload reusable across occurrences.

        Returns:
            The name, value, and unit. Two annotations sharing this triple share one stored payload.
        """
        return (self.name, self.value, self.unit)


class _Annotatable:
    """Shared annotation list and ``annotate`` method for every object that can carry annotations."""

    annotations: list[Annotation]

    def annotate(self, annotation: Annotation) -> Self:
        """Attach one annotation to this object.

        Args:
            annotation: The annotation to attach.

        Returns:
            This object, so a caller can chain calls.
        """
        self.annotations.append(annotation)
        return self


@dataclass
class Signal(_Annotatable):
    """One sequence of values: the leaf of the hierarchy."""

    name: str
    """A human-readable name, for example ``acceleration_x``."""
    values: np.ndarray
    """The values themselves. The writer stores them in the values plane, not in the database."""
    time_axis: TimeAxis
    """How the values are placed in time. Several signals can share one axis object."""
    spec: TimeSeriesSpec
    """What the values mean: unit, dtype, and shape. Several signals can share one spec object."""
    time_offsets_us: np.ndarray | None = None
    """One time offset per value, for an irregular axis. It rides the values plane beside the values."""
    id: str = field(default_factory=new_id)
    """The signal's id. It is generated when the caller does not supply one."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Anything else about the signal, such as calibration or sensor orientation."""
    annotations: list[Annotation] = field(default_factory=list)
    """Annotations that apply to this signal."""


@dataclass
class Source(_Annotatable):
    """One data source: a device, a sensor, or an assembly holding other sources.

    A source nests. An IMU is a source holding an accelerometer, a gyroscope, and a magnetometer,
    each of which is a source holding three signals.
    """

    name: str
    """A human-readable name, for example ``IMU`` or ``Bedside monitor``."""
    sources: list["Source"] = field(default_factory=list)
    """Child sources. This is what makes the hierarchy recursive."""
    signals: list[Signal] = field(default_factory=list)
    """Signals this source produces directly."""
    id: str = field(default_factory=new_id)
    """The source's id. It is generated when the caller does not supply one."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Anything else about the source, such as manufacturer, model, or location."""
    annotations: list[Annotation] = field(default_factory=list)
    """Annotations that apply to this source."""

    def walk(self) -> "list[Source]":
        """Return this source and every source beneath it, parents before children.

        Returns:
            The sources in breadth-first order.
        """
        found: list[Source] = []
        queue: list[Source] = [self]
        while queue:
            current = queue.pop(0)
            found.append(current)
            queue.extend(current.sources)
        return found

    def select(self, *, signals: Sequence[Signal] = (), signal_names: Sequence[str] = ()) -> "_Selection":
        """Pick signals beneath this source so a caller can annotate all of them at once.

        Args:
            signals: Signal objects to select.
            signal_names: Names to select, matched against every signal beneath this source.

        Returns:
            A selection whose ``annotate`` attaches one annotation to each selected signal.

        Raises:
            TimeFValidationError: If a name in ``signal_names`` matches no signal.
        """
        picked = list(signals)
        if signal_names:
            beneath = [signal for source in self.walk() for signal in source.signals]
            by_name = {signal.name: signal for signal in beneath}
            for name in signal_names:
                if name not in by_name:
                    raise TimeFValidationError(f"no signal named {name!r} beneath source {self.name!r}")
                picked.append(by_name[name])
        return _Selection(picked)


@dataclass
class _Selection:
    """Several signals picked out of a source, so one annotation can reach all of them."""

    signals: list[Signal]

    def annotate(self, annotation: Annotation) -> "_Selection":
        """Attach one annotation to every selected signal.

        Args:
            annotation: The annotation to attach.

        Returns:
            This selection, so a caller can chain calls.
        """
        for signal in self.signals:
            signal.annotate(annotation)
        return self


@dataclass
class Record(_Annotatable):
    """One recording session: a patient's examination, or one run of a machine."""

    sources: list[Source] = field(default_factory=list)
    """The root sources of this session. Each one can hold further sources and signals."""
    id: str = field(default_factory=new_id)
    """The record's id. It is generated when the caller does not supply one."""
    start_time_us: int | None = None
    """The wall-clock time the session started, in Unix microseconds, when it is known."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Anything else about the session."""
    annotations: list[Annotation] = field(default_factory=list)
    """Annotations that apply to the whole session, such as the subject's sex or age."""

    def walk_sources(self) -> list[Source]:
        """Return every source in this record, parents before children.

        Returns:
            The sources in breadth-first order.
        """
        return [found for root in self.sources for found in root.walk()]

    def signals(self) -> list[Signal]:
        """Return every signal in this record.

        Returns:
            The signals, in the order their sources are walked.
        """
        return [signal for source in self.walk_sources() for signal in source.signals]


@dataclass(frozen=True)
class RecordRef:
    """A reference to a record by id, for a task built without the record object in hand.

    A streaming build yields its records and drops them, so by the time it builds the tasks the
    record objects are gone. Naming the id keeps the reference a record reference rather than
    degrading into a text item, and the write-time validation still rejects an id that names no
    record.
    """

    id: str


@dataclass
class Task(_Annotatable):
    """What a model is asked to do with some records, and what the right answer is."""

    prompt: str
    """What the model is supposed to do, as text."""
    inputs: list["Record | RecordRef | str"] = field(default_factory=list)
    """The records and text the task gives the model. Order is preserved."""
    target: list["Record | RecordRef | str"] = field(default_factory=list)
    """The desired output, as text and records. Order is preserved."""
    id: str = field(default_factory=new_id)
    """The task's id. It is generated when the caller does not supply one."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Anything else about the task, such as its type."""
    annotations: list[Annotation] = field(default_factory=list)
    """Annotations that apply to the task."""


@dataclass
class DeclarativeDataset:
    """The records and tasks a builder has assembled, ready for the writer to compile.

    A record can be added on its own. An unlabelled pretraining corpus is records with no tasks, and
    needs no invented prompt to exist.
    """

    metadata: DatasetMetadata
    """The dataset's identity: its ``org/name`` id, version, license, and domains."""
    records: list[Record] = field(default_factory=list)
    """Every record, including those only referenced by a task."""
    tasks: list[Task] = field(default_factory=list)
    """Every task."""
    annotations: list[Annotation] = field(default_factory=list)
    """Annotations that apply to the dataset as a whole."""

    def add_record(self, record: Record) -> Record:
        """Add a record to the dataset, ignoring a record already added.

        Args:
            record: The record to add.

        Returns:
            The record.
        """
        if not any(existing.id == record.id for existing in self.records):
            self.records.append(record)
        return record

    def add_task(self, task: Task) -> Task:
        """Add a task, adding any record it references that is not in the dataset yet.

        Args:
            task: The task to add.

        Returns:
            The task.
        """
        self.tasks.append(task)
        for item in (*task.inputs, *task.target):
            if isinstance(item, Record):
                self.add_record(item)
        return task

    def annotate(self, annotation: Annotation) -> "DeclarativeDataset":
        """Attach one annotation to the dataset as a whole.

        Args:
            annotation: The annotation to attach.

        Returns:
            This dataset, so a caller can chain calls.
        """
        self.annotations.append(annotation)
        return self
