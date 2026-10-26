"""TRACE's tasks: event type, retrieval, and forecasting, each built as the TimeNet task it becomes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    ForecastingTask,
    Split,
    Task,
    TimeInterval,
    TSCorrespondenceTask,
)
from timenet_connectors.datasets.trace.weather_events.archive import TraceSplit
from timenet_connectors.datasets.trace.weather_events.records import TraceWindow
from timenet_connectors.datasets.trace.weather_events.retrieval import EventReport, RetrievalRow
from timenet_connectors.datasets.trace.weather_events.vocabulary import EVENT_SCHEMA, Channel


@dataclass(frozen=True)
class TraceTask(ABC):
    """One task of one published window, on the Record that holds the window's measurements."""

    KIND: ClassVar[str]
    """The task's part of its id, such as ``event-type``."""
    TIMENET_TYPE: ClassVar[type[Task]]
    """The TimeNet task type ``to_timenet_task`` returns."""

    window: TraceWindow
    record: Record

    @property
    def id(self) -> str:
        """Return the task's id: its Record, its kind, and a hash of its window's id."""
        return f"{self.window.measurement.record_id}-{self.KIND}-{self.window.ref.task_suffix}"

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split of the window's file."""
        return self.window.ref.file.split.timenet_split

    @abstractmethod
    def to_timenet_task(self) -> ClassificationTask | TSCorrespondenceTask | ForecastingTask:
        """Return the task as TimeNet stores it."""


@dataclass(frozen=True)
class TraceEventTypeTask(TraceTask):
    """A classified window's task: name the event that followed the window."""

    KIND: ClassVar[str] = "event-type"
    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    report: EventReport
    """The event's report, whose event type RetrievalIndex.from_files checked is the window's."""

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the event report as its rationale."""
        event_type = self.report.event_type
        return ClassificationTask(
            id=self.id,
            inputs=(self.record,),
            targets=(event_type.label,),
            split=self.split,
            target_schema=EVENT_SCHEMA,
            # The report names the target and describes the event after the window, so it is
            # supervision only, never an input.
            rationale=self.report.text,
            metadata=self.window.ref.provenance
            | {"event_type_index": event_type.value, "event_id": self.report.event_id},
        )


@dataclass(frozen=True)
class TraceRetrievalTask(TraceTask):
    """A retrieval row's task: its description must find the window's Record (paper Stage 2).

    The candidates must stay series only. A description opens with its window's county and dates,
    which the candidates' ``county`` metadata and ``window_start``/``window_end`` annotations also
    hold, so naming either as an input would let a model match by string.
    """

    KIND: ClassVar[str] = "retrieval"
    TIMENET_TYPE: ClassVar[type[Task]] = TSCorrespondenceTask

    row: RetrievalRow
    retrieval_split: TraceSplit
    """The split of the retrieval file, which is the authors' own and not the window file's."""
    candidate_records: tuple[Record, ...]
    """The Records a test task ranks; empty for train, whose negatives the authors draw per batch."""

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split of the retrieval file, not the window's."""
        return self.retrieval_split.timenet_split

    def to_timenet_task(self) -> TSCorrespondenceTask:
        """Return the correspondence task, with the description's channel sentences as its annotations."""
        description, report = self.row.description, self.row.event_report
        task = TSCorrespondenceTask(
            id=self.id,
            # The query is text only: the description as the authors' generate_dsp composes it.
            prompt=description.text,
            targets=(self.record,),
            candidate_records=self.candidate_records,
            split=self.split,
            # The event_id joins the row to its event-type task, whose rationale is the row's event text.
            metadata=self.window.ref.provenance | ({} if report is None else {"event_id": report.event_id}),
        )
        # The authors embed each channel sentence on its own (generate_channel_description) and align it
        # with that channel's embedding. The sentences belong to this row, and copies of one measurement
        # have their own, so they sit on the task, not on the shared Record's Signals.
        for channel in Channel:
            task.annotate(
                Annotation(
                    key="channel_description",
                    value=description.channels.sentence(channel),
                    description="The authors' generated description of one channel over this window",
                    id=f"{task.id}-{channel.replace('_', '-')}",
                    metadata={"channel": channel.value},
                )
            )
        return task


@dataclass(frozen=True)
class TraceForecastTask(TraceTask):
    """A forecasting window's task: forecast its last ``FORECAST_HORIZON`` values from the rest."""

    KIND: ClassVar[str] = "forecast"
    TIMENET_TYPE: ClassVar[type[Task]] = ForecastingTask

    def to_timenet_task(self) -> ForecastingTask:
        """Return the forecasting task, whose scope is the window before its target values."""
        target = self.window.forecast_target_span()
        return ForecastingTask(
            id=self.id,
            inputs=(self.record,),
            # Offsets count from the window's first timestamp. The model sees the window before the target.
            scope=TimeInterval(start_us=0, end_us=target.start_us),
            targets=(target,),
            split=self.split,
            metadata=self.window.ref.provenance,
        )


# The TimeNet task types TRACE's tasks become, one per kind of TRACE task.
TASK_TYPES = tuple(task.TIMENET_TYPE for task in (TraceEventTypeTask, TraceRetrievalTask, TraceForecastTask))
