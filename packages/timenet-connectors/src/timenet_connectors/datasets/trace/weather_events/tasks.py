"""TRACE's tasks: event type, retrieval, and forecasting, each built as the TimeNet task it becomes."""

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
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.trace.weather_events.keys import AnnotationKey
from timenet_connectors.datasets.trace.weather_events.records import TraceWindow
from timenet_connectors.datasets.trace.weather_events.release import TraceSplit
from timenet_connectors.datasets.trace.weather_events.retrieval import EventReport, RetrievalRow
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel


@dataclass(frozen=True)
class TraceTask(DatasetTask):
    """One task of one published window, on the Record that holds the window's measurements."""

    window: TraceWindow
    record: Record

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split of the window's file."""
        return self.window.ref.file.split.timenet_split


@dataclass(frozen=True)
class TraceEventTypeTask(TraceTask):
    """A classified window's task: name the event that followed the window."""

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    report: EventReport
    """The event's report, whose event type RetrievalIndex.from_files checked is the window's."""

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the event report as its rationale."""
        event_type = self.report.event_type
        return ClassificationTask(
            inputs=(self.record,),
            targets=(event_type.label,),
            split=self.split,
            target_schema=AnnotationKey.EVENT_TYPE_VOCABULARY,
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
                    key=AnnotationKey.CHANNEL_DESCRIPTION,
                    value=description.channels.sentence(channel),
                    description="The authors' generated description of one channel over this window",
                    metadata={"channel": channel.value},
                )
            )
        return task


@dataclass(frozen=True)
class TraceForecastTask(TraceTask):
    """A forecasting window's task: forecast its last ``FORECAST_HORIZON`` values from the rest."""

    TIMENET_TYPE: ClassVar[type[Task]] = ForecastingTask

    def to_timenet_task(self) -> ForecastingTask:
        """Return the forecasting task, whose scope is the window before its target values."""
        target = self.window.forecast_target_span()
        return ForecastingTask(
            inputs=(self.record,),
            # Offsets count from the window's first timestamp. The model sees the window before the target.
            scope=TimeInterval(start_us=0, end_us=target.start_us),
            targets=(target,),
            split=self.split,
            metadata=self.window.ref.provenance,
        )
