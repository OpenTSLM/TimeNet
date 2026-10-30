"""TRACE's tasks: event type, retrieval, and forecasting, each built as a TimeNet task."""

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    ForecastingTask,
    TimeInterval,
    TSCorrespondenceTask,
)
from timenet_connectors.datasets.trace.weather_events.keys import AnnotationKey
from timenet_connectors.datasets.trace.weather_events.records import TraceWindow
from timenet_connectors.datasets.trace.weather_events.release import TraceSplit
from timenet_connectors.datasets.trace.weather_events.retrieval import EventReport, RetrievalRow
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel


def event_type_task(window: TraceWindow, record: Record, report: EventReport) -> ClassificationTask:
    """Return a classified window's task: name the event that followed the window.

    Args:
        window: The classified window, whose file gives the split.
        record: The Record that holds the window's measurements.
        report: The event's report, whose event type RetrievalIndex.from_files checked is the window's.

    Returns:
        The classification task, with the event report as its rationale.
    """
    event_type = report.event_type
    return ClassificationTask(
        inputs=(record,),
        targets=(event_type.label,),
        split=window.ref.file.split.timenet_split,
        target_schema=AnnotationKey.EVENT_TYPE_VOCABULARY,
        # The report names the target and describes the event after the window, so it is
        # supervision only, never an input.
        rationale=report.text,
        metadata=window.ref.provenance | {"event_type_index": event_type.value, "event_id": report.event_id},
    )


def retrieval_task(
    window: TraceWindow,
    record: Record,
    row: RetrievalRow,
    retrieval_split: TraceSplit,
    candidate_records: tuple[Record, ...],
) -> TSCorrespondenceTask:
    """Return a retrieval row's task: its description must find the window's Record (paper Stage 2).

    The candidates must stay series only. A description opens with its window's county and dates,
    which the candidates' ``county`` metadata and ``window_start``/``window_end`` annotations also
    hold, so naming either as an input would let a model match by string.

    Args:
        window: The row's window.
        record: The Record that holds the window's measurements.
        row: The retrieval row, with its generated description.
        retrieval_split: The split of the retrieval file, which is the authors' own and not the window file's.
        candidate_records: The Records a test task ranks; empty for train, whose negatives the authors
            draw per batch.

    Returns:
        The correspondence task, with the description's channel sentences as its annotations.
    """
    description, report = row.description, row.event_report
    task = TSCorrespondenceTask(
        # The query is text only: the description as the authors' generate_dsp composes it.
        prompt=description.text,
        targets=(record,),
        candidate_records=candidate_records,
        split=retrieval_split.timenet_split,
        # The event_id joins the row to its event-type task, whose rationale is the row's event text.
        metadata=window.ref.provenance | ({} if report is None else {"event_id": report.event_id}),
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


def forecast_task(window: TraceWindow, record: Record) -> ForecastingTask:
    """Return a forecasting window's task: forecast its last ``FORECAST_HORIZON`` values from the rest.

    Args:
        window: The forecasting window, whose file gives the split.
        record: The Record that holds the window's measurements.

    Returns:
        The forecasting task, whose scope is the window before its target values.
    """
    target = window.forecast_target_span()
    return ForecastingTask(
        inputs=(record,),
        # Offsets count from the window's first timestamp. The model sees the window before the target.
        scope=TimeInterval(start_us=0, end_us=target.start_us),
        targets=(target,),
        split=window.ref.file.split.timenet_split,
        metadata=window.ref.provenance,
    )
