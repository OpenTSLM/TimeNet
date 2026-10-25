"""Check the retrieval rows against the windows, and build TRACE's event-type, retrieval, and forecasting tasks."""

from collections.abc import Iterator

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, ForecastingTask, Split, TimeInterval, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events.records import IndexedWindow
from timenet_connectors.datasets.trace.weather_events.source import (
    CHANNEL_NAMES,
    EVENT_SCHEMA,
    Description,
    EventReport,
    EventType,
    TraceFileRef,
    check_inventory,
    retrieval_rows,
)


def _check_report(window: IndexedWindow, report: EventReport | None) -> None:
    """Check that a window has an event report exactly when it is classified, and the right one.

    Raises:
        TimeFValidationError: If the window's event report is missing, mismatched, or unexpected.
    """
    if window.event_type is not None:
        if report is None or report.event_type != window.event_type:
            raise TimeFValidationError(f"{window.ref.context}: no event report with event_type {window.event_type}")
    elif report is not None:
        raise TimeFValidationError(f"{window.ref.context}: event report for a window without an event-type task")


def check_retrieval(files: list[TraceFileRef], windows: dict[str, IndexedWindow]) -> list[str]:
    """Check every retrieval row against the indexed windows, keeping no descriptions or reports.

    Every row must belong to a window, every classified window must have its event report, and every
    report a classified window. The task stream rereads the rows and relies on these checks.

    Returns:
        The window keys of the test file's rows, in file order.

    Raises:
        TimeFValidationError: If a row, a report, or a file's pinned inventory is invalid.
    """
    test_keys: list[str] = []
    seen: set[str] = set()
    orphans: list[str] = []
    for file in files:
        rows = events = 0
        for key, _, report in retrieval_rows(file):
            if key in seen:
                raise TimeFValidationError(f"{file.path}: window id {key} appears in more than one row")
            seen.add(key)
            if file.split == "test":
                test_keys.append(key)
            rows += 1
            events += report is not None
            if key in windows:
                _check_report(windows[key], report)
            else:
                orphans.append(key)
        check_inventory(file, rows, events)
    if orphans:
        raise TimeFValidationError(f"{len(orphans)} retrieval rows have no window, such as {min(orphans)}")
    for window in windows.values():
        if window.event_type is not None and window.ref.key not in seen:
            raise TimeFValidationError(f"{window.ref.context}: classified window has no retrieval row")
    return test_keys


def _event_task(window: IndexedWindow, report: EventReport) -> ClassificationTask:
    """Build a classified window's event-type task.

    Returns:
        The task, with the event report as its rationale.
    """
    # check_retrieval checked that the report's event type is the window's.
    event_type = EventType(report.event_type)
    return ClassificationTask(
        id=f"{window.measurement.record_id}-event-type-{window.ref.task_suffix}",
        inputs=(window.record,),
        targets=(event_type.label,),
        split=window.ref.file.timenet_split,
        target_schema=EVENT_SCHEMA,
        # The report names the target and describes the event after the window, so it is
        # supervision only, never an input.
        rationale=report.text,
        metadata=window.ref.provenance | {"event_type_index": event_type.value, "event_id": report.event_id},
    )


def _retrieval_task(
    window: IndexedWindow,
    split: Split,
    description: Description,
    report: EventReport | None,
    pool: tuple[Record, ...],
) -> TSCorrespondenceTask:
    """Build a retrieval row's task: its description must find the window's Record (paper Stage 2).

    Returns:
        The task, under the retrieval file's split, with the description's channel sentences as its
        annotations.
    """
    task = TSCorrespondenceTask(
        id=f"{window.measurement.record_id}-retrieval-{window.ref.task_suffix}",
        # The query is text only: the description as the authors' generate_dsp composes it.
        prompt=description.text,
        targets=(window.record,),
        # Empty for train: the authors draw training negatives from each batch, not from a pool.
        candidate_records=pool,
        split=split,
        # The event_id joins the row to its event-type task, whose rationale is the row's event text.
        metadata=window.ref.provenance | ({} if report is None else {"event_id": report.event_id}),
    )
    # The authors embed each channel sentence on its own (generate_channel_description) and align it
    # with that channel's embedding. The sentences belong to this row, and copies of one measurement
    # have their own, so they sit on the task, not on the shared Record's Signals.
    for name, sentence in zip(CHANNEL_NAMES, description.channels, strict=True):
        task.annotate(
            Annotation(
                key="channel_description",
                value=sentence,
                description="The authors' generated description of one channel over this window",
                id=f"{task.id}-{name.replace('_', '-')}",
                metadata={"channel": name},
            )
        )
    return task


def _forecast_task(window: IndexedWindow, forecast: TimeInterval) -> ForecastingTask:
    """Build a forecasting window's task.

    Returns:
        The task, whose input is the window up to its last ``FORECAST_HORIZON`` values and whose
        target is those values.
    """
    return ForecastingTask(
        id=f"{window.measurement.record_id}-forecast-{window.ref.task_suffix}",
        inputs=(window.record,),
        # Offsets count from the window's first timestamp. The model sees the window before the target.
        scope=TimeInterval(start_us=0, end_us=forecast.start_us),
        targets=(forecast,),
        split=window.ref.file.timenet_split,
        metadata=window.ref.provenance,
    )


def iter_tasks(
    files: tuple[TraceFileRef, ...], windows: dict[str, IndexedWindow], test_pool: tuple[Record, ...]
) -> Iterator[ClassificationTask | TSCorrespondenceTask | ForecastingTask]:
    """Rebuild every task from the window index and the retrieval files, which ``convert`` checked.

    Each call rereads the retrieval files, so the writer can iterate more than once.

    Yields:
        Each row's event-type task, if its window is classified, then its retrieval task. Then each
        forecasting task.
    """
    for file in files:
        # Test tasks rank the test windows' Records; train tasks name no pool.
        pool = test_pool if file.split == "test" else ()
        for key, description, report in retrieval_rows(file):
            window = windows[key]
            # A row has a report exactly when its window is classified, as check_retrieval checked.
            if report is not None:
                yield _event_task(window, report)
            yield _retrieval_task(window, file.timenet_split, description, report, pool)
    for window in windows.values():
        if window.forecast is not None:
            yield _forecast_task(window, window.forecast)
