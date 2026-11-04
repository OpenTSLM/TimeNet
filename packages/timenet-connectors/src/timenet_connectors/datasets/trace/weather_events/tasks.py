"""TRACE's three tasks, one per published window or retrieval row, each under the authors' split.

- Event type: a classified ``pretrain`` window must name the event that followed it.
- Retrieval: a retrieval row's generated description must find its window's Record.
- Forecasting: a ``forecasting`` window's last values must follow from the rest.
"""

from collections.abc import Iterator, Mapping

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, ForecastingTask, Task, TimeInterval, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events.archive import SplitName
from timenet_connectors.datasets.trace.weather_events.measurements import Measurement, WindowIndex
from timenet_connectors.datasets.trace.weather_events.retrieval import RetrievalIndex
from timenet_connectors.datasets.trace.weather_events.vocabulary import EVENT_ID, AnnotationKey, Channel


TASK_TYPES = (ClassificationTask, TSCorrespondenceTask, ForecastingTask)
"""The task types the stream yields, in stream order."""


class TaskStream:
    """Build every task from the indexed windows and retrieval rows, rereading the retrieval files each time."""

    def __init__(self, windows: WindowIndex, retrieval: RetrievalIndex, records: Mapping[Measurement, Record]):
        self._windows = windows
        self._retrieval = retrieval
        self._records = records
        # The authors rank each test description against every test window (context_align_task.py).
        # Copies of one measurement share a Record, so each candidate Record appears once.
        test_records = (records[window.measurement] for window in retrieval.test_windows())
        self._test_candidates = tuple({id(record): record for record in test_records}.values())

    def tasks(self) -> Iterator[Task]:
        """Return every task: the event-type tasks, then the retrieval tasks, then the forecasting tasks.

        Each call rereads the retrieval files, so the stream can be iterated more than once.

        Yields:
            Each task, in the same order on each call.
        """
        yield from self._event_type_tasks()
        yield from self._retrieval_tasks()
        yield from self._forecast_tasks()

    def _event_type_tasks(self) -> Iterator[ClassificationTask]:
        """Yield one classification task for each retrieval row whose window is classified."""
        for _, window, row in self._retrieval.rows():
            # A row has a report exactly when its window is classified, as RetrievalIndex.read checked.
            report = row.report
            if report is None:
                continue
            yield ClassificationTask(
                inputs=(self._records[window.measurement],),
                targets=(report.event_type.text,),
                split=window.file.split.timef_split,
                target_schema=AnnotationKey.EVENT_TYPE_VOCABULARY,
                # The report names the target and describes the event after the window, so it is
                # supervision only, never an input.
                rationale=report.text,
                metadata=window.provenance() | {"event_type_index": report.event_type.value, EVENT_ID: report.event_id},
            )

    def _retrieval_tasks(self) -> Iterator[TSCorrespondenceTask]:
        """Yield one correspondence task for each retrieval row: its description must find the window's Record.

        A task's split is its retrieval file's, which is the authors' own: most retrieval test tasks are
        on windows from the ``pretrain`` train and val files. The query is text only. The candidates must
        stay series only: a description opens with its window's county and dates, which the Records'
        ``county`` metadata and ``window_start`` and ``window_end`` annotations also hold, so naming
        either as an input would let a model match by string.
        """
        for file, window, row in self._retrieval.rows():
            task = TSCorrespondenceTask(
                prompt=row.prompt,
                targets=(self._records[window.measurement],),
                # Test tasks rank the retrieval test windows' Records; train tasks name no candidates,
                # since the authors draw their negatives per batch.
                candidate_records=self._test_candidates if file.split is SplitName.TEST else (),
                split=file.split.timef_split,
                # The event id joins the row to its event-type task, whose rationale is the row's report.
                metadata=window.provenance() | ({} if row.report is None else {EVENT_ID: row.report.event_id}),
            )
            # The authors embed each channel sentence on its own (generate_channel_description) and align
            # it with that channel's embedding. The sentences belong to this row, and copies of one
            # measurement have their own, so they sit on the task, not on the shared Record's Signals.
            for channel, sentence in zip(Channel, row.sentences, strict=True):
                task.annotate(
                    Annotation(
                        key=AnnotationKey.CHANNEL_DESCRIPTION,
                        value=sentence,
                        description="The authors' generated description of one channel over this window",
                        metadata={"channel": channel.value},
                    )
                )
            yield task

    def _forecast_tasks(self) -> Iterator[ForecastingTask]:
        """Yield one forecasting task for each forecasting window long enough for one.

        The model sees the window before its last values, and the target is those values.
        """
        for window in self._windows.windows.values():
            target = window.forecast_target
            if target is not None:
                yield ForecastingTask(
                    inputs=(self._records[window.measurement],),
                    # Offsets count from the window's first timestamp.
                    scope=TimeInterval(start_us=0, end_us=target.start_us),
                    targets=(target,),
                    split=window.file.split.timef_split,
                    metadata=window.provenance(),
                )
