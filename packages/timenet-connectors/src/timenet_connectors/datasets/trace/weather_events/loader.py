"""Load TRACE's archive into its windows, retrieval rows, Records, and tasks."""

from collections.abc import Iterable, Iterator
from itertools import chain

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, ForecastingTask, Split, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events.archive import TraceArchive, TraceFileRef, TraceSplit
from timenet_connectors.datasets.trace.weather_events.index import RetrievalIndex, TraceRecords, WindowIndex
from timenet_connectors.datasets.trace.weather_events.records import WindowLoader
from timenet_connectors.datasets.trace.weather_events.tasks import (
    TASK_TYPES,
    TraceEventTypeTask,
    TraceForecastTask,
    TraceRetrievalTask,
    TraceTask,
)
from timenet_connectors.datasets.trace.weather_events.vocabulary import EVENT_SCHEMA, EventLabel


class TraceDatasetLoader:
    """TRACE's archive, read and checked: its windows, its retrieval rows, one Record per measurement, and tasks.

    TRACE defines three tasks, one per published window or row, each under the authors' split:

    - Event type: a classified ``pretrain`` window must name the event that followed it.
    - Retrieval: a retrieval row's generated description must find its window's Record.
    - Forecasting: a ``forecasting`` window's last values must follow from the rest.
    """

    TASK_TYPES = TASK_TYPES
    """The TimeNet task types ``timenet_tasks`` yields."""

    def __init__(self, windows: WindowIndex, retrieval_rows: RetrievalIndex, records: TraceRecords) -> None:
        self._windows = windows
        self._retrieval_rows = retrieval_rows
        self._records = records
        # The authors rank each test description against every test window (context_align_task.py).
        # Copies of one measurement share a Record, so each candidate Record appears once.
        self._retrieval_test_candidates = records.distinct_of(retrieval_rows.test_windows())

    @classmethod
    def load(cls, refs: Iterable[TraceFileRef]) -> "TraceDatasetLoader":
        """Read and check the archive's split files, and build one lazy Record per measurement.

        The windows are grouped into measurements, then every retrieval row is matched to its window.
        An invalid file set, window, retrieval row, or pinned count raises ``TimeFValidationError``.

        Returns:
            The loaded archive.
        """
        archive = TraceArchive.from_refs(refs)
        windows = WindowIndex.from_files(archive.window_files)
        archive.check_distinct_measurements(windows.measurement_count)
        retrieval_rows = RetrievalIndex.from_files(archive.retrieval_files, windows)
        records = windows.build_records(WindowLoader())
        return cls(windows, retrieval_rows, records)

    def dataset_annotations(self) -> tuple[Annotation, ...]:  # noqa: PLR6301 — the loader interface
        """Return the dataset's annotations of the event labels: the source's no-event label, and the vocabulary.

        The event-type task names the vocabulary's key as its ``target_schema``.

        Returns:
            The ``no_event_label`` and ``event_type_vocabulary`` annotations.
        """
        return (
            Annotation(key="no_event_label", value=EventLabel.NO_EVENT.value),
            Annotation(key=EVENT_SCHEMA, value=EventLabel.vocabulary()),
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per distinct measurement, whose values stay in the window files until written."""
        return iter(self._records)

    def get_tasks(self, *, split: Split | None = None, kind: type[TraceTask] | None = None) -> Iterator[TraceTask]:
        """Every TRACE task, optionally only one split and/or one kind, in stream order.

        Each call rereads the retrieval files, so the tasks can be iterated more than once. A retrieval
        task's split is its retrieval file's, which is the authors' own: most retrieval test tasks are
        on windows from the ``pretrain`` train and val files.

        Args:
            split: Only the tasks of this split, if given.
            kind: Only the tasks of this TRACE task type, if given.

        Yields:
            Each retrieval row's event-type task, if its window is classified, then its retrieval task.
            Then each forecasting task.
        """
        # Each kind reads only what it needs: forecasts never reread the retrieval files.
        row_tasks = () if kind is TraceForecastTask else self._retrieval_row_tasks()
        forecasts = self._forecast_tasks() if kind in {None, TraceForecastTask} else ()
        for task in chain(row_tasks, forecasts):
            if (kind is None or isinstance(task, kind)) and (split is None or task.split is split):
                yield task

    def timenet_tasks(self) -> Iterator[ClassificationTask | TSCorrespondenceTask | ForecastingTask]:
        """Return every task as TimeNet stores it, in the order of ``get_tasks``."""
        return (task.to_timenet_task() for task in self.get_tasks())

    def _retrieval_row_tasks(self) -> Iterator[TraceEventTypeTask | TraceRetrievalTask]:
        for file, window, row in self._retrieval_rows:
            record = self._records.of(window)
            # A row has a report exactly when its window is classified, as RetrievalIndex.from_files checked.
            if row.event_report is not None:
                yield TraceEventTypeTask(window, record, row.event_report)
            # Test tasks rank the retrieval test windows' Records; train tasks name no candidates.
            candidates = self._retrieval_test_candidates if file.split is TraceSplit.TEST else ()
            yield TraceRetrievalTask(window, record, row, file.split, candidates)

    def _forecast_tasks(self) -> Iterator[TraceForecastTask]:
        for window in self._windows:
            if window.is_eligible_for_forecasting():
                yield TraceForecastTask(window, self._records.of(window))
