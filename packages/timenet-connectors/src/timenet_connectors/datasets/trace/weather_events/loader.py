"""Load TRACE's archive into its windows, retrieval rows, Records, and tasks."""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Self

from timenet.dataset import Record
from timenet.types import Annotation
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.trace.weather_events.index import RetrievalIndex, TraceRecords, WindowIndex
from timenet_connectors.datasets.trace.weather_events.keys import AnnotationKey
from timenet_connectors.datasets.trace.weather_events.records import WindowLoader
from timenet_connectors.datasets.trace.weather_events.release import TraceArchive, TraceFileRef, TraceSplit
from timenet_connectors.datasets.trace.weather_events.tasks import (
    TraceEventTypeTask,
    TraceForecastTask,
    TraceRetrievalTask,
    TraceTask,
)
from timenet_connectors.datasets.trace.weather_events.vocabulary import EventLabel


class TraceDatasetLoader(DatasetLoader[TraceTask]):
    """TRACE's archive, read and checked: its windows, its retrieval rows, one Record per measurement, and tasks.

    TRACE defines three tasks, one per published window or row, each under the authors' split:

    - Event type (:class:`TraceEventTypeTask`): a classified ``pretrain`` window must name the event
      that followed it.
    - Retrieval (:class:`TraceRetrievalTask`): a retrieval row's generated description must find its
      window's Record.
    - Forecasting (:class:`TraceForecastTask`): a ``forecasting`` window's last values must follow
      from the rest.

    Select one kind, or one kind of one split, with ``tasks_of``, such as
    ``tasks_of(TraceRetrievalTask, split=Split.TRAIN)``.
    """

    TASK_KINDS = (TraceEventTypeTask, TraceRetrievalTask, TraceForecastTask)

    def __init__(self, windows: WindowIndex, retrieval_rows: RetrievalIndex, records: TraceRecords) -> None:
        self._windows = windows
        self._retrieval_rows = retrieval_rows
        self._records = records
        # The authors rank each test description against every test window (context_align_task.py).
        # Copies of one measurement share a Record, so each candidate Record appears once.
        self._retrieval_test_candidates = records.distinct_of(retrieval_rows.test_windows())

    @classmethod
    def parse_from(cls, refs: Iterable[TraceFileRef]) -> Self:
        """Read and check the archive's split files, and build one lazy Record per measurement.

        The windows are grouped into measurements, then every retrieval row is matched to its window.
        An invalid file set, window, retrieval row, or pinned count raises ``TimeFValidationError``.

        Args:
            refs: The archive's eight split files, each once.

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
            Annotation(key=AnnotationKey.NO_EVENT_LABEL, value=EventLabel.NO_EVENT.value),
            Annotation(key=AnnotationKey.EVENT_TYPE_VOCABULARY, value=EventLabel.vocabulary()),
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per distinct measurement, whose values stay in the window files until written."""
        return iter(self._records)

    def tasks(self) -> Iterator[TraceTask]:
        """Return every TRACE task, in stream order.

        Each call rereads the retrieval files, so the tasks can be iterated more than once. A retrieval
        task's split is its retrieval file's, which is the authors' own: most retrieval test tasks are
        on windows from the ``pretrain`` train and val files.

        Returns:
            Each retrieval row's event-type task, if its window is classified, then its retrieval task.
            Then each forecasting task.
        """
        return chain(self._retrieval_row_tasks(), self._forecast_tasks())

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
