"""Load TRACE's archive into its windows, retrieval rows, Records, and tasks."""

from collections.abc import Iterable, Iterator
from typing import Self

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, ForecastingTask, Task, TSCorrespondenceTask
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.trace.weather_events.index import RetrievalIndex, TraceRecords, WindowIndex
from timenet_connectors.datasets.trace.weather_events.keys import AnnotationKey
from timenet_connectors.datasets.trace.weather_events.records import WindowLoader
from timenet_connectors.datasets.trace.weather_events.release import TraceArchive, TraceFileRef, TraceSplit
from timenet_connectors.datasets.trace.weather_events.tasks import event_type_task, forecast_task, retrieval_task
from timenet_connectors.datasets.trace.weather_events.vocabulary import EventLabel


class TraceDatasetLoader(DatasetLoader):
    """TRACE's archive, read and checked: its windows, its retrieval rows, one Record per measurement, and tasks.

    TRACE defines three tasks, one per published window or row, each under the authors' split. Each
    has its own method:

    - Event type (:meth:`event_type_tasks`): a classified ``pretrain`` window must name the event
      that followed it.
    - Retrieval (:meth:`retrieval_tasks`): a retrieval row's generated description must find its
      window's Record.
    - Forecasting (:meth:`forecast_tasks`): a ``forecasting`` window's last values must follow from
      the rest.
    """

    TASK_TYPES = (ClassificationTask, TSCorrespondenceTask, ForecastingTask)

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

    def tasks(self) -> Iterator[Task]:
        """Return every TRACE task, in stream order.

        Each call rereads the retrieval files, so the tasks can be iterated more than once.

        Yields:
            The event-type tasks, then the retrieval tasks, then the forecasting tasks.
        """
        yield from self.event_type_tasks()
        yield from self.retrieval_tasks()
        yield from self.forecast_tasks()

    def event_type_tasks(self) -> Iterator[ClassificationTask]:
        """Return the event-type tasks.

        Yields:
            One classification task for each retrieval row whose window is classified.
        """
        for _, window, row in self._retrieval_rows:
            # A row has a report exactly when its window is classified, as RetrievalIndex.from_files checked.
            if row.event_report is not None:
                yield event_type_task(window, self._records.of(window), row.event_report)

    def retrieval_tasks(self) -> Iterator[TSCorrespondenceTask]:
        """Return the retrieval tasks.

        A retrieval task's split is its retrieval file's, which is the authors' own: most retrieval test
        tasks are on windows from the ``pretrain`` train and val files.

        Yields:
            One correspondence task for each retrieval row.
        """
        for file, window, row in self._retrieval_rows:
            # Test tasks rank the retrieval test windows' Records; train tasks name no candidates.
            candidates = self._retrieval_test_candidates if file.split is TraceSplit.TEST else ()
            yield retrieval_task(window, self._records.of(window), row, file.split, candidates)

    def forecast_tasks(self) -> Iterator[ForecastingTask]:
        """Return the forecasting tasks.

        Yields:
            One forecasting task for each forecasting window long enough for one.
        """
        for window in self._windows:
            if window.is_eligible_for_forecasting():
                yield forecast_task(window, self._records.of(window))
