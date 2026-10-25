"""Convert TRACE's weather windows into TimeF: Records of distinct measurements, and tasks.

The archive's JSON files hold 74,337 windows under the authors' splits. Windows with the same
station, mode, timestamps and values become one Record (63,366), and every published window keeps
its own tasks: event type, retrieval, and forecasting. The README explains the source, the tasks,
local clock time, and the leaks in the authors' splits.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from functools import partial
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.types import Annotation, ClassificationTask, ForecastingTask, TSCorrespondenceTask
from timenet_connectors.datasets.trace.weather_events.records import (
    IndexedWindow,
    Measurement,
    WindowLoader,
    index_windows,
    make_record,
)
from timenet_connectors.datasets.trace.weather_events.source import (
    DISTINCT_WINDOWS,
    EVENT_SCHEMA,
    NO_EVENT,
    SOURCE_FILES,
    SOURCE_SHA256,
    SOURCE_URL,
    EventType,
    TraceFileRef,
    split_file_path,
)
from timenet_connectors.datasets.trace.weather_events.tasks import check_retrieval, iter_tasks
from timenet_connectors.download import ensure_archive


class TraceWeatherEventsConnector(BaseConnector[TraceFileRef]):
    """Build TRACE's distinct measurement windows, with one task per published event, description, and forecast."""

    async def download_async(self, cache_dir: Path) -> list[TraceFileRef]:  # noqa: PLR6301 — BaseConnector interface
        """Download and extract the pinned Drive archive once, then locate its split files.

        Returns:
            One reference per JSON window file and retrieval parquet file, with its pinned counts.

        Raises:
            TimeNetDownloadError: If the archive lacks a split file.
        """
        root = await ensure_archive(SOURCE_URL, cache_dir, filename="trace_dataset.zip", sha256=SOURCE_SHA256)
        # The sha256 pins the layout: every split file sits under the archive's top-level dataset/.
        refs = [
            TraceFileRef(root / split_file_path(task, split), task, split, expected)
            for (task, split), expected in SOURCE_FILES.items()
        ]
        missing = [str(ref.path) for ref in refs if not ref.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"TRACE archive is missing {missing}")
        return refs

    def convert(self, raw_refs: list[TraceFileRef]) -> TimeFDataset:
        """Group the windows into lazy Records, check the retrieval rows, and stream the tasks.

        The references must be the archive's eight split files, each once. Every retrieval row must
        belong to a window, and every event-type task gets its event report as rationale.

        Returns:
            A dataset with one Record per distinct measurement window and every kind of task.

        Raises:
            TimeFValidationError: If references, rows, reports, descriptions, or the pinned inventory
                are invalid.
        """
        if sorted((ref.task, ref.split) for ref in raw_refs) != sorted(SOURCE_FILES):
            raise TimeFValidationError(f"TRACE requires each of the split files {sorted(SOURCE_FILES)} once")
        retrieval_files = [ref for ref in raw_refs if ref.task == "retrieval"]
        measurements: dict[str, Measurement] = {}
        windows: dict[str, IndexedWindow] = {}
        for file in raw_refs:
            if file.task != "retrieval":
                index_windows(file, measurements, windows)
        # Fixtures carry no pinned counts, so only the pinned archive checks the distinct count.
        if all(ref.expected is not None for ref in raw_refs) and len(measurements) != DISTINCT_WINDOWS:
            raise TimeFValidationError(
                f"expected {DISTINCT_WINDOWS} distinct measurement windows, found {len(measurements)}"
            )
        test_keys = check_retrieval(retrieval_files, windows)

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key="no_event_label", value=NO_EVENT, id="trace-no-event-label"))
        dataset.annotate(
            Annotation(key="event_type_vocabulary", value=[event.label for event in EventType], id=EVENT_SCHEMA)
        )
        loader = WindowLoader()
        for measurement in measurements.values():
            dataset.add_record(record=make_record(measurement, loader))
        # The authors rank each test description against every test window (context_align_task.py).
        # Copies of one measurement share a Record, so the pool holds each Record once.
        test_pool = tuple({windows[key].record.record_id: windows[key].record for key in test_keys}.values())
        dataset.set_task_stream(
            [ClassificationTask, TSCorrespondenceTask, ForecastingTask],
            partial(iter_tasks, tuple(retrieval_files), windows, test_pool),
        )
        return dataset


CONNECTOR = TraceWeatherEventsConnector
