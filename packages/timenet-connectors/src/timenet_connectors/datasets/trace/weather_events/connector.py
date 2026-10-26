"""Convert TRACE's weather windows into TimeF: Records of distinct measurements, and tasks.

The archive's JSON files hold 74,337 windows under the authors' splits. Windows with the same
station, mode, timestamps and values become one Record (63,366), and every TraceWindow keeps
its own tasks: event type, retrieval, and forecasting. The README explains the source, the tasks,
local clock time, and the leaks in the authors' splits.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet_connectors.datasets.trace.weather_events.archive import (
    SOURCE_FILES,
    SOURCE_SHA256,
    SOURCE_URL,
    TraceFileRef,
)
from timenet_connectors.datasets.trace.weather_events.loader import TraceDatasetLoader
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
            TraceFileRef(root / kind.file_path(split), kind, split, expected)
            for (kind, split), expected in SOURCE_FILES.items()
        ]
        missing = [str(ref.path) for ref in refs if not ref.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"TRACE archive is missing {missing}")
        return refs

    def convert(self, raw_refs: list[TraceFileRef]) -> TimeFDataset:
        """Convert the archive's split files into one Record per distinct measurement, and its tasks.

        ``TraceDatasetLoader`` reads the six window files, groups identical windows into one Measurement
        each, matches every row of the two retrieval files to its window, and builds one lazy Record
        per Measurement, whose values stay in the window files until written. Tasks are streamed, not
        held: each published window keeps its own event-type, retrieval, or forecasting task, under the
        authors' split.

        ``raw_refs`` must be the archive's eight split files, each once. An invalid file set, window,
        retrieval row, or pinned count raises ``TimeFValidationError``.

        Returns:
            A dataset with the label annotations, the Records, and a stream of every task.
        """
        loader = TraceDatasetLoader.load(raw_refs)
        dataset = TimeFDataset(metadata=self.metadata())
        for annotation in loader.dataset_annotations():
            dataset.annotate(annotation)
        for record in loader.records():
            dataset.add_record(record=record)
        dataset.set_task_stream(loader.TASK_TYPES, loader.timenet_tasks)
        return dataset


CONNECTOR = TraceWeatherEventsConnector
