"""Convert TRACE's weather windows into TimeF: one Record per distinct measurement, and three task types.

The archive holds JSON window files (``pretrain`` and ``forecasting``) and parquet retrieval files, each
under the authors' train, val and test splits. Windows with the same station, sampling, timestamps and
values become one Record, and each window or retrieval row that qualifies keeps its own task: event type,
retrieval or forecasting. README.md explains the source, the tasks, local clock time, and the leaks in
the authors' splits.

The connector reads each file twice. Pass one checks every window and retrieval row and keeps only small
facts and digests. Pass two runs when the dataset is written: a Signal rereads its window, and the task
stream rereads the retrieval rows. Both passes must see the same bytes, so a rewritten file fails loudly.

Metadata is provenance and never model input. An annotation is model input only for a task that names it
in ``input_annotations``.

The modules, in reading order:

- ``archive``: the source archive and its eight split files.
- ``vocabulary``: what published values mean: event labels, samplings, channels and their units.
- ``fields`` and ``window_rows``: read and check one published window.
- ``measurements``: group windows into distinct measurements, and build their Records.
- ``retrieval``: read and check the retrieval rows against the windows.
- ``tasks``: build the three task types.
"""

from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet_connectors.datasets.trace.weather_events.archive import (
    ARCHIVE_FILENAME,
    FILES,
    SOURCE_SHA256,
    SOURCE_URL,
    TraceFile,
    archive_path,
)
from timenet_connectors.datasets.trace.weather_events.measurements import WindowIndex, WindowReader
from timenet_connectors.datasets.trace.weather_events.retrieval import RetrievalIndex
from timenet_connectors.datasets.trace.weather_events.tasks import TASK_TYPES, TaskStream
from timenet_connectors.datasets.trace.weather_events.vocabulary import DATASET_ANNOTATIONS
from timenet_connectors.download import ensure_archive


class TraceWeatherEventsConnector(BaseConnector[TraceFile]):
    """Build TRACE's distinct measurement windows, with one task per published event, description and forecast."""

    # ruff: ignore[no-self-use] (BaseConnector interface)
    async def download_async(self, cache_dir: Path) -> list[TraceFile]:
        """Download and extract the pinned Drive archive once, then locate its split files.

        Returns:
            One reference per JSON window file and retrieval parquet file.

        Raises:
            TimeNetDownloadError: If the archive lacks a split file.
        """
        root = await ensure_archive(SOURCE_URL, cache_dir, filename=ARCHIVE_FILENAME, sha256=SOURCE_SHA256)
        # The sha256 pins the layout: every split file sits under the archive's top-level dataset/.
        files = [TraceFile(root / archive_path(kind, split), kind, split) for kind, split in FILES]
        missing = [str(file.path) for file in files if not file.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"TRACE archive is missing {missing}")
        return files

    def convert(self, raw_refs: list[TraceFile]) -> TimeFDataset:
        """Convert the archive's split files into one Record per distinct measurement, and its tasks.

        Pass one reads the six window files, groups identical windows into one measurement each, and
        checks every row of the two retrieval files against its window. The Records hold lazy signals
        whose values stay in the window files until written. The tasks stream, so the dataset does not
        hold them: each qualifying window keeps its own event-type, retrieval or forecasting task, under
        the authors' split. The files must stay unchanged until the dataset is written.

        Args:
            raw_refs: The archive's eight split files, each once.

        Returns:
            A dataset with the event-type vocabulary, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If the file set, a window or a retrieval row is invalid.
        """
        if sorted((file.kind, file.split) for file in raw_refs) != sorted(FILES):
            names = [archive_path(kind, split) for kind, split in FILES]
            raise TimeFValidationError(f"TRACE requires each of the split files {names} once")
        by_key = {(file.kind, file.split): file for file in raw_refs}
        files = [by_key[key] for key in FILES]

        windows = WindowIndex()
        for file in files:
            if file.kind.holds_windows:
                windows.add_file(file)
        retrieval = RetrievalIndex.read([file for file in files if not file.kind.holds_windows], windows)
        reader = WindowReader()
        records = {measurement: measurement.to_record(reader) for measurement in windows.measurements.values()}

        dataset = TimeFDataset(metadata=self.metadata())
        for annotation in DATASET_ANNOTATIONS:
            dataset.annotate(annotation)
        for record in records.values():
            dataset.add_record(record=record)
        dataset.set_task_stream(TASK_TYPES, TaskStream(windows, retrieval, records).tasks)
        return dataset


CONNECTOR = TraceWeatherEventsConnector
