"""Convert TRACE's published windows into event, retrieval and forecasting tasks.

Conversion indexes validated windows and retrieval rows. Signals load lazily and tasks stream;
rereads check the measurements and task content against the initial scan. Each published window
keeps its own Record, while copies of the same measurements share a bounded channel reader.

See README.md for task mappings, local-clock timing and overlap in the authors' splits.
"""

from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.trace.weather_events.archive import (
    ARCHIVE_FILENAME,
    FILES,
    SOURCE_SHA256,
    SOURCE_URL,
    TraceFile,
    archive_path,
)
from timenet_connectors.datasets.trace.weather_events.records import build_record
from timenet_connectors.datasets.trace.weather_events.retrieval import RetrievalIndex
from timenet_connectors.datasets.trace.weather_events.source import WindowIndex, WindowReader
from timenet_connectors.datasets.trace.weather_events.tasks import TASK_TYPES, TaskStream
from timenet_connectors.download import ensure_archive


class TraceWeatherEventsConnector(BaseConnector[TraceFile]):
    """Build TRACE's weather windows, with one task per published event, description and forecast."""

    # ruff: ignore[no-self-use] (BaseConnector interface)
    async def download_async(self, cache_dir: Path) -> list[TraceFile]:
        """Download and extract the pinned Drive archive once, then locate its split files.

        Returns:
            One reference per JSON window file and retrieval parquet file.

        Raises:
            TimeNetConnectorError: If the archive lacks a split file.
        """
        root = await ensure_archive(SOURCE_URL, cache_dir, filename=ARCHIVE_FILENAME, sha256=SOURCE_SHA256)
        # The sha256 pins the layout: every split file sits under the archive's top-level dataset/.
        files = [TraceFile(root / archive_path(kind, split), kind, split) for kind, split in FILES]
        missing = [str(file.path) for file in files if not file.path.is_file()]
        if missing:
            raise TimeNetConnectorError(f"TRACE archive is missing {missing}")
        return files

    def convert(self, raw_refs: list[TraceFile]) -> TimeFDataset:
        """Convert the archive's split files into one Record per window, and its tasks.

        Pass one reads the six window files, finds the windows that copy another's measurements, and
        checks every row of the two retrieval files against its window. The Records hold lazy signals
        whose values stay in the window files until written. The tasks stream, so the dataset does not
        hold them: each qualifying window keeps its own event-type, retrieval or forecasting task, under
        the authors' split. The files must stay unchanged until the dataset is written.

        Args:
            raw_refs: The archive's eight split files, each once.

        Returns:
            A dataset with the Records and a stream of every task.

        Raises:
            TimeNetConnectorError: If the file set, a window or a retrieval row is invalid.
        """
        if sorted((file.kind, file.split) for file in raw_refs) != sorted(FILES):
            names = [archive_path(kind, split) for kind, split in FILES]
            raise TimeNetConnectorError(f"TRACE requires each of the split files {names} once")
        by_key = {(file.kind, file.split): file for file in raw_refs}
        files = [by_key[key] for key in FILES]

        windows = WindowIndex()
        for file in files:
            if file.kind.holds_windows:
                windows.add_file(file)
        retrieval = RetrievalIndex.read([file for file in files if not file.kind.holds_windows], windows)
        reader = WindowReader()
        records = {window: build_record(window, reader) for window in windows.windows.values()}

        dataset = TimeFDataset(metadata=self.metadata())
        for record in records.values():
            dataset.add_record(record=record)
        dataset.set_task_stream(TASK_TYPES, TaskStream(windows, retrieval, records).tasks)
        return dataset


CONNECTOR = TraceWeatherEventsConnector
