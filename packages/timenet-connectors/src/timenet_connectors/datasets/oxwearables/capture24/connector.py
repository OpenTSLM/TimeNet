"""Convert the pinned CAPTURE-24 archive into Records and classification tasks.

CAPTURE-24 contains about one day of wrist acceleration at 100 Hz from each of 151 participants.
Activity labels come from body-camera images and sleep diaries. Each participant becomes one Record.
Each 10 s window kept by the authors' benchmark becomes one task per scheme that the benchmark scores.
README.md describes the benchmark rules.

The connector delegates to five modules:

- ``source.py`` reads and checks the release files and summarizes each recording.
- ``windowing.py`` applies the benchmark rules to validated samples.
- ``records.py`` builds the Records, whose acceleration values load from the files when needed.
- ``tasks.py`` streams the tasks. ``schemes.py`` holds the label schemes that both use.
"""

from pathlib import Path

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask
from timenet_connectors.datasets.oxwearables.capture24 import records, source, tasks
from timenet_connectors.download import ensure_archive, find_dir_containing
from timenet_connectors.sources.channels import SharedChannelLoader


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The 6.9 GB zip file attached to the Oxford Research Archive record."""
_ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ARCHIVE_ROOT = "capture24"
"""The archive's one top-level directory, which holds every file below."""


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from the pinned zip of its Oxford Research Archive record."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use] - the connector interface
        """Fetch and extract the pinned archive.

        The download fails, before anything is extracted, if the archive's SHA-256 digest differs from
        ``ARCHIVE_SHA256``.
        Later calls reuse the extracted directory without downloading the archive again.

        Returns:
            A list containing the extracted ``capture24`` directory.
        """
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=ARCHIVE_SHA256)
        return [find_dir_containing(root, f"{_ARCHIVE_ROOT}/{source.METADATA_FILE}")]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Build one Record per recording and stream one task per window and benchmarked scheme.

        The first pass reads and checks each recording. It fails the build on a bad header, timestamp,
        raw label or acceleration value, and keeps the segments, raw label runs, kept windows and
        acceleration digest. The writer reads the acceleration values from the files later, when it
        needs them. If a file is missing or its row count or acceleration values changed, the write fails with
        ``TimeNetConnectorError``. Source errors name the file. Checks that locate a bad row also
        include its line; Arrow parsing errors retain Arrow's diagnostics, which may omit the line.

        Args:
            raw_refs: The one extracted ``capture24`` directory.

        Returns:
            A dataset with the Records and a stream of every task. Each task carries its scheme's label
            set.

        Raises:
            TimeNetConnectorError: If ``raw_refs`` is not one directory, or a table or recording is malformed.
        """
        if len(raw_refs) != 1 or not raw_refs[0].is_dir():
            raise TimeNetConnectorError(f"CAPTURE-24 loads from one directory, got {raw_refs}")
        (root,) = raw_refs
        labels = source.read_labels(root)
        # One participant's samples are held at a time: the first pass keeps only small results.
        summaries = [
            source.summarize_recording(participant, labels) for participant in source.read_participant_metadata(root)
        ]
        summaries_by_path = {summary.participant.path: summary for summary in summaries}

        def read_channels(path: Path) -> tuple[pa.Array, ...]:
            values = source.read_acceleration(summaries_by_path[path])
            return tuple(pa.array(values[axis], type=source.AXIS_TYPE) for axis in source.AXES)

        # One cache per conversion, shared across axes and segments of the same recording.
        shared_channels = SharedChannelLoader(read_channels)
        participant_records = [records.build_record(summary, labels, shared_channels) for summary in summaries]

        dataset = TimeFDataset(metadata=self.metadata())
        for record in participant_records:
            dataset.add_record(record=record)
        dataset.set_task_stream((ClassificationTask,), tasks.task_stream(summaries, participant_records))
        return dataset


CONNECTOR = Capture24Connector
