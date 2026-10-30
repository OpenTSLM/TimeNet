"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The connector downloads the original JSONL files at a pinned commit through the HuggingFace base,
indexes rows by byte offset, and keeps just one decoded sensor row cached across the six channel
loaders, so sensor values are never all held in memory. Tasks use the original system message as
their prompt, with the activity summary and optional sensor location as input annotations. Each
derived feature is a Record annotation, held in memory with the Records.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.bash_lab.sensorcaps import release
from timenet_connectors.datasets.bash_lab.sensorcaps.loader import SensorCapsLoader


class SensorCapsConnector(BaseHuggingFaceConnector):
    """Build SensorCaps with lazy values and streamed captions."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_ROWS: ClassVar[Mapping[str, int]] = release.SOURCE_ROWS
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        partition: release.source_file(partition) for partition in release.SOURCE_ROWS
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Index local rows into six-channel Records and a re-iterable caption stream.

        Returns:
            A dataset retaining its hierarchy, small annotations, and file-backed loaders.

        Raises:
            TimeFValidationError: If references or source rows violate the SensorCaps mapping.
        """
        loader = SensorCapsLoader.parse_from(raw_refs)
        for file, count in loader.row_counts.items():
            # Every pinned count is nonzero, so this also rejects an empty partition.
            expected = self.SOURCE_ROWS[file.part]
            if count != expected:
                raise TimeFValidationError(f"{file.path} ({file.part}): expected {expected} rows, found {count}")
        return loader.populate(TimeFDataset(metadata=self.metadata()))


CONNECTOR = SensorCapsConnector
