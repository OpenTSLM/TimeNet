"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

SensorCaps is the LLaSA paper's set of IMU-caption pairs (https://arxiv.org/abs/2406.14498). The
connector downloads the original v2 JSONL files at a pinned commit through the HuggingFace base,
indexes rows by byte offset, and keeps just one decoded row cached across a Record's lazy values, so
sensor values are never all held in memory. Each row becomes one Record and one
caption task that repeats how GPT-4o-mini generated the caption: the row's system message is the
prompt, its gyroscope and accelerometer readings are the Record's signals, and the rest of its user
message (the summary, the derived features, and PAMAP2's location) are Record annotations, given to
the task as input annotations.
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
            A dataset retaining its hierarchy, its annotations, and file-backed loaders.

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
