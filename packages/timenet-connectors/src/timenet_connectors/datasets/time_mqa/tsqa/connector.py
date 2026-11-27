"""The connector of the TSQA dataset of Time-MQA.

The connector downloads the CSV files of the release from the pinned Hub revision. It reads each
row through the shared :mod:`.release` module and hands the row to the parser of its task family.
Each family module parses the texts of its rows and builds their Records and tasks.
"""

from collections.abc import Callable, Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.time_mqa.tsqa.anomaly_detection import AnomalyDetectionRow
from timenet_connectors.datasets.time_mqa.tsqa.release import (
    HF_REPO,
    SOURCE_FILES,
    SOURCE_REVISION,
    ReleaseFile,
    ReleaseRow,
    TaskFamily,
    TaskRow,
)


_PARSERS: Mapping[TaskFamily, Callable[[ReleaseRow], TaskRow]] = {
    TaskFamily.ANOMALY_DETECTION: AnomalyDetectionRow.from_release_row,
}
"""The parser of each task family."""


class TimeMqaConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA release of Time-MQA (Hub repo ``Time-MQA/TSQA``)."""

    HF_REPO = HF_REPO
    SOURCE_REVISION = SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = SOURCE_FILES

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Create the Records and the task of each question-and-answer row.

        Args:
            raw_refs: The downloaded files, one for each part.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            for row in ReleaseFile.from_hub_file_ref(ref).read_rows():
                _PARSERS[row.family](row).add_records_and_task_to(dataset)
        return dataset


CONNECTOR = TimeMqaConnector
