"""Convert the TSQA dataset of Time-MQA into records and tasks."""

from collections.abc import Callable, Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.time_mqa.tsqa.anomaly_detection import AnomalyDetectionRow
from timenet_connectors.datasets.time_mqa.tsqa.classification import ClassificationRow
from timenet_connectors.datasets.time_mqa.tsqa.forecasting import parse_forecasting_row
from timenet_connectors.datasets.time_mqa.tsqa.free_answer import FreeAnswerRow
from timenet_connectors.datasets.time_mqa.tsqa.imputation import ImputationRow
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
    TaskFamily.CLASSIFICATION: ClassificationRow.from_release_row,
    TaskFamily.FORECASTING: parse_forecasting_row,
    TaskFamily.IMPUTATION: ImputationRow.from_release_row,
    TaskFamily.OPEN_ENDED_QA: FreeAnswerRow.from_release_row,
}
"""The parser for each task family."""


class TimeMqaConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA release of Time-MQA (Hub repo ``Time-MQA/TSQA``)."""

    HF_REPO = HF_REPO
    SOURCE_REVISION = SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = SOURCE_FILES

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert question-and-answer rows into records and tasks.

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
