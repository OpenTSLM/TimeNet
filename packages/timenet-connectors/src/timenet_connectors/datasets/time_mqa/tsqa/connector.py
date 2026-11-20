"""Convert the TSQA dataset of Time-MQA into records and tasks."""

from collections.abc import Callable, Mapping
from pathlib import Path

from timenet.dataset import TimeFDataset
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.time_mqa.tsqa.anomaly_detection import AnomalyDetectionRow
from timenet_connectors.datasets.time_mqa.tsqa.classification import ClassificationRow
from timenet_connectors.datasets.time_mqa.tsqa.forecasting import parse_forecasting_row
from timenet_connectors.datasets.time_mqa.tsqa.free_answer import FreeAnswerRow
from timenet_connectors.datasets.time_mqa.tsqa.imputation import ImputationRow
from timenet_connectors.datasets.time_mqa.tsqa.release import (
    SOURCE_FILES,
    ReleaseFile,
    ReleasePart,
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


class TimeMqaConnector(
    BaseHuggingFaceConnector,
    hf_repo="Time-MQA/TSQA",
    source_revision="a7f9e95667b695d2647b9eff92bb6a38b5183a34",
    source_files=SOURCE_FILES,
):
    """Connector for the TSQA release of Time-MQA (Hub repo ``Time-MQA/TSQA``)."""

    def convert(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Convert question-and-answer rows into records and tasks.

        Args:
            files: The local path of each downloaded part.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for part, path in files.items():
            for row in ReleaseFile(part=ReleasePart(part), path=path).read_rows():
                _PARSERS[row.family](row).add_records_and_task_to(dataset)
        return dataset


CONNECTOR = TimeMqaConnector
