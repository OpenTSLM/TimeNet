"""Build TSAQA with raw series and unchanged published answers.

Temporal questions refer to labelled raw candidate inputs instead of printed normalized arrays.
Other questions retain their published text. Every original question remains in task metadata.
This is a raw-input variant, not the paper's normalized-input evaluation.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path

from datasets import Dataset
from datasets.exceptions import DatasetGenerationError
from pydantic import ValidationError

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.types import Split
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import AnalysisTask, TSAQARow
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.tasks import (
    AnomalyDetectionTask,
    CharacterizationTask,
    ClassificationTask,
    ComparisonTask,
    DataTransformationTask,
    TemporalRelationshipTask,
    TSAQATask,
)


class TSAQARegistry:
    """Keep the parsed tasks in source order, including repeated questions."""

    def __init__(self) -> None:
        self._tasks: list[TSAQATask] = []

    def store(self, task: TSAQATask) -> None:
        """Keep this task as a separate sample, even if another row asks the same question."""
        self._tasks.append(task)

    def register_in_timenet_dataset(self, dataset: TimeFDataset) -> None:
        """Add the tasks and all raw series to TimeNet, including series not used as model inputs."""
        for task in self:
            for record in task.records:
                dataset.add_record(record)
        dataset.add_tasks(task.to_timenet_task() for task in self)

    def __iter__(self) -> Iterator[TSAQATask]:
        return iter(self._tasks)


class TSAQAConnector(
    BaseHuggingFaceConnector,
    hf_repo="TSAQA/TSAQA-Benchmark",
    source_revision="883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd",
    source_files={"train": "train.parquet", "validation": "val.parquet", "test": "test.parquet"},
):
    """Convert the pinned TSAQA release into tasks using its raw input series."""

    def convert(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Preserve all rows and raw series, with one question task per published sample.

        Args:
            files: The downloaded train, validation and test Parquet files.

        Returns:
            A dataset with materialized raw signals, labelled temporal candidates, and original targets and splits.

        Raises:
            TimeFValidationError: If the split files are missing or unexpected.
        """
        if set(files) != set(self.SOURCE_FILES):
            raise TimeFValidationError(f"TSAQA requires splits {tuple(self.SOURCE_FILES)}, got {tuple(files)}")
        registry = self.read_tasks(files)
        dataset = TimeFDataset(metadata=self.metadata())
        registry.register_in_timenet_dataset(dataset)
        return dataset

    def read_tasks(self, files: Mapping[str, Path]) -> TSAQARegistry:
        """Read each row and build the TSAQA task class for its question type.

        Returns:
            The parsed tasks in their original split and row order.

        Raises:
            TimeNetConnectorError: If a source cannot be loaded or a row is invalid.
        """
        registry = TSAQARegistry()
        for name in self.SOURCE_FILES:
            split = Split.from_name(name)
            path = files[name]
            try:
                source = Dataset.from_parquet(str(path), columns=list(TSAQARow.model_fields), keep_in_memory=False)
            except (DatasetGenerationError, OSError, ValueError) as exc:
                raise TimeNetConnectorError(f"{path}: cannot load TSAQA source: {exc}") from exc
            for index, raw_row in enumerate(source):
                try:
                    row = TSAQARow.model_validate(raw_row)
                except ValidationError as exc:
                    raise TimeNetConnectorError(f"{path}: row {index + 1}: {exc}") from exc
                task: TSAQATask
                if row.task is AnalysisTask.ANOMALY_DETECTION:
                    task = AnomalyDetectionTask(row, index, split=split, filename=path.name)
                elif row.task is AnalysisTask.CLASSIFICATION:
                    task = ClassificationTask(row, index, split=split, filename=path.name)
                elif row.task is AnalysisTask.CHARACTERIZATION:
                    task = CharacterizationTask(row, index, split=split, filename=path.name)
                elif row.task is AnalysisTask.COMPARISON:
                    task = ComparisonTask(row, index, split=split, filename=path.name)
                elif row.task is AnalysisTask.DATA_TRANSFORMATION:
                    task = DataTransformationTask(row, index, split=split, filename=path.name)
                else:
                    task = TemporalRelationshipTask(row, index, split=split, filename=path.name)
                registry.store(task)
        return registry


CONNECTOR = TSAQAConnector
