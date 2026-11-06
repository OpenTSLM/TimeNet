"""Orchestrate the pinned Time Series Reasoning release, records and repeatable tasks."""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import AnswerTask
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import RecordIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.source import Release
from timenet_connectors.datasets.mikeam.time_series_reasoning.tasks import TaskStream
from timenet_connectors.datasets.mikeam.time_series_reasoning.vocabulary import AnnotationKey, File


class TimeSeriesReasoningConnector(BaseHuggingFaceConnector):
    """Build the dataset from the six files of its pinned Hugging Face release."""

    HF_REPO = "mikeam/time-series-reasoning"
    SOURCE_REVISION = "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {file.part: file.value for file in File}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the six files into lazy Records and a stream of one task per task row.

        Pass one reads each file once and checks every row. The tasks reread the five task files, so
        the files must stay available until writing ends.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.

        Raises:
            TimeNetConnectorError: If a file is missing, a row is malformed, or a row's copy of its
                series differs from ``TS_Dataset.jsonl``.
        """
        by_part = {ref.part: ref.path for ref in raw_refs}
        missing = [file.value for file in File if file.part not in by_part]
        if missing:
            raise TimeNetConnectorError(f"the release needs each of its files, missing {missing}")
        release = Release.read({file: by_part[file.part] for file in File})
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.add_annotations(
            {
                AnnotationKey.SOURCE_REPOSITORY: self.HF_REPO,
                AnnotationKey.SOURCE_REVISION: self.SOURCE_REVISION,
            }
        )
        records = RecordIndex.build(release)
        for record in records.all():
            dataset.add_record(record=record)
        dataset.set_task_stream((AnswerTask,), TaskStream(release, records).tasks)
        return dataset


CONNECTOR = TimeSeriesReasoningConnector
