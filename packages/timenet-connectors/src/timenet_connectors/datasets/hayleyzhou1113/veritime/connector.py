"""Orchestrate VeriTime conversion and repeatable tasks."""

from collections.abc import Iterator, Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    ScalarPredictionTask,
    Task,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.hayleyzhou1113.veritime.records import build_record
from timenet_connectors.datasets.hayleyzhou1113.veritime.source import Row, SeriesKey, read_rows, to_part
from timenet_connectors.datasets.hayleyzhou1113.veritime.tasks import build_task
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import Part


class VeriTimeConnector(BaseHuggingFaceConnector):
    """Build VeriTime from the pinned JSON Lines files of its Hugging Face release."""

    HF_REPO = "HayleyZhou1113/VeriTime"
    SOURCE_REVISION = "fc88cb1305a96a208093e13b69d766b19c84c99b"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        Part.SCENARIO_TRAIN.value: "Dataset/Scenario/scenario_shuffled_train.jsonl",
        Part.SCENARIO_TEST.value: "Dataset/Scenario/scenario_shuffled_test.jsonl",
        # "knowldege" is the release's own spelling.
        Part.KNOWLEDGE_TRAIN.value: "Dataset/Knowledge/knowldege_shuffled_train.jsonl",
        Part.KNOWLEDGE_TEST.value: "Dataset/Knowledge/knowledge_test.jsonl",
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per distinct series and a stream of one task per row.

        Pass one reads each file once, checks every row, and keeps its small fields, its byte range and a
        digest of its series. Rows that publish equal values of the same kind share one Record. Pass two
        builds the Records and the tasks. The series values stay in the files until written, so the files
        must stay available until then.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.

        Raises:
            TimeNetConnectorError: If a reference names no VeriTime part.
        """  # ruff: ignore[docstring-extraneous-exception] - a file's rows are checked by read_rows
        rows = [row for ref in raw_refs for row in read_rows(ref.path, to_part(ref.part))]
        by_series: dict[SeriesKey, list[Row]] = {}
        for row in rows:
            by_series.setdefault(row.series_key, []).append(row)
        records = {key: build_record(series) for key, series in by_series.items()}

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.add_annotations({"source_repository": self.HF_REPO, "source_revision": self.SOURCE_REVISION})
        for record in records.values():
            dataset.add_record(record=record)

        def tasks() -> Iterator[Task]:
            for row in rows:
                yield build_task(row, records[row.series_key])

        dataset.set_task_stream((ClassificationTask, AnswerTask, ScalarPredictionTask), tasks)
        return dataset


CONNECTOR = VeriTimeConnector
