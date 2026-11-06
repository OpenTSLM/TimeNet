"""Convert the pinned TSAQA parquet files into lazy Records and one question task per row.

Each row's series become Records: its input series, then each candidate series its question prints,
which the prompt shows as ``<series N>``. A true-or-false row becomes a ClassificationTask, a
multiple-choice or puzzle row an AnswerTask, each with the published answer as target. README.md
explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from collections.abc import Mapping
from functools import partial
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation, AnswerTask, ClassificationTask
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import SplitFile
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.records import add_records
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import (
    SeriesLoader,
    find_issues,
    read_rows,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.tasks import iter_tasks


class TSAQAConnector(BaseHuggingFaceConnector):
    """Build TSAQA from the pinned train, val and test parquet files of its Hugging Face release."""

    HF_REPO = "TSAQA/TSAQA-Benchmark"
    SOURCE_REVISION = "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {split.value: split.filename for split in SplitFile}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert each split file into lazy Records and a stream of one task per row.

        Every file is read once up front to check each row and keep its small fields. The source issues
        need all three splits, because a copy of a question may be in another one. The series values stay
        in the files until written, so the files must stay available until then.

        Args:
            raw_refs: The downloaded split files.

        Returns:
            A dataset with the published labels, the Records, and a stream of every task.

        Raises:
            TimeNetConnectorError: If a split file is missing, or a row is malformed.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by read_rows
        paths = {ref.part: ref.path for ref in raw_refs}
        rows = read_rows(paths)
        issues = find_issues(rows)
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", self.SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value))
        records = add_records(dataset, rows, SeriesLoader(paths))
        # Each Record's loader holds its row, so the stream reuses the checked rows instead of a reread.
        dataset.set_task_stream((ClassificationTask, AnswerTask), partial(iter_tasks, dataset, rows, records, issues))
        return dataset


CONNECTOR = TSAQAConnector
