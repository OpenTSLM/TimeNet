"""Convert the pinned TSAQA parquet files into lazy Records and one question task per row.

Each row's series become Records: its input series, then each candidate series its question prints,
which the prompt shows as ``<series N>``. A true-or-false row becomes a ClassificationTask, a
multiple-choice or puzzle row an AnswerTask, each with the published answer as target. README.md
explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import release
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import AnnotationKey
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.loader import TSAQADatasetLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import InventoryKey, SourceSplit


class TSAQAConnector(BaseHuggingFaceConnector):
    """Build TSAQA from the pinned train, val and test parquet files of its Hugging Face release."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {split.value: split.source_file for split in SourceSplit}
    SOURCE_INVENTORY: ClassVar[Mapping[SourceSplit, Mapping[InventoryKey, int]]] = release.PINNED_INVENTORY
    """Each file's pinned row counts, which every build checks."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert each split file into lazy Records and a stream of one task per row.

        The connector adds the pinned source's provenance. ``TSAQADatasetLoader`` reads each file once,
        validates every row against the authors' templates, checks the pinned inventory, and builds
        the Records; its ``populate`` adds them and streams the tasks, not holding them. A malformed
        row or an inventory that differs from the pinned release raises ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_REPOSITORY, self.HF_REPO),
            (AnnotationKey.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = TSAQADatasetLoader.parse_from(raw_refs, inventory=self.SOURCE_INVENTORY)
        return loader.populate(dataset)


CONNECTOR = TSAQAConnector
