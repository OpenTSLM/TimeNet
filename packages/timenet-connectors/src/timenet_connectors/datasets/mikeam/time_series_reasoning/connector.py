"""Convert the pinned release into lazy Records and the authors' three multiple-choice tasks.

The six files become one Record per series, per rounded copy of a series and per changed series, and one
``AnswerTask`` per row of the authors' task files, whose prompt is the text their loader gives a model.
README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that names
it in ``input_annotations``.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.mikeam.time_series_reasoning import release
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import AnnotationKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.loader import TimeSeriesReasoningLoader
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile


class TimeSeriesReasoningConnector(BaseHuggingFaceConnector):
    """Build the dataset from the six files of its pinned Hugging Face release."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {file.part: file.value for file in SourceFile}
    PINNED: ClassVar[bool] = True
    """Whether every build checks the files' row counts and blank lines against ``release``'s pins."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the six files into lazy Records and a stream of one task per task row.

        The connector adds the pinned source's provenance. ``TimeSeriesReasoningLoader`` reads each file
        once, validates every row, checks that every copy of a series agrees with ``TS_Dataset.jsonl``
        and, when :attr:`PINNED`, that the counts match the pins, and builds every Record. Its
        ``populate`` adds them and streams the tasks, rereading the two question files. A malformed or
        inconsistent row, or a count that differs, raises ``TimeFValidationError``.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_REPOSITORY, self.HF_REPO),
            (AnnotationKey.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = TimeSeriesReasoningLoader.parse_from(raw_refs, pinned=self.PINNED)
        return loader.populate(dataset)


CONNECTOR = TimeSeriesReasoningConnector
