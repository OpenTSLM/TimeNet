"""Convert the pinned EngineMT-QA files into lazy Records and one task per QA pair.

Each HDF5 row a QA names becomes a Record whose 33 signals read the row on demand, and each labelled QA
pair becomes the task of its stage. README.md explains how this follows the authors.

Metadata is provenance and never model input.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.pandalin98.enginemt_qa import release
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import AnnotationKey
from timenet_connectors.datasets.pandalin98.enginemt_qa.loader import EngineMTQADatasetLoader
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import Part, PinnedRelease


class EngineMTQAConnector(BaseHuggingFaceConnector):
    """Build EngineMT-QA from the pinned files of its Hugging Face release."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {part.value: part.source_file for part in Part}
    SOURCE_RELEASE: ClassVar[PinnedRelease] = release.PINNED_RELEASE
    """What the pinned files hold, which every build checks."""
    ALL_ROWS: ClassVar[bool] = False
    """Build a Record for every one of the 118,921 HDF5 rows, as the authors' encoder pretraining reads
    them, rather than for the 7,755 rows a QA names. The extra Records have no task."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per row, and a stream of one task per labelled QA pair.

        The connector adds the pinned source's provenance. ``EngineMTQADatasetLoader`` checks the HDF5
        layout, reads and checks both QA files, and builds the Records; its ``populate`` adds them and
        streams the tasks, not holding them. A malformed file, or counts that differ from
        :attr:`SOURCE_RELEASE`, raise ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_REPOSITORY, self.HF_REPO),
            (AnnotationKey.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = EngineMTQADatasetLoader.parse_from(raw_refs, all_rows=self.ALL_ROWS, release=self.SOURCE_RELEASE)
        return loader.populate(dataset)


CONNECTOR = EngineMTQAConnector
