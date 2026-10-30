"""Convert the pinned RATS40K JSON files into lazy Records and one category task per row.

Each row becomes a Record whose signals reread their entry on demand, and a ClassificationTask whose
target is the published category name. README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.time_ra.rats40k import release
from timenet_connectors.datasets.time_ra.rats40k.keys import AnnotationKey
from timenet_connectors.datasets.time_ra.rats40k.loader import RATS40KDatasetLoader
from timenet_connectors.datasets.time_ra.rats40k.release import PinnedInventory, Variant


class RATS40KConnector(BaseHuggingFaceConnector):
    """Build RATS40K from the pinned, gated JSON files of its Hugging Face release."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {variant.value: variant.source_file for variant in Variant}
    SOURCE_INVENTORY: ClassVar[Mapping[Variant, PinnedInventory]] = release.PINNED_INVENTORY
    """Each file's pinned rows and null keys, which every build checks."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert each file into one lazy Record per usable row, and a stream of one task per Record.

        The connector adds the pinned source's provenance. ``RATS40KDatasetLoader`` reads each file once,
        validates every row, checks the pinned inventory, and builds one lazy Record per row; its
        ``populate`` adds them and streams the tasks, not holding them. A malformed row or an inventory
        that differs from :attr:`SOURCE_INVENTORY` raises ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_REPOSITORY, self.HF_REPO),
            (AnnotationKey.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = RATS40KDatasetLoader.parse_from(raw_refs, inventory=self.SOURCE_INVENTORY)
        return loader.populate(dataset)


CONNECTOR = RATS40KConnector
