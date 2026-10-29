"""Convert the pinned VeriTime JSON Lines files into lazy Records and one reasoning task per row.

Each distinct series becomes a Record whose signal rereads its first row on demand, and each row a task
whose prompt is the authors' question and whose rationale is the published chain of thought.
README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that names
it in ``input_annotations``, and no VeriTime task names one.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.hayleyzhou1113.veritime import release
from timenet_connectors.datasets.hayleyzhou1113.veritime.keys import AnnotationKey
from timenet_connectors.datasets.hayleyzhou1113.veritime.loader import VeriTimeDatasetLoader
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import Part, PinnedInventory


class VeriTimeConnector(BaseHuggingFaceConnector):
    """Build VeriTime from the pinned JSON Lines files of its Hugging Face release."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {part.value: part.source_file for part in Part}
    SOURCE_INVENTORY: ClassVar[PinnedInventory] = release.PINNED_INVENTORY
    """The pinned rows per file and task, and distinct series per family, which every build checks."""

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per distinct series, and a stream of one task per row.

        The connector adds the pinned source's provenance. ``VeriTimeDatasetLoader`` reads each file once,
        validates every row, checks the pinned inventory, and builds the Records; its ``populate`` adds
        them and streams the tasks, not holding them. A malformed row or an inventory that differs from
        :attr:`SOURCE_INVENTORY` raises ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_REPOSITORY, self.HF_REPO),
            (AnnotationKey.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = VeriTimeDatasetLoader.parse_from(raw_refs, inventory=self.SOURCE_INVENTORY)
        return loader.populate(dataset)


CONNECTOR = VeriTimeConnector
