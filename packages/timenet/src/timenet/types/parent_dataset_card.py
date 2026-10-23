"""Copies of parent dataset cards retained by a child dataset."""

from timenet.types._model import TimeFModel
from timenet.types.metadata import DatasetMetadata
from timenet.types.references import DatasetRef


class ParentDatasetCard(TimeFModel):
    """A copy of the card for a parent dataset used by a build.

    A child manifest stores these cards in ``parent_cards``, indexed by input alias.
    The card records the parent's exact release, license, citation, and access requirements.
    Readers can inspect this information without opening the parent dataset.
    """

    metadata: DatasetMetadata
    """Source license, citation, access, and descriptive provenance."""

    @property
    def source(self) -> DatasetRef:
        """Return the exact parent release, as identified by the copied metadata."""
        return DatasetRef(dataset_id=self.metadata.dataset_id, version=self.metadata.dataset_version)
