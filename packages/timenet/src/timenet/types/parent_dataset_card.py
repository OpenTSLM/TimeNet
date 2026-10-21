"""Copies of parent dataset cards retained by a child dataset."""

from typing import Self

from pydantic import model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata import DatasetMetadata
from timenet.types.references import DatasetRef


class ParentDatasetCard(TimeFModel):
    """A copy of the card for a parent dataset used by a build.

    A child manifest stores these cards in ``parent_cards``, indexed by input alias.
    The card records the parent's exact release, license, citation, and access requirements.
    Readers can inspect this information without opening the parent dataset.
    """

    source: DatasetRef
    metadata: DatasetMetadata
    """Source license, citation, access, and descriptive provenance."""

    @model_validator(mode="after")
    def _matching_source(self) -> Self:
        if self.source.dataset_id != self.metadata.dataset_id or self.source.version != self.metadata.dataset_version:
            raise TimeFValidationError(
                f"parent dataset card source {self.source} does not match metadata release "
                f"{self.metadata.dataset_id}@{self.metadata.dataset_version}"
            )
        return self
