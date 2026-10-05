"""Stable references to dataset versions in composition graphs."""

from functools import total_ordering
from typing import Annotated

from pydantic import Field, StrictStr

from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId, SemanticVersion
from timenet.types.wire import Checksum


@total_ordering
class DatasetRef(TimeFModel):
    """An exact, immutable dataset-version reference."""

    dataset_id: DatasetId
    version: SemanticVersion

    def __str__(self) -> str:
        """Return the canonical ``org/name@version`` representation."""
        return f"{self.dataset_id}@{self.version}"

    def __lt__(self, other: object) -> bool:
        """Order references by dataset id and then semantic version.

        Returns:
            Whether this reference sorts before the other reference.
        """
        if not isinstance(other, DatasetRef):
            return NotImplemented
        return (self.dataset_id, self.version) < (other.dataset_id, other.version)


class ParentDataset(TimeFModel):
    """A named, exact parent declared by a dataset card."""

    alias: Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
    """The name the connector addresses this parent by; unique within the card."""
    dataset_id: DatasetId
    version: SemanticVersion

    @property
    def dataset(self) -> DatasetRef:
        """Return this parent as an exact dataset reference."""
        return DatasetRef(dataset_id=self.dataset_id, version=self.version)


class LockedDependency(TimeFModel):
    """One exact version in a dataset's dependency closure, pinned by its manifest checksum.

    The checksum covers the parent's ``manifest.json``, which lists the checksum of every file in
    that version, so one row pins the parent's complete content.
    """

    dataset_id: DatasetId
    version: SemanticVersion
    manifest_checksum: Checksum

    @property
    def dataset(self) -> DatasetRef:
        """Return the locked version as an exact dataset reference."""
        return DatasetRef(dataset_id=self.dataset_id, version=self.version)
