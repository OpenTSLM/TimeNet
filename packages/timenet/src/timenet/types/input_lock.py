"""Validated parent metadata recorded with an exact build input."""

from typing import Self

from pydantic import model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata import DatasetMetadata
from timenet.types.references import DatasetRef


class InputLock(TimeFModel):
    """An exact build input with its validated dataset metadata snapshot."""

    source: DatasetRef
    metadata: DatasetMetadata
    """Source license, citation, access, and descriptive provenance."""

    @model_validator(mode="after")
    def _matching_source(self) -> Self:
        if self.source.dataset_id != self.metadata.dataset_id or self.source.version != self.metadata.dataset_version:
            raise TimeFValidationError(
                f"input lock source {self.source} does not match metadata release "
                f"{self.metadata.dataset_id}@{self.metadata.dataset_version}"
            )
        return self
