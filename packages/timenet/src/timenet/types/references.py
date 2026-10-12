"""Stable references to dataset versions in composition graphs."""

from functools import total_ordering
from typing import Any

from pydantic import GetJsonSchemaHandler, model_serializer, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId, SemanticVersion
from timenet.types.wire import Checksum


@total_ordering
class DatasetRef(TimeFModel):
    """An exact, immutable dataset-version reference."""

    dataset_id: DatasetId
    version: SemanticVersion

    @model_validator(mode="before")
    @classmethod
    def _parse_reference(cls, value: Any) -> Any:
        """Parse an exact ``org/name@version`` reference.

        Returns:
            The parsed fields, or an existing model input.

        Raises:
            TimeFValidationError: If a string does not contain one exact version.
        """
        if isinstance(value, str):
            if value.count("@") != 1:
                raise TimeFValidationError("dataset reference must be org/name@version")
            dataset_id, version = value.split("@")
            return {"dataset_id": dataset_id, "version": version}
        return value

    @model_serializer(when_used="json")
    def _serialize_reference(self) -> str:
        """Encode an exact reference as a string.

        Returns:
            The canonical reference.
        """
        return str(self)

    @classmethod
    def __get_pydantic_json_schema__(  # noqa: PLW3201 - Pydantic schema hook
        cls, _schema: CoreSchema, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Describe the exact reference's wire representation.

        Returns:
            The schema of an ``org/name@version`` string.
        """
        return {
            "type": "string",
            # An org/name@major.minor.patch reference with numeric version components.
            "pattern": r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*@\d+\.\d+\.\d+$",
        }

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
