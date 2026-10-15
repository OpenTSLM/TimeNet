"""Exact dataset releases used by build inputs and physical payload owners."""

import hashlib
import json
from typing import Any

from pydantic import Field

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId, SemanticVersion


class DatasetRef(TimeFModel):
    """An exact release, independent of its registry location."""

    dataset_id: DatasetId
    version: SemanticVersion

    def __str__(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> "DatasetRef":
        """Parse an exact ``org/name@version`` reference.

        Returns:
            The validated reference.

        Raises:
            TimeFValidationError: If the reference does not contain an exact version.
        """
        dataset_id, separator, version = value.rpartition("@")
        if not separator or version == "latest":
            raise TimeFValidationError(f"an exact dataset release is required: {value!r}")
        return cls.model_validate({"dataset_id": dataset_id, "version": version})


class InputLock(TimeFModel):
    """A build input pinned to its canonical manifest digest."""

    source: DatasetRef
    manifest_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def manifest_digest(value: dict[str, Any]) -> str:
    """Return a digest independent of manifest whitespace and property order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
