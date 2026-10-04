"""Reusable Pydantic field types for dataset metadata."""

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, PlainSerializer, StrictStr, TypeAdapter, WithJsonSchema

from timenet.types.version import Version


_SEMANTIC_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"


def _parse_version(value: Any) -> Any:
    """Parse the wire representation of a semantic version.

    Returns:
        A parsed version, or the original value for Pydantic to validate.
    """
    if type(value) is str:
        return Version.parse(value)
    return value


DatasetId = Annotated[
    StrictStr,
    Field(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*$"),
]
"""A safe, case-sensitive ``org/name`` dataset identifier."""

_DATASET_ID_ADAPTER = TypeAdapter(DatasetId)

SemanticVersion = Annotated[
    Version,
    BeforeValidator(_parse_version),
    PlainSerializer(str, return_type=str),
    WithJsonSchema({"type": "string", "pattern": _SEMANTIC_VERSION_PATTERN}),
]
"""A :class:`Version` encoded as a semantic-version string on the wire."""

NonEmptyText = Annotated[StrictStr, Field(min_length=1)]
"""A non-empty string; callers may apply stronger whitespace validation."""
