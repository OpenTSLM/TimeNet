"""Reusable Pydantic field types for dataset metadata."""

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, PlainSerializer, StrictStr, TypeAdapter, WithJsonSchema

from timenet.types.version import Version


_DATASET_COMPONENT = r"[A-Za-z0-9_-][A-Za-z0-9._-]*"
_DATASET_ID = rf"{_DATASET_COMPONENT}/{_DATASET_COMPONENT}"
_SEMANTIC_VERSION = r"\d+\.\d+\.\d+"

DATASET_ID_PATTERN = rf"^{_DATASET_ID}$"
"""Match a case-sensitive ``org/name`` identifier.

Both components start with a letter, digit, underscore, or hyphen. Remaining
characters may also include dots. ``/`` separates the components. ``^`` and ``$``
anchor the identifier at the start and end of the string.
"""

SEMANTIC_VERSION_PATTERN = rf"^{_SEMANTIC_VERSION}$"
r"""Match a ``major.minor.patch`` version with three numeric components.

Each ``\d+`` accepts one or more digits, and ``\.`` matches a literal dot.
The anchors exclude surrounding text. Version ranges and labels such as
``latest`` do not match.
"""

DATASET_REF_PATTERN = rf"^{_DATASET_ID}@{_SEMANTIC_VERSION}$"
"""Match an exact ``org/name@major.minor.patch`` release reference.

Reuse the identifier and version fragments above, with ``@`` between them.
Anchor the complete reference once so the same rules apply to standalone
metadata fields and release references.
"""

INPUT_ALIAS_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
"""Match a case-sensitive local name for a dataset input.

The first character is an ASCII letter. Remaining characters may be ASCII
letters, digits, underscores, or hyphens. ``^`` and ``$`` anchor the alias at
the start and end of the string.
"""


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
    Field(pattern=DATASET_ID_PATTERN),
]
"""A safe, case-sensitive ``org/name`` dataset identifier."""

InputAlias = Annotated[StrictStr, Field(pattern=INPUT_ALIAS_PATTERN)]
"""A case-sensitive local alias for an exact dataset input."""

_DATASET_ID_ADAPTER = TypeAdapter(DatasetId)


def validate_dataset_id(value: object) -> str:
    """Validate a safe, case-sensitive dataset identifier.

    Args:
        value: Candidate ``org/name`` identifier.

    Returns:
        The validated identifier.
    """
    return _DATASET_ID_ADAPTER.validate_python(value)


SemanticVersion = Annotated[
    Version,
    BeforeValidator(_parse_version),
    PlainSerializer(str, return_type=str),
    WithJsonSchema({"type": "string", "pattern": SEMANTIC_VERSION_PATTERN}),
]
"""A :class:`Version` encoded as a semantic-version string on the wire."""

NonEmptyText = Annotated[StrictStr, Field(min_length=1)]
"""A non-empty string; callers may apply stronger whitespace validation."""
