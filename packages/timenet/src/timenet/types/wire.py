"""Strict reusable field types for TimeF wire data."""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveInt, Strict, StrictStr


StrictNonNegativeInt = Annotated[NonNegativeInt, Strict()]
"""An integer greater than or equal to zero, excluding booleans."""

StrictPositiveInt = Annotated[PositiveInt, Strict()]
"""An integer greater than zero, excluding booleans."""

NonEmptyString = Annotated[StrictStr, Field(min_length=1)]
"""A strict, non-empty string."""

Checksum = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
"""A lowercase SHA-256 digest with its algorithm prefix."""

ValueDtype = Literal[
    "bool",
    "enum",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "str",
    "uint8",
    "uint16",
    "uint32",
]
"""A supported TimeF signal value dtype."""

ValueType = Literal["bool", "int", "float", "str", "list"]
"""A supported annotation value-type tag."""

ValueEncoding = Literal["dictionary", "byte_stream_split", "plain"]
"""A supported values-column encoding."""
