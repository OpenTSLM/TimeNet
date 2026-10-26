"""Distinct names for declared inputs and physical payload owners."""

from typing import Annotated, NewType

from pydantic import Field


InputAlias = NewType("InputAlias", str)
"""A stable, connector-local name for a declared input."""

OwnerKey = NewType("OwnerKey", str)
"""A catalog key naming the exact release that stores a payload."""

InputAliasField = Annotated[InputAlias, Field(strict=True)]
"""An input alias validated as a strict string."""

OwnerKeyField = Annotated[OwnerKey, Field(strict=True)]
"""A payload owner key validated as a strict string."""
