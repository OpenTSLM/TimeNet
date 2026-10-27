"""Distinct names for declared inputs and payload sources."""

from typing import Annotated, NewType

from pydantic import Field


InputAlias = NewType("InputAlias", str)
"""A stable, connector-local name for a declared input."""

SourceKey = NewType("SourceKey", str)
"""A catalog key naming the payload source: the exact release that stores a payload file."""

InputAliasField = Annotated[InputAlias, Field(strict=True)]
"""An input alias validated as a strict string."""

SourceKeyField = Annotated[SourceKey, Field(strict=True)]
"""A payload source key validated as a strict string."""
