"""Distinct names for declared parents and payload sources."""

from typing import Annotated, Any, NewType, Self

from pydantic import Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from timenet.errors import TimeFValidationError


class ParentAlias(str):
    """A stable, connector-local name for a declared parent release, such as ``ptbxl``.

    An alias compares and hashes like the string it holds, so it works as a ``str`` key, but it is a
    type of its own: ``isinstance(value, ParentAlias)`` tells it apart from a plain ``str``. It is
    never blank. Pydantic models validate a strict string into an alias and serialize it as a plain
    string.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """Return ``value`` as an alias.

        Returns:
            The alias.

        Raises:
            TimeFValidationError: If ``value`` is not a string or is blank.
        """
        if not isinstance(value, str) or not value.strip():
            raise TimeFValidationError(f"a parent alias must be a nonblank string, got {value!r}")
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: PLW3201 - pydantic's hook for a custom type
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(strict=True),
            serialization=core_schema.plain_serializer_function_ser_schema(str, return_schema=core_schema.str_schema()),
        )


SourceKey = NewType("SourceKey", str)
"""A catalog key naming the payload source: the exact release that stores a payload file."""

SourceKeyField = Annotated[SourceKey, Field(strict=True)]
"""A payload source key validated as a strict string."""
