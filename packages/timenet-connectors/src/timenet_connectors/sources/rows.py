"""Typed source rows with consistent validation errors and explicit source locations."""

from enum import StrEnum
from typing import Annotated, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from timenet.errors import TimeNetConnectorError


NonBlankText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
"""Text containing a non-whitespace character; accepted text is preserved unchanged."""


_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)
PublishedStrEnum = Annotated[_StrEnumT, Field(strict=False), BeforeValidator(TypeAdapter(StrictStr).validate_python)]
"""A published string converted to its enum without coercing other decoded types."""


class SourceRow(BaseModel):
    """A typed source row: reject extra fields and coercion of already-decoded values.

    Fields cannot be reassigned. CSV readers explicitly use Pydantic's string-input validation;
    JSON and Parquet readers use strict decoded-value validation. Defaults and source-specific
    missing-value conversions belong in each connector's model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


RowT = TypeVar("RowT", bound=SourceRow)


def row_error_message(error: ValidationError, location: str) -> str:
    """Return an error message naming the first invalid field, including nested positions."""
    first = error.errors(include_url=False)[0]
    field = ".".join(str(part) for part in first["loc"]) or "row"
    return f"{location}: {field}: {first['msg']}"


def validate_row(values: object, model: type[RowT], *, location: str) -> RowT:
    """Validate a decoded row without coercion, retaining its source location.

    Returns:
        The connector's typed row.

    Raises:
        TimeNetConnectorError: If a field violates the model's contract.
    """
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        raise TimeNetConnectorError(row_error_message(exc, location)) from exc
