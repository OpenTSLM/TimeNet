"""Shared configuration for immutable TimeF data models, and the text of their validation errors."""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class TimeFModel(BaseModel):
    """Reject unknown fields and validate immutable TimeF model instances."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        serialize_by_alias=True,
        validate_by_name=True,
        validate_default=True,
    )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle as the validated serialized fields rather than as raw instance state.

        Returns:
            The rebuild callable and its arguments: this class and the serialized fields.
        """
        return type(self).model_validate, (self.model_dump(),)


def _first_problem(error: ValidationError) -> str:
    """Describe the first problem of a validation error as ``<location>: <message>``.

    Args:
        error: The validation error.

    Returns:
        The location and message of the first problem, or the message alone if it has no location.
    """
    first = error.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    message = first["msg"].removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
