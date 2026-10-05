"""Shared configuration for immutable TimeF data models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


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
