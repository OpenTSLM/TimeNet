"""Shared configuration for immutable TimeF data models."""

from pydantic import BaseModel, ConfigDict


class TimeFModel(BaseModel):
    """Reject unknown fields and validate immutable TimeF model instances."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        serialize_by_alias=True,
        validate_by_name=True,
        validate_default=True,
    )
