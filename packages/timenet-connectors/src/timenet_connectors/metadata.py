"""A Record's metadata: the fields every dataset shares, plus one dataset's own typed fields.

A connector declares its Record metadata as one subclass of ``RecordMetadata`` that adds its own fields, and a
Record holds one such object. ``to_json`` writes it as one flat JSON object::

    {"subject_id": "USW00093242", "coordinates": ..., "sampling": "random", "windows": [...]}

Design choices:

- One class per dataset, one level of keys: the stored JSON is plain keys and values, with no type tags or
  wrappers, so a reader without the connector, such as a consumer who has not installed it, can still read it.
  A field may still hold a structured value, such as a pydantic model or a tuple of them.
- Values keep their types: the fields are typed pydantic fields, not a bag of strings.
- Metadata is frozen, so metadata shared between objects never changes.
- A subclass adds fields but never redefines a shared one, so a shared field means the same in every dataset.
- Reading needs the dataset's class: ``CGMRecordMetadata.from_json(...)``.

A dataset whose metadata may change shape over time declares a ``version`` field as a ``Literal``, so a stored
version it does not read is rejected rather than misread.

Metadata is provenance and never model input.
"""

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, StrictStr, StringConstraints, ValidationError

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping


class RecordMetadata(BaseModel):
    """The metadata every Record shares. A dataset subclasses it to add its own fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="Record metadata")

    subject_id: Annotated[StrictStr, StringConstraints(pattern=r"\S")] | None = None
    """What was measured: a person, a device, or a site such as a weather station."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201 - pydantic's subclass hook
        """Reject a subclass that redefines a shared field.

        Raises:
            TimeFValidationError: If the subclass changes a shared field's type.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in RecordMetadata.model_fields.items():
            if cls.model_fields[name].annotation != field.annotation:
                raise TimeFValidationError(f"{cls.__name__} redefines the shared metadata field {name!r}")

    def to_json(self) -> JsonMapping:
        """Return the metadata as TimeNet stores it.

        Every field is written, ``None`` as ``null``, so ``from_json`` reads back the same values; enums are
        written as their values.

        Returns:
            The JSON object.
        """
        stored: dict[str, Any] = self.model_dump(mode="json")
        return stored

    @classmethod
    def from_json(cls, stored: JsonMapping) -> Self:
        """Read stored metadata back as this class.

        Args:
            stored: The JSON object ``to_json`` wrote.

        Returns:
            The metadata.

        Raises:
            TimeFValidationError: If the stored object does not match this class, for example an unknown key or
                another version.
        """
        try:
            return cls.model_validate(stored)
        except ValidationError as error:
            raise TimeFValidationError(f"invalid {cls.__name__}: {error}") from error
