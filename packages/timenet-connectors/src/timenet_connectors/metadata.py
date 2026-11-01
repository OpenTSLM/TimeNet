"""A Record's metadata: one dataset's own typed fields.

A connector declares its Record metadata as one subclass of ``RecordMetadata``. ``Record.metadata`` holds
what ``to_json`` returns, one flat JSON object::

    {"version": 1, "coordinates": ..., "sampling": "random", "windows": [...]}

Design choices:

- One class per dataset, one level of keys: the stored JSON is plain keys and values, with no type tags or
  wrappers, so a reader without the connector, such as a consumer who has not installed it, can still read it.
  A field may still hold a structured value, such as a frozen pydantic model or a tuple of them.
- Values keep their types: the fields are strict pydantic fields, not a bag of strings. A wrong type is
  rejected, not converted.
- Metadata is frozen against assignment. A Record keeps the dict that ``to_json`` returns, not this object,
  so ``from_json`` gives a new object each time. Use tuples and frozen models for structured values.
- Reading needs the dataset's class: ``CGMRecordMetadata.from_json(...)``.

A dataset whose metadata may change shape over time declares a ``version`` field as a ``Literal`` from its
first release, so a stored version it does not read is rejected rather than misread.

The subjects of a Record are not metadata. As the TimeF docs say, they are a list-valued Record annotation
with the key ``subject_ids``.

Metadata is provenance and never model input.
"""

import json
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError

from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping


class RecordMetadata(BaseModel):
    """The base of one dataset's Record metadata. A dataset subclasses it to add its own fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def to_json(self) -> JsonMapping:
        """Return the metadata as TimeNet stores it.

        Every field is written, ``None`` as ``null``, so ``from_json`` reads back the same values. Enums are
        written as their values.

        Returns:
            The JSON object.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, stored: JsonMapping) -> Self:
        """Read stored metadata back as this class.

        ``to_json`` writes every field, so a stored object without one of them was written by another shape of
        the class. It is rejected, not completed with a default.

        Args:
            stored: The JSON object ``to_json`` wrote.

        Returns:
            The metadata.

        Raises:
            TimeFValidationError: If the stored object does not match this class, for example a missing or an
                unknown key, a value of the wrong type, or another version.
        """
        missing = sorted(cls.model_fields.keys() - stored.keys())
        if missing:
            raise TimeFValidationError(f"invalid {cls.__name__}: the stored object lacks {missing}")
        try:
            # JSON mode reads enum values, lists as tuples and ISO dates as strict mode allows them in JSON.
            return cls.model_validate_json(json.dumps(stored))
        except ValidationError as error:
            raise TimeFValidationError(f"invalid {cls.__name__}: {error}") from error
