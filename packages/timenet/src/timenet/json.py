"""Strict validation for metadata documented as JSON-compatible."""

from typing import Annotated, cast

from pydantic import (
    BeforeValidator,
    ConfigDict,
    JsonValue,
    StrictStr,
    TypeAdapter,
)
from typing_extensions import TypeAliasType

from timenet.errors import TimeFValidationError


def _reject_float_subclasses(value: object) -> object:
    """Reject float subclasses before Pydantic converts them to built-in floats.

    Returns:
        The original value for Pydantic to validate and copy.

    Raises:
        TimeFValidationError: If any nested value is a float subclass.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, float) and type(item) is not float:
            raise TimeFValidationError("must be a built-in float")
        if isinstance(item, (list, dict)) and id(item) not in seen:
            # Leave cycles intact so Pydantic reports them as validation errors.
            seen.add(id(item))
            pending.extend(item.values() if isinstance(item, dict) else item)
    return value


JsonMapping = TypeAliasType("JsonMapping", dict[StrictStr, JsonValue])
"""A strict JSON object with string keys."""

_JSON_CONFIG = ConfigDict(strict=True, allow_inf_nan=False)
_VALUE_ADAPTER = TypeAdapter(Annotated[JsonValue, BeforeValidator(_reject_float_subclasses)], config=_JSON_CONFIG)
_MAPPING_ADAPTER = TypeAdapter(Annotated[JsonMapping, BeforeValidator(_reject_float_subclasses)], config=_JSON_CONFIG)


def validate_json_value(value: object) -> JsonValue:
    """Validate and copy one strict JSON value.

    Args:
        value: Candidate JSON value.

    Returns:
        A detached hierarchy containing only JSON-native values.
    """
    return _VALUE_ADAPTER.validate_python(value)


def validate_json_mapping(value: object) -> JsonMapping:
    """Validate and copy a string-keyed JSON object.

    Args:
        value: Candidate metadata mapping.

    Returns:
        A detached JSON-native dictionary.
    """
    return cast("JsonMapping", _MAPPING_ADAPTER.validate_python(value))
