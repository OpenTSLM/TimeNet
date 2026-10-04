"""Strict validation for metadata documented as JSON-compatible."""

from typing import Annotated, Any, cast

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
)
from typing_extensions import TypeAliasType


def _exact_float(value: Any) -> Any:
    if type(value) is not float:
        raise ValueError("must be a built-in float")
    return value


FiniteFloat = Annotated[float, BeforeValidator(_exact_float), Field(allow_inf_nan=False)]
JsonValue = TypeAliasType(
    "JsonValue",
    StrictBool | StrictInt | FiniteFloat | StrictStr | list["JsonValue"] | dict[StrictStr, "JsonValue"] | None,
)
"""A strict JSON value composed only of built-in JSON-native types."""

JsonMapping = TypeAliasType("JsonMapping", dict[StrictStr, JsonValue])
"""A strict JSON object with string keys."""

_VALUE_ADAPTER = TypeAdapter(JsonValue, config=ConfigDict(strict=True))
_MAPPING_ADAPTER = TypeAdapter(JsonMapping, config=ConfigDict(strict=True))


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
