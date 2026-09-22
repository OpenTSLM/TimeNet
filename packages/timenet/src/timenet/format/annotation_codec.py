"""Codec between an annotation's value and the typed value columns of ``annotation_contents``.

An annotation value is a string, an integer, a float, a boolean, a list of strings, or absent for a
pure marker. Each shape has its own column so DuckDB can filter on it and the reader hands back the
exact Python type that was written: an integer stays an integer and a boolean stays a boolean.
"""

from collections.abc import Sequence
from typing import Any

from timenet.errors import TimeFFormatError, TimeFValidationError


VALUE_COLUMNS = ("value_kind", "text_value", "integer_value", "float_value", "boolean_value", "text_list_value")
"""The ``annotation_contents`` columns that together hold one value."""

VALUE_KINDS = ("text", "integer", "float", "boolean", "text_list")
"""Every ``value_kind`` a stored annotation may carry. A marker has NULL."""


def encode_annotation_value(value: object, *, label: str) -> dict[str, Any]:
    """Project one annotation value onto the value columns.

    Args:
        value: The annotation's value, or ``None`` for a marker.
        label: How to name the annotation in an error message.

    Returns:
        One entry per :data:`VALUE_COLUMNS`, all ``None`` for a marker.

    Raises:
        TimeFValidationError: If the value is not one of the supported shapes.
    """
    columns: dict[str, Any] = dict.fromkeys(VALUE_COLUMNS)
    if value is None:
        return columns
    # bool before int, since bool is an int subclass.
    if isinstance(value, bool):
        columns.update(value_kind="boolean", boolean_value=value)
    elif isinstance(value, int):
        columns.update(value_kind="integer", integer_value=value)
    elif isinstance(value, float):
        columns.update(value_kind="float", float_value=value)
    elif isinstance(value, str):
        columns.update(value_kind="text", text_value=value)
    elif isinstance(value, Sequence) and all(isinstance(item, str) for item in value):
        columns.update(value_kind="text_list", text_list_value=list(value))
    else:
        raise TimeFValidationError(
            f"{label} has an unsupported value {value!r}; use a string, integer, float, boolean, or list of strings"
        )
    return columns


def decode_annotation_value(  # noqa: PLR0913, PLR0917 - one positional argument per stored column
    value_kind: str | None,
    text_value: str | None,
    integer_value: int | None,
    float_value: float | None,
    boolean_value: bool | None,
    text_list_value: list[str] | None,
) -> Any:
    """Rebuild one annotation value from its columns.

    Returns:
        The value in the Python type it was written with, or ``None`` for a marker.

    Raises:
        TimeFFormatError: If the kind is unknown or its column is NULL.
    """
    if value_kind is None:
        return None
    stored = {
        "text": text_value,
        "integer": integer_value,
        "float": float_value,
        "boolean": boolean_value,
        "text_list": text_list_value,
    }
    if value_kind not in stored:
        raise TimeFFormatError(f"annotation value has unknown kind {value_kind!r}")
    value = stored[value_kind]
    if value is None:
        raise TimeFFormatError(f"annotation value of kind {value_kind!r} has no stored value")
    return value
