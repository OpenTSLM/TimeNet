"""Check one published JSON value at a time, and fail naming the field.

The window files and the retrieval files share these checks. Each returns the value with the exact
type it must have, so a JSON ``true`` never passes as an integer and blank text never passes as text.
"""

from typing import Any

import numpy as np

from timenet.errors import TimeFValidationError


def as_object(value: object) -> dict[str, Any]:
    """Return a published JSON object.

    Returns:
        The object.

    Raises:
        TimeFValidationError: If the value is not an object.
    """
    if not isinstance(value, dict):
        raise TimeFValidationError(f"expected a JSON object, got {type(value).__name__}")
    return value


def field(published: dict[str, Any], name: str) -> Any:
    """Return a field of a published object, naming the field if it is absent.

    Returns:
        The field's value, unchecked.

    Raises:
        TimeFValidationError: If the object lacks the field.
    """
    if name not in published:
        # str() so that a StrEnum field reads as its name, not its repr.
        raise TimeFValidationError(f"missing field {str(name)!r}")
    return published[name]


def text(value: object, name: str) -> str:
    """Return a published text that has a visible character.

    Returns:
        The text, as published.

    Raises:
        TimeFValidationError: If the value is not a string or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise TimeFValidationError(f"{name}: expected text, got {value!r}")
    return value


def text_field(published: dict[str, Any], name: str) -> str:
    """Return a published text field that has a visible character.

    Returns:
        The text, as published.
    """
    return text(field(published, name), name)


def exact_int(value: object, name: str) -> int:
    """Return a published integer, not a JSON ``true`` or a float.

    Returns:
        The integer.

    Raises:
        TimeFValidationError: If the value is not an int.
    """
    if type(value) is not int:
        raise TimeFValidationError(f"{name}: expected an integer, got {value!r}")
    return value


def degrees(value: object, name: str, limit: float) -> float:
    """Return a finite number within ``limit`` degrees of zero.

    Returns:
        The number, as a float.

    Raises:
        TimeFValidationError: If the value is not such a number.
    """
    # A JSON true is an int to Python, so the type is checked exactly. NaN fails the range check.
    if not isinstance(value, int | float) or isinstance(value, bool) or not abs(value) <= limit:
        raise TimeFValidationError(f"{name}: expected a number from -{limit} to {limit}, got {value!r}")
    return float(value)


def exact_numbers(values: list[Any], dtype: type[np.int64] | type[np.float64], name: str, problem: str) -> np.ndarray:
    """Convert published numbers of the exact JSON types ``dtype`` allows, or fail naming the problem.

    Returns:
        The values as an array.

    Raises:
        TimeFValidationError: If a value has another type, such as a JSON true or a string, or is too big.
    """
    allowed = {int} if dtype is np.int64 else {int, float}
    if not set(map(type, values)) <= allowed:
        raise TimeFValidationError(f"{name}: expected {problem}")
    try:
        return np.array(values, dtype=dtype)
    except OverflowError as exc:
        raise TimeFValidationError(f"{name}: expected {problem}, got a value out of range") from exc
