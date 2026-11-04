"""Check one published JSON value at a time, and fail naming the field.

The window files and the retrieval files share these checks. Each returns the value with the exact
type it must have, so a JSON ``true`` never passes as an integer and blank text never passes as text.
"""

from typing import Any

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.trace.weather_events.vocabulary import EventLabel


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


def event_label(value: object, name: str) -> EventLabel:
    """Return the label that a published event type names.

    Returns:
        The label.

    Raises:
        TimeFValidationError: If the value is not an int that EVENT_MAP numbers.
    """
    number = exact_int(value, name)
    try:
        return EventLabel(number)
    except ValueError as exc:
        raise TimeFValidationError(f"{name}: no event type is numbered {number}") from exc
