"""Strict pydantic field types and helpers shared by the models of TRACE's published rows."""

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, FiniteFloat, Strict, StrictInt, StrictStr, StringConstraints

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.trace.weather_events.vocabulary import EventLabel, SkyCover


# Strict types: a JSON true or "1" must not pass as a number, and a blank string not as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
Number = Annotated[FiniteFloat, Strict()]
Numbers = Annotated[list[Number], Field(min_length=1)]
# The codes stay ints, which the loader hands to the codebook as indices, so they are not SkyCover members.
SkyCodes = Annotated[list[Annotated[StrictInt, Field(ge=0, lt=len(SkyCover))]], Field(min_length=1)]
# Matches YYYY-MM-DDTHH:MM:SS, such as 2023-01-03T00:00:00, with no fractional seconds or time zone.
# This rejects inputs NumPy would otherwise parse, such as "now", "NaT", or a "Z" suffix.
Timestamp = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")]


def exact_int(value: object) -> object:
    """Reject what an IntEnum field would otherwise take as an int: ``true``, ``1.0``, or ``"1"``.

    Returns:
        The value, unchanged, for the enum to read.

    Raises:
        TimeFValidationError: If the value is not an int.
    """
    if type(value) is not int:
        raise TimeFValidationError(f"expected an integer, got {value!r}")
    return value


EventLabelField = Annotated[EventLabel, BeforeValidator(exact_int)]


def nest(data: Any, keys: tuple[str, ...], field: str) -> Any:
    """Move the published ``keys`` of a row under one ``field``, the way a model groups them.

    Returns:
        The row with those keys moved, or ``data`` unchanged if it is not a dict.

    Raises:
        TimeFValidationError: If the published row already has a key named ``field``.
    """
    if not isinstance(data, dict):
        return data
    if field in data:
        raise TimeFValidationError(f"a published row has no {field!r} key")
    nested = {key: data[key] for key in keys if key in data}
    return {key: value for key, value in data.items() if key not in keys} | {field: nested}
