"""Validate source values, and report the field and the source location when a value is invalid.

A connector reads each source value through one of these functions. Each function returns the value
with the exact type that it must have, or raises :class:`~timenet.errors.TimeNetConnectorError`.
Boolean values are not accepted as numbers. Floating-point values are not accepted as integers.
Blank strings are rejected unless explicitly allowed. No function repairs or defaults a value. An
integer becomes a float only if the conversion keeps its exact value.

The functions name the field in the message. :func:`error_context` adds the source location, such as
the file and the row, so the full message reads ``train.jsonl line 4: answer: expected text, got ''``::

    for number, value in enumerate(rows, start=1):
        with error_context(f"{path.name} line {number}"):
            row = as_object(value, "row")
            answer = text_field(row, "answer")

There are two groups of functions. ``as_object``, ``as_list``, ``field``, ``text``, ``text_field``,
and ``numbers`` check a decoded value, for example from JSON. ``parse_int`` and
``parse_whole_number`` parse a text cell, for example from CSV.
"""

from collections.abc import Generator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import math
import re
import reprlib
from typing import Any

import numpy as np

from timenet.errors import TimeNetConnectorError


_SHOWN_LENGTH = 60
"""The longest source value that an error message shows in full, so that a message stays one line."""
_INTEGER = re.compile(r"[+-]?[0-9]+")
"""Plain ASCII digits with an optional sign: no spaces, underscores, or other digit scripts."""
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
"""A decimal number such as ``-1.5``, ``.5``, or ``2e-3``, with no ``nan`` or ``inf``."""
_MAX_DIGITS = 4300
"""Python's default limit on the digits of an integer that it parses from text."""


@contextmanager
def error_context(where: str) -> Generator[None]:
    """Add a source location to the connector errors raised inside the block.

    The error keeps its type and its traceback. Nested contexts give messages such as
    ``file: row: message``.

    Args:
        where: The place in the source, such as ``"train.jsonl line 4"`` or ``"exams.csv row 17"``.

    Yields:
        Nothing. The block runs inside the context.

    Raises:
        TimeNetConnectorError: The error from the block, with ``where`` before its message.
    """
    try:
        yield
    except TimeNetConnectorError as error:
        message = str(error.args[0]) if error.args else str(error)
        error.args = (f"{where}: {message}", *error.args[1:])
        raise


_PRINTABLE_INT_BITS = 14_000
"""An int of fewer bits has fewer than 4300 digits, so ``repr`` can write it."""


class _ShortRepr(reprlib.Repr):
    """Write an int too long for ``repr`` as its approximate number of digits.

    Python refuses to write an int of more than 4300 digits as text. Python 3.12 and later make
    ``reprlib`` write a placeholder for it, and 3.11 raises ``ValueError``. This gives one message
    on every version.
    """

    def repr_int(self, x: int, level: int) -> str:
        if x.bit_length() < _PRINTABLE_INT_BITS:
            return super().repr_int(x, level)
        return f"an int of about {int(x.bit_length() * math.log10(2)) + 1} digits"


_REPR = _ShortRepr()
"""Writes at most a few items of a large list or dict, so that a message does not grow with the source."""
# Python 3.11 takes no arguments to Repr(), so set the limits as attributes.
_REPR.maxlevel = 2
_REPR.maxstring = _REPR.maxother = _REPR.maxlong = _SHOWN_LENGTH


def _shown(value: object) -> str:
    """Return a short ``repr`` of a source value for an error message."""
    shown = _REPR.repr(value)
    return shown if len(shown) <= _SHOWN_LENGTH else f"{shown[: _SHOWN_LENGTH - 3]}..."


def _expected(name: str, expected: str, value: object) -> str:
    """Return the message for a value that is not what ``name`` must hold."""
    return f"{name}: expected {expected}, got {_shown(value)}"


def _in_range(value: float, name: str, expected: str, minimum: float | None, maximum: float | None) -> None:
    """Raise if ``value`` is below ``minimum`` or above ``maximum``.

    Raises:
        TimeNetConnectorError: If the value is out of range.
    """
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        low = "" if minimum is None else f" of at least {minimum}"
        high = "" if maximum is None else f"{' and' if low else ' of'} at most {maximum}"
        raise TimeNetConnectorError(_expected(name, f"{expected}{low}{high}", value))


def _exact_float(value: int) -> float | None:
    """Return ``value`` as a float if the float holds it exactly, else ``None``."""
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and int(number) == value else None


def _text_cell(cell: object, name: str, expected: str) -> str:
    """Return ``cell`` if it is text. A reader can give ``None`` or a number for a cell, against the type.

    Raises:
        TimeNetConnectorError: If the cell is not text.
    """
    if not isinstance(cell, str):
        raise TimeNetConnectorError(_expected(name, expected, cell))
    return cell


def as_object(value: object, name: str) -> dict[str, Any]:
    """Return a decoded JSON object.

    Args:
        value: The decoded value.
        name: The field name for the message.

    Returns:
        The object.

    Raises:
        TimeNetConnectorError: If the value is not an object.
    """
    if not isinstance(value, dict):
        raise TimeNetConnectorError(_expected(name, "an object", value))
    return value


def as_list(value: object, name: str, *, nonempty: bool = False) -> list[Any]:
    """Return a decoded JSON list.

    Args:
        value: The decoded value.
        name: The field name for the message.
        nonempty: Whether the list must hold at least one item.

    Returns:
        The list.

    Raises:
        TimeNetConnectorError: If the value is not a list, or is empty when ``nonempty`` is set.
    """
    if not isinstance(value, list) or (nonempty and not value):
        raise TimeNetConnectorError(_expected(name, "a nonempty list" if nonempty else "a list", value))
    return value


def field(source: dict[str, Any], name: str) -> Any:
    """Return a required field of a decoded object, without checking its value.

    Args:
        source: The object.
        name: The field name.

    Returns:
        The field's value.

    Raises:
        TimeNetConnectorError: If the object has no such field.
    """
    if name not in source:
        # str() so that a StrEnum field name reads as its value, not its repr.
        raise TimeNetConnectorError(f"missing field {str(name)!r}")
    return source[name]


def text(value: object, name: str, *, allow_blank: bool = False) -> str:
    """Return a string that is not only whitespace, unless ``allow_blank`` is set.

    Args:
        value: The decoded value.
        name: The field name for the message.
        allow_blank: Whether an empty or blank string passes.

    Returns:
        The string, unchanged.

    Raises:
        TimeNetConnectorError: If the value is not a string, or is blank when that is not allowed.
    """
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise TimeNetConnectorError(_expected(name, "a string" if allow_blank else "text", value))
    return value


def text_field(source: dict[str, Any], name: str, *, allow_blank: bool = False) -> str:
    """Return a text field of a decoded object, as :func:`text` checks it.

    Args:
        source: The object.
        name: The field name.
        allow_blank: Whether an empty or blank string passes.

    Returns:
        The string, unchanged.
    """
    return text(field(source, name), name, allow_blank=allow_blank)


def numbers(
    values: object,
    name: str,
    *,
    dtype: type[np.int64] | type[np.float64] = np.float64,
    nonempty: bool = True,
) -> np.ndarray:
    """Return a decoded list of numbers as an array, such as the values of one signal.

    Args:
        values: The decoded value, which must be a list.
        name: The field name for the message.
        dtype: ``np.float64`` accepts ints and finite floats. ``np.int64`` accepts only ints that
            fit in 64 bits.
        nonempty: Whether the list must hold at least one number.

    Returns:
        The values, as a one-dimensional array of ``dtype``.

    Raises:
        TypeError: If ``dtype`` is neither ``np.int64`` nor ``np.float64``.
        TimeNetConnectorError: If the value is not a list of such numbers, for example if it holds a
            JSON ``true``, a string, a nested list, or a NaN.
    """
    if dtype is not np.int64 and dtype is not np.float64:
        raise TypeError(f"numbers takes dtype np.int64 or np.float64, got {dtype!r}")
    expected = "an integer" if dtype is np.int64 else "a finite number"
    items = as_list(values, name, nonempty=nonempty)
    allowed = {int} if dtype is np.int64 else {int, float}
    if not set(map(type, items)) <= allowed:
        index = next(index for index, item in enumerate(items) if type(item) not in allowed)
        raise TimeNetConnectorError(_expected(f"{name}[{index}]", expected, items[index]))
    # A float holds no int of magnitude 2**1024 or more.
    low, high = (-(2**63), 2**63 - 1) if dtype is np.int64 else (-(2**1024) + 1, 2**1024 - 1)
    for index, item in enumerate(items):
        # Every int up to 2**53 has an exact float, so only a larger one needs the slower check.
        if type(item) is not int or -(2**53) <= item <= 2**53:
            continue
        if not low <= item <= high:
            raise TimeNetConnectorError(_expected(f"{name}[{index}]", f"{expected} in range", item))
        if dtype is np.float64 and _exact_float(item) is None:
            raise TimeNetConnectorError(_expected(f"{name}[{index}]", "a number that a float holds exactly", item))
    array = np.array(items, dtype=dtype)
    if dtype is np.float64 and not np.isfinite(array).all():
        index = int(np.flatnonzero(~np.isfinite(array))[0])
        raise TimeNetConnectorError(_expected(f"{name}[{index}]", expected, items[index]))
    return array


def parse_int(cell: str, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse a text cell that holds an integer in plain digits, such as ``"42"`` or ``"-3"``.

    The cell must not hold spaces, underscores, or a decimal point. Use :func:`parse_whole_number`
    for a source that writes whole numbers as floats.

    Args:
        cell: The text cell.
        name: The column name for the message.
        minimum: The smallest allowed value, if any.
        maximum: The largest allowed value, if any.

    Returns:
        The integer.

    Raises:
        TimeNetConnectorError: If the cell is not text, is not such an integer, or is out of range.
    """
    cell = _text_cell(cell, name, "an integer")
    if not _INTEGER.fullmatch(cell) or len(cell.lstrip("+-")) > _MAX_DIGITS:
        raise TimeNetConnectorError(_expected(name, "an integer", cell))
    value = int(cell)
    _in_range(value, name, "an integer", minimum, maximum)
    return value


def parse_whole_number(cell: str, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse an integer from text, also in decimal or scientific notation, such as ``"56.0"`` or ``"5.6e1"``.

    The function reads the cell as an exact decimal and rejects a fractional value instead of
    rounding it. ``"56.5"`` and
    ``"2.0000000000000001"`` fail, and ``"9007199254740993.0"`` gives exactly 9007199254740993.

    Args:
        cell: The text cell.
        name: The column name for the message.
        minimum: The smallest allowed value, if any.
        maximum: The largest allowed value, if any.

    Returns:
        The integer.

    Raises:
        TimeNetConnectorError: If the cell is not text, is not a whole number, or is out of range.
    """
    cell = _text_cell(cell, name, "a whole number")
    if _INTEGER.fullmatch(cell):
        return parse_int(cell, name, minimum=minimum, maximum=maximum)
    if not _DECIMAL.fullmatch(cell):
        raise TimeNetConnectorError(_expected(name, "a whole number", cell))
    try:
        number = Decimal(cell)
        whole = number.adjusted() < _MAX_DIGITS and number == number.to_integral_value()
    except InvalidOperation:
        # An exponent beyond what Decimal can hold, such as "1e-99999999999999999999".
        whole = False
    if not whole:
        raise TimeNetConnectorError(_expected(name, "a whole number", cell))
    value = int(number)
    _in_range(value, name, "a whole number", minimum, maximum)
    return value
