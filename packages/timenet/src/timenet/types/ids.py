"""Generate and check entity ids.

TimeF entity ids (records, sources, signals, axes, annotations, tasks) are positive int64 values.
A generated id is ``(tag << 23) | counter``. Each process draws a random 30-bit tag and counts up
from there, so ids from different processes do not collide in practice. Generated ids stay below
``2**53``, so they survive a float64 column (for example a nullable pandas column) and JSON readers
that parse numbers as doubles.
"""

import itertools
import numbers
import operator
import os
import secrets
import threading

from timenet.errors import TimeFValidationError


_COUNTER_BITS = 23
_COUNTER_LIMIT = 1 << _COUNTER_BITS
_TAG_BITS = 30
_MAX_ID = 1 << 63


class _IdState:
    """The per-process tag and counter behind :func:`new_id`."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Make a new lock, draw a fresh tag, and restart the counter."""
        self.lock = threading.Lock()
        self.redraw()

    def redraw(self) -> None:
        """Draw a fresh tag and restart the counter."""
        self.tag = secrets.randbits(_TAG_BITS - 1) | (1 << (_TAG_BITS - 1))
        self.counter = itertools.count()


_state = _IdState()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_state.reset)


def new_id() -> int:
    """Return a new entity id, the default id for every TimeF entity.

    Returns:
        A positive int below ``2**53``, unique within this process.
    """
    with _state.lock:
        counter = next(_state.counter)
        if counter >= _COUNTER_LIMIT:
            _state.redraw()
            counter = next(_state.counter)
        return (_state.tag << _COUNTER_BITS) | counter


def check_id(value: object, what: str) -> int:
    """Return ``value`` as an int if it is a valid entity id.

    Any integral value is accepted, including NumPy integers read back from Arrow or pandas.

    Args:
        value: The candidate id.
        what: The name of the id, used in the error message.

    Returns:
        ``value`` as a Python int.

    Raises:
        TimeFValidationError: If ``value`` is not an integral number (bool excluded) in
            ``0 < value < 2**63``.
    """
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        number = operator.index(value)
        if 0 < number < _MAX_ID:
            return number
    raise TimeFValidationError(f"{what} must be a positive int64 id, got {value!r}")
