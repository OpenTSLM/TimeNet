"""Generate and check entity ids.

TimeF entity ids (records, sources, signals, axes, annotations, tasks) are positive int64 values.

An id at or above :data:`FINAL_ID_LIMIT` (``2**52``) is provisional. :func:`new_id` hands out
provisional ids so that objects can refer to each other before they are written. The writer stores
a final id below ``2**52`` for every provisional id and leaves the objects unchanged (see
:mod:`timenet.format.final_ids`). A first write numbers each kind ``1..N`` in the order it stores
the objects. An id below ``2**52`` is final and the writer keeps it.

A generated id is ``(tag << 23) | counter``. Each process draws a random 30-bit tag with its top bit
set and counts up from there, so generated ids from different processes do not collide in practice.
Generated ids stay below ``2**53``, so they survive a float64 column (for example a nullable pandas
column) and JSON readers that parse numbers as doubles.
"""

import itertools
import numbers
import operator
import os
import secrets
import threading

from timenet.errors import TimeFValidationError
from timenet.types.clock import INT64_MAX


_COUNTER_BITS = 23
_COUNTER_LIMIT = 1 << _COUNTER_BITS
_TAG_BITS = 30

FINAL_ID_LIMIT = 1 << 52
"""Ids below this are final. Ids at or above it are provisional and the writer replaces them."""


class _IdState:
    """The per-process tag and counter behind :func:`new_id`."""

    def __init__(self) -> None:
        # Kept across forks, so a child never draws a tag its parent used.
        self.used_tags: set[int] = set()
        self.reset()

    def reset(self) -> None:
        """Make a new lock, draw a fresh tag, and restart the counter."""
        self.lock = threading.Lock()
        self.redraw()

    def redraw(self) -> None:
        """Draw a tag this process has not used yet and restart the counter."""
        tag = secrets.randbits(_TAG_BITS - 1) | (1 << (_TAG_BITS - 1))
        while tag in self.used_tags:
            tag = secrets.randbits(_TAG_BITS - 1) | (1 << (_TAG_BITS - 1))
        self.used_tags.add(tag)
        self.tag = tag
        self.counter = itertools.count()


_state = _IdState()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_state.reset)


def new_id() -> int:
    """Return a new provisional entity id, the default id for every TimeF entity.

    Returns:
        An int in ``[2**52, 2**53)``, unique within this process.
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
    if type(value) is int and 0 < value <= INT64_MAX:
        return value
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        number = operator.index(value)
        if 0 < number <= INT64_MAX:
            return number
    raise TimeFValidationError(f"{what} must be a positive int64 id, got {value!r}")


def is_provisional(value: int) -> bool:
    """Return whether an entity id is provisional, so the writer replaces it with a final id.

    Args:
        value: A valid entity id.

    Returns:
        ``True`` if ``value`` is at or above :data:`FINAL_ID_LIMIT`.
    """
    return value >= FINAL_ID_LIMIT
