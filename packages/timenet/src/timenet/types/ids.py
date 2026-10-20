"""Entity ids and the per-dataset counters that hand them out.

TimeF ids are positive int64 values. Records, Signals, and Tasks have a public, read-only ``id``. A
dataset sets it when the object is added, and the reader sets it when it reads the object back.
Sources, axes, and annotations have no public id. The writer gives them storage ids when it writes
their rows. Ids are unique per kind within one dataset and stay the same across edits. They do not
follow storage order, and two datasets can use the same id.
"""

from collections.abc import Mapping
from enum import StrEnum, unique
import numbers
import operator
from typing import TypeAlias

from timenet.errors import TimeFValidationError
from timenet.types.clock import INT64_MAX


EntityId: TypeAlias = int
"""The id of one TimeF entity, a positive int64."""


@unique
class IdKind(StrEnum):
    """The entity kinds that each have their own id counter."""

    RECORD = "record"
    SOURCE = "source"
    SIGNAL = "signal"
    AXIS = "axis"
    CONTENT = "content"
    OCCURRENCE = "occurrence"
    TASK = "task"


class IdCounters:
    """The last id handed out for each entity kind in one dataset."""

    def __init__(self, last: Mapping[IdKind, EntityId] | None = None) -> None:
        """Start the counters at zero or continue from stored values.

        Args:
            last: The last id used per kind, for example as read from a stored version.
        """
        self._last = dict.fromkeys(IdKind, 0)
        for kind, value in (last or {}).items():
            if value != 0:
                self.claim(IdKind(kind), value)

    def next(self, kind: IdKind) -> EntityId:
        """Return the next unused id of one kind.

        Returns:
            The new id.

        Raises:
            TimeFValidationError: If the kind has used every int64 id.
        """
        if self._last[kind] >= INT64_MAX:
            raise TimeFValidationError(f"no {kind} ids are left in this dataset")
        self._last[kind] += 1
        return self._last[kind]

    def claim(self, kind: IdKind, value: object) -> EntityId:
        """Record an id that is already in use, so that :meth:`next` never returns it.

        Returns:
            ``value`` as a Python int.
        """
        number = check_id(value, f"{kind} id")
        self._last[kind] = max(self._last[kind], number)
        return number

    def last(self) -> dict[IdKind, EntityId]:
        """Return the last id used per kind, the state that a stored version keeps."""
        return dict(self._last)


def check_id(value: object, what: str) -> EntityId:
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


def assigned_id(value: EntityId | None, owner: str) -> EntityId:
    """Return the id a dataset or the reader gave an object.

    Args:
        value: The object's private id, ``None`` until it has one.
        owner: The object's type name, used in the error message.

    Returns:
        The id.

    Raises:
        TimeFValidationError: If the object has no id yet.
    """
    if value is None:
        raise TimeFValidationError(f"this {owner} has no id yet; add it to a dataset to give it one")
    return value
