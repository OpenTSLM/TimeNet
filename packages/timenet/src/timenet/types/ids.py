"""Entity ids and the per-dataset counters that hand them out.

TimeF entity ids (records, sources, signals, axes, annotation contents and occurrences, tasks) are
positive int64 values. An object has no id until it is added to a dataset. The dataset then gives it
the next id of its kind. Ids are unique per kind within one dataset and stay the same across edits.
They do not follow storage order, and two datasets can use the same id.
"""

from collections.abc import Mapping
from enum import StrEnum, unique
import numbers
import operator
from typing import Protocol, TypeAlias

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
        """Record an explicit id, so that :meth:`next` never returns it.

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
        TimeFValidationError: If ``value`` is ``None`` (the object is not in a dataset yet), or not an
            integral number (bool excluded) in ``0 < value < 2**63``.
    """
    if type(value) is int and 0 < value <= INT64_MAX:
        return value
    if value is None:
        raise TimeFValidationError(f"{what} is not set; add the object to a dataset to give it an id")
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        number = operator.index(value)
        if 0 < number <= INT64_MAX:
            return number
    raise TimeFValidationError(f"{what} must be a positive int64 id, got {value!r}")


def check_optional_id(value: object, what: str) -> EntityId | None:
    """Return ``value`` unchanged if it is ``None``, else check it with :func:`check_id`.

    Returns:
        ``None`` or ``value`` as a Python int.
    """
    return None if value is None else check_id(value, what)


class _HasId(Protocol):
    @property
    def id(self) -> EntityId | None: ...


def entity_id(entity: _HasId) -> EntityId:
    """Return the id of an object that must already be in a dataset.

    Returns:
        The object's id.
    """
    return check_id(entity.id, f"{type(entity).__name__} id")
