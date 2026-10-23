"""The read interface shared by values that may be held in memory or read from storage."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


_Value_co = TypeVar("_Value_co", covariant=True)


class ValueHandle(ABC, Generic[_Value_co]):
    """Refer to one value and read it only on request.

    A handle built in memory holds its value. A stored handle can read through to storage on each
    ``read()``.
    """

    # Empty slots let slotted subclasses stay free of a per-instance __dict__.
    __slots__ = ()

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the value is held in memory, so ``read()`` needs no storage access."""

    @abstractmethod
    def read(self) -> _Value_co:
        """Return the value, from memory or read through from storage."""
