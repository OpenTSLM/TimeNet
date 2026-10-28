"""A cell that a frozen value handle keeps a loaded value in."""

from typing import Generic, Self, TypeVar

from timenet.errors import TimeNetError


_T = TypeVar("_T")


class WriteOnce(Generic[_T]):
    """Hold a value that is set at most once.

    A new cell is empty. :meth:`set` stores a value and raises if the cell already holds one, and
    :meth:`setdefault` stores a value only in an empty cell, so the value a cell holds never changes.
    A frozen handle holds a cell as a field, so it keeps a value it loads without assigning to its
    own fields. Pickling or copying a cell carries its value, while a shallow copy of a handle shares
    the handle's cell.
    """

    __slots__ = ("_value",)

    def __init__(self, value: _T | None = None) -> None:
        self._value = value

    @property
    def value(self) -> _T | None:
        """Return the value the cell holds, or ``None`` if it is empty."""
        return self._value

    def set(self, value: _T) -> None:
        """Hold a value in an empty cell.

        Raises:
            TimeNetError: If the cell already holds a value.
        """
        if self._value is not None:
            # A second set is a bug in TimeNet, not bad input, so it raises TimeNet's base error.
            raise TimeNetError(f"cannot set a write-once cell that already holds a {type(self._value).__name__}")
        self._value = value

    def setdefault(self, value: _T) -> _T:
        """Hold a value unless the cell already holds one.

        Returns:
            The value the cell holds: ``value``, or the value it held already.
        """
        held = self._value
        if held is None:
            self._value = held = value
        return held

    def __reduce__(self) -> tuple[type[Self], tuple[_T | None]]:
        # Rebuild the cell from its value, which also pickles it at every protocol despite its slots.
        return type(self), (self._value,)
