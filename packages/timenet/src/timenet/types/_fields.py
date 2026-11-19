"""Descriptor fields that normalize constructor inputs and later assignments."""

from collections.abc import Callable, Iterable, Mapping
from typing import Generic, TypeVar

from timenet.errors import TimeFValidationError


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")
_Item = TypeVar("_Item")


def as_tuple(value: Iterable[_Item]) -> tuple[_Item, ...]:
    """Convert the input of a collection field to a tuple.

    Returns:
        The items in the order given.

    Raises:
        TimeFValidationError: If ``value`` is a string, a mapping, or not iterable.
    """
    if isinstance(value, (str, Mapping)) or not isinstance(value, Iterable):
        raise TimeFValidationError(
            f"a collection field takes an iterable of items such as a list or a tuple, got {type(value).__name__}"
        )
    return tuple(value)


class ValueField(Generic[_Input, _Output]):
    """Expose a normalized field type while accepting convenient constructor inputs."""

    def __init__(self, convert: Callable[[_Input], _Output], default: _Output) -> None:
        self._convert = convert
        self._default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self._slot = f"_value_{name}"

    def __get__(self, instance: object | None, owner: type | None = None) -> _Output:
        if instance is None:
            return self._default
        return instance.__dict__[self._slot]

    def __set__(self, instance: object, value: _Input) -> None:
        # dataclasses.field(default=descriptor) passes the descriptor for an omitted argument.
        instance.__dict__[self._slot] = self._default if value is self else self._convert(value)
