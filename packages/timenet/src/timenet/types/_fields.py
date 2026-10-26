"""Descriptor fields that normalize constructor inputs and later assignments."""

from collections.abc import Callable
from typing import Generic, TypeVar


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


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
