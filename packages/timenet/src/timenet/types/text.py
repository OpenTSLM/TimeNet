"""Explicit reads for text stored in memory or in external payloads."""

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from timenet.errors import TimeFValidationError


_Text_co = TypeVar("_Text_co", covariant=True)
_NativeText = TypeVar("_NativeText", bound=str | list[str])
_Value = TypeVar("_Value")


@runtime_checkable
class TextValue(Protocol[_Text_co]):
    """Describe text without loading it; read it explicitly to obtain its native value."""

    @property
    def value_type(self) -> Literal["str", "list"]:
        """Return the native value's type without reading it."""
        ...

    @property
    def length(self) -> int:
        """Return the string length or list length without reading it."""
        ...

    def read(self) -> _Text_co:
        """Return the native string or string list."""
        ...


@dataclass(frozen=True)
class InlineText(Generic[_NativeText]):
    """Hold an in-memory text value behind the same read interface as packed text."""

    value: _NativeText

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) and not (
            isinstance(self.value, list) and all(isinstance(item, str) for item in self.value)
        ):
            raise TimeFValidationError("inline text must be a string or a list of strings")

    @property
    def value_type(self) -> Literal["str", "list"]:
        """Return the native value's type without reading it."""
        return "str" if isinstance(self.value, str) else "list"

    @property
    def length(self) -> int:
        """Return the string length or list length."""
        return len(self.value)

    def read(self) -> _NativeText:
        """Return the in-memory text value."""
        return self.value


def optional_text(value: str | TextValue[str] | None) -> TextValue[str] | None:
    """Wrap native text and preserve existing string handles.

    Returns:
        A text handle, or ``None`` when the optional field is absent.

    Raises:
        TimeFValidationError: If the input is not a string or a string handle.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return InlineText(value)
    if isinstance(value, TextValue) and value.value_type == "str":
        return value
    raise TimeFValidationError("text fields require a string or a string handle")


def read_value(value: _Value | TextValue[_Value]) -> _Value:
    """Read a text handle or preserve a native scalar, object, or absent value.

    Returns:
        The materialized value, independent of how it was stored.
    """
    return cast("_Value", value.read() if isinstance(value, TextValue) else value)
