"""Text values that callers load with ``read()``.

:class:`TextHandle` represents one string. :class:`TextListHandle` represents a list of strings.
Tasks and annotations use these handles to refer to text in memory or in a file.

:class:`InlineText` and :class:`InlineTextList` hold text in memory.
:class:`LazyText` and :class:`LazyTextList` call a supplied function to load the text on request.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, ClassVar, TypeVar, overload

from timenet.errors import TimeFValidationError
from timenet.types.handles import ValueHandle


if TYPE_CHECKING:
    # tasks.py imports this module at runtime.
    from timenet.types.tasks import TargetItem, TargetValue


_Other = TypeVar("_Other")


class _TextBehaviour(ABC):
    """Give both text kinds a length, a truth test, and content equality that avoid needless reads.

    A hash that agrees with content equality would have to read the text, so no handle is hashable.
    ``_content_key()`` gives a hashable key without a read instead. Dataclass subclasses must set
    ``eq=False`` to keep this comparison.
    """

    __slots__ = ()
    __hash__ = None

    _native: ClassVar[type[str] | type[list]]
    """The native type this kind reads as. A handle compares equal to values of this type."""

    @property
    @abstractmethod
    def length(self) -> int:
        """The value of ``len(read())``, known without reading the text.

        That is the string's number of code points, or the list's number of items. ``__eq__`` skips
        the read when lengths differ, so a stored handle must report its length the same way.
        """

    @abstractmethod
    def read(self) -> str | list[str]:
        """Return the native string or string list."""

    @abstractmethod
    def _content_key(self) -> Hashable:
        """Return a hashable key for the text, without reading it.

        Equal keys mean equal text, so a caller can group content by key, as the writer does for
        annotation contents. Text held in memory is its own key: a string, or a tuple for a list. A
        stored handle gives where its payload lives, which never equals the key of text in memory.
        Equal text in different places can therefore get different keys, but different text never
        shares one.
        """

    def __bool__(self) -> bool:
        """Test whether the text is nonempty without reading it.

        Returns:
            True when the string or list contains at least one item.
        """
        return self.length > 0

    def __eq__(self, other: object) -> bool:
        """Compare text content with a handle of the same kind or a native value of its type.

        Returns:
            Whether both contents are equal. A handle equals itself, and values of different length
            compare unequal, without a read. A handle of the other kind is never equal; any other
            value is not comparable.
        """
        if self is other:
            return True
        if isinstance(other, _TextBehaviour):
            return other._native is self._native and self.length == other.length and self.read() == other.read()
        # The union narrows the type for the checker; _native then picks this kind's type.
        if isinstance(other, str | list) and isinstance(other, self._native):
            return self.length == len(other) and self.read() == other
        return NotImplemented


class TextHandle(_TextBehaviour, ValueHandle[str]):
    """Refer to one string: a task prompt, rationale, or text target, or an annotation value."""

    __slots__ = ()
    _native = str

    @abstractmethod
    def read(self) -> str:
        """Return the string, from memory or read through from storage."""


class TextListHandle(_TextBehaviour, ValueHandle[list[str]]):
    """Refer to a list of strings, such as a recording's labels in one annotation value."""

    __slots__ = ()
    _native = list

    @abstractmethod
    def read(self) -> list[str]:
        """Return the list of strings, from memory or read through from storage."""


@dataclass(frozen=True, eq=False, slots=True)
class InlineText(TextHandle):
    """Hold one string in memory. It is always loaded."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TimeFValidationError(f"inline text must be a string, got {type(self.value).__name__}")

    @property
    def is_loaded(self) -> bool:
        """Always True, since the string is held in memory."""
        return True

    @property
    def length(self) -> int:
        """The string's length."""
        return len(self.value)

    def read(self) -> str:
        """Return the string."""
        return self.value

    def _content_key(self) -> str:
        """Return the string, which is its own key."""
        return self.value


@dataclass(frozen=True, eq=False, slots=True)
class InlineTextList(TextListHandle):
    """Hold a list of strings in memory. It is always loaded."""

    value: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.value, list) or not all(isinstance(item, str) for item in self.value):
            raise TimeFValidationError("an inline text list must be a list of strings")

    @property
    def is_loaded(self) -> bool:
        """Always True, since the list is held in memory."""
        return True

    @property
    def length(self) -> int:
        """The number of strings in the list."""
        return len(self.value)

    def read(self) -> list[str]:
        """Return the list of strings."""
        return self.value

    def _content_key(self) -> tuple[str, ...]:
        """Return the strings as a tuple, the hashable form of the list."""
        return tuple(self.value)


@dataclass(frozen=True, eq=False, slots=True)
class LazyText(TextHandle):
    """Read one string through the supplied function on each request.

    A connector can use this handle to load text from a source file.
    The writer stores the text in a separate payload file so readers can load it on request.
    """

    function: Callable[[], str]
    """A function that returns the string."""
    _: KW_ONLY
    length: int
    """The number of characters in the string."""

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TimeFValidationError(
                f"a lazy text handle takes a function that returns a string, got {type(self.function).__name__}"
            )
        _check_length(self.length)

    @property
    def is_loaded(self) -> bool:
        """Always False."""
        return False

    def read(self) -> str:
        """Return the string from a new call to the function.

        Raises:
            TimeFValidationError: If the result is not a string or its length differs from
                the supplied length.
        """
        value = self.function()
        if not isinstance(value, str):
            raise TimeFValidationError(f"a lazy text function must return a string, got {type(value).__name__}")
        if len(value) != self.length:
            raise TimeFValidationError(
                f"a lazy text function returned {len(value)} characters, but the handle was given length {self.length}"
            )
        return value

    def _content_key(self) -> tuple[str, int, int]:
        """Return a content key based on the supplied function and length."""
        return ("lazy", id(self.function), self.length)


@dataclass(frozen=True, eq=False, slots=True)
class LazyTextList(TextListHandle):
    """Read a list of strings through the supplied function on each request.

    A connector can use this handle to load strings from a source file.
    The writer stores the list in a separate payload file so readers can load it on request.
    """

    function: Callable[[], list[str]]
    """A function that returns the list of strings."""
    _: KW_ONLY
    length: int
    """The number of strings in the list."""

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TimeFValidationError(
                f"a lazy text list handle takes a function that returns a list of strings, "
                f"got {type(self.function).__name__}"
            )
        _check_length(self.length)

    @property
    def is_loaded(self) -> bool:
        """Always False."""
        return False

    def read(self) -> list[str]:
        """Return the list from a new call to the function.

        Raises:
            TimeFValidationError: If the result is not a list of strings or its length differs
                from the supplied length.
        """
        value = self.function()
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TimeFValidationError("a lazy text list function must return a list of strings")
        if len(value) != self.length:
            raise TimeFValidationError(
                f"a lazy text list function returned {len(value)} strings, but the handle was given length {self.length}"
            )
        return value

    def _content_key(self) -> tuple[str, int, int]:
        """Return a content key based on the supplied function and length."""
        return ("lazy", id(self.function), self.length)


def _check_length(length: object) -> None:
    """Check that the supplied length is a non-negative integer.

    Raises:
        TimeFValidationError: If the length is a boolean, is not an integer, or is less than zero.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise TimeFValidationError(f"a lazy text handle takes a non-negative integer length, got {length!r}")


def optional_text(value: str | TextHandle | None) -> TextHandle | None:
    """Wrap a native string in memory and keep an existing string handle.

    Returns:
        A text handle, or ``None`` when the optional field is absent.

    Raises:
        TimeFValidationError: If the input is neither a string nor a ``TextHandle``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return InlineText(value)
    if isinstance(value, TextHandle):
        return value
    raise TimeFValidationError(f"text fields take a string or a TextHandle, got {type(value).__name__}")


@overload
def read_value(value: TextHandle) -> str: ...
@overload
def read_value(value: TextListHandle) -> list[str]: ...
@overload
def read_value(value: TextHandle | None) -> str | None: ...
@overload
def read_value(value: TextHandle | TextListHandle) -> str | list[str]: ...
@overload
def read_value(value: "TargetItem") -> "TargetValue": ...
@overload
def read_value(value: TextHandle | TextListHandle | _Other) -> str | list[str] | _Other: ...
def read_value(value: object) -> object:
    """Read a text handle and return any other value unchanged, ``None`` included.

    Returns:
        The handle's string or list of strings, or the value itself when it is not text.
    """
    if isinstance(value, TextHandle | TextListHandle):
        return value.read()
    return value


@overload
def read_values(values: None) -> None: ...
@overload
def read_values(values: Iterable[TextHandle]) -> tuple[str, ...]: ...
@overload
def read_values(values: Iterable[TextListHandle]) -> tuple[list[str], ...]: ...
@overload
def read_values(values: Iterable["TargetItem"]) -> tuple["TargetValue", ...]: ...
@overload
def read_values(
    values: Iterable[TextHandle | TextListHandle | _Other],
) -> tuple[str | list[str] | _Other, ...]: ...
def read_values(values: Iterable[object] | None) -> tuple[object, ...] | None:
    """Read every text handle among ``values`` and return every other item unchanged.

    Each item is read as :func:`read_value` reads it.

    Returns:
        The items in order, or ``None`` when ``values`` is ``None``.

    Raises:
        TimeFValidationError: If ``values`` is a bare string, which would split into characters.
    """
    if values is None:
        return None
    if isinstance(values, str):
        raise TimeFValidationError("read_values takes a sequence of items; use read_value for one string")
    return tuple(read_value(value) for value in values)
