"""Immutable content handles with explicit reads and cheap type descriptors."""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any, Generic, Literal, TypeVar, cast

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types.payload_references import PayloadReference


T = TypeVar("T")
PayloadKind = Literal["bool", "int", "float", "str", "list", "array"]
StoragePolicy = Literal["auto", "external"]


@dataclass(frozen=True)
class PayloadDescriptor:
    """Describe content without reading its body."""

    kind: PayloadKind
    length: int | None = None
    encoded_size: int | None = None
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    nullable: bool = False
    categories: tuple[str, ...] = ()
    dimension_names: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Payload(Generic[T]):
    """Inline or produced content that callers resolve with ``read()``."""

    descriptor: PayloadDescriptor
    storage: StoragePolicy = "auto"
    _value: Any = field(default=None, repr=False)
    _loader: Callable[[], T] | None = field(default=None, repr=False)
    reference: PayloadReference | None = None

    @classmethod
    def from_value(cls, value: T, *, storage: StoragePolicy = "auto") -> "Payload[T]":
        """Wrap a supported native value without changing its semantic type.

        Returns:
            An immutable handle. List reads return a fresh list.

        Raises:
            TimeFValidationError: If the value or storage policy is unsupported.
        """
        if storage not in {"auto", "external"}:
            raise TimeFValidationError(f"unsupported payload storage policy {storage!r}")
        if isinstance(value, Payload):
            return value
        if isinstance(value, bool):
            descriptor = PayloadDescriptor("bool")
        elif isinstance(value, int):
            descriptor = PayloadDescriptor("int")
        elif isinstance(value, float):
            descriptor = PayloadDescriptor("float")
        elif isinstance(value, str):
            descriptor = PayloadDescriptor("str", len(value), len(value.encode("utf-8")))
        elif isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
            descriptor = PayloadDescriptor(
                "list", len(value), len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            )
            return cls(descriptor, storage, tuple(value))
        else:
            raise TimeFValidationError(f"unsupported payload value type {type(value).__name__}")
        return cls(descriptor, storage, value)

    @classmethod
    def from_loader(cls, loader: Callable[[], T], *, descriptor: PayloadDescriptor) -> "Payload[T]":
        """Wrap a producer callback without invoking it.

        Returns:
            A handle that materializes only on an explicit read or write.
        """
        return cls(descriptor, _loader=loader)

    def read(self) -> T:
        """Return the content, explicitly invoking its producer when necessary.

        Returns:
            The native typed value.
        """
        value = self._loader() if self._loader is not None else self._value
        if self.descriptor.kind == "list":
            return cast("T", list(cast("Any", value)))
        return cast("T", value)

    def __call__(self) -> T:
        """Read content through the producer callback interface.

        Returns:
            The native value.
        """
        return self.read()

    def __deepcopy__(self, memo: dict[int, object]) -> "Payload[T]":
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Payload) or self.descriptor != other.descriptor:
            return False
        if self.reference is not None or other.reference is not None:
            return self.reference == other.reference
        if self._loader is not None or other._loader is not None:
            return self._loader is other._loader
        if isinstance(self._value, pa.Array):
            return self._value is other._value
        return self._value == other._value

    def __hash__(self) -> int:
        backing = (
            self.reference.model_dump_json()
            if self.reference is not None
            else (id(self._loader) if self._loader is not None else self._value)
        )
        if isinstance(backing, pa.Array):
            backing = id(backing)
        return hash((self.descriptor, backing))


@dataclass(frozen=True, eq=False)
class ArrayPayload(Payload[pa.Array]):
    """An Arrow array payload with optional bounded reads."""

    _read_range: Callable[[int, int], pa.Array] | None = field(default=None, repr=False)

    @classmethod
    def from_array(cls, value: pa.Array, *, descriptor: PayloadDescriptor) -> "ArrayPayload":
        """Wrap an immutable Arrow array.

        Returns:
            The array handle.
        """
        return cls(descriptor, _value=value)

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[], pa.Array],
        *,
        descriptor: PayloadDescriptor,
        read_range: Callable[[int, int], pa.Array] | None = None,
    ) -> "ArrayPayload":
        """Wrap an array producer and its optional range-read capability.

        Returns:
            A lazy array handle.
        """
        return cls(descriptor, _loader=loader, _read_range=read_range)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open range, using bounded storage reads when available.

        Returns:
            The selected Arrow values.

        Raises:
            TimeFValidationError: If the range is negative or reversed.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError(f"expected 0 <= start <= stop, got start={start}, stop={stop}")
        if self._read_range is not None:
            return self._read_range(start, stop)
        return self.read().slice(start, stop - start)
