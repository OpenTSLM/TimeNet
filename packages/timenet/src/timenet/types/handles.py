"""The read interface shared by values that may be held in memory or read from storage.

Text handles live in :mod:`timenet.types.text`. This module also holds :class:`ArrayHandle`, which
refers to an array with one value per step, and the classes that hold such an array in memory or
read it through a function: :class:`NumpyArray`, :class:`ArrowArray`, and :class:`CallableArray`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pyarrow as pa

from timenet.errors import TimeFNullValuesError, TimeFValidationError


if TYPE_CHECKING:
    # Stored values and their locations live in timenet.format, which imports this module.
    from timenet.format.stored_values import PayloadLocation


_Value_co = TypeVar("_Value_co", covariant=True)


class ValueHandle(ABC, Generic[_Value_co]):
    """Refer to one value and read it only on request.

    A handle built in memory holds its value. A stored handle can read through to storage on each
    ``read()``.
    """

    # Empty slots let slotted subclasses stay free of a per-instance __dict__.
    __slots__ = ()

    # Type checkers see a read-only property, which a stored handle overrides with a field or a
    # property. Declaring it for them only keeps get_type_hints() free of timenet.format imports.
    if TYPE_CHECKING:

        @property
        def _location(self) -> PayloadLocation | None:
            """Internal: where the value's bytes are stored, for writers, or ``None`` if not stored.

            Packed values and loaded arrays need no I/O. An unloaded array resolves its chunks
            through its reader, in batches.
            """
    else:
        _location = None

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the value is held in memory, so ``read()`` needs no storage access."""

    @abstractmethod
    def read(self) -> _Value_co:
        """Return the value, from memory or read through from storage."""


class ArrayHandle(ValueHandle[pa.Array]):
    """Refer to one array with one value per step, as Arrow, as NumPy, or as a range of steps.

    A handle returns the values as it holds or stores them, nulls included, and checks them against
    no spec.
    """

    __slots__ = ()

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return the half-open step range ``[start, stop)``.

        The default slices :meth:`read`, so a ``stop`` past the end reads to the end. A stored
        handle reads only the chunks that cover the range.

        Args:
            start: First step, inclusive, and not negative.
            stop: Last step, exclusive, and not before ``start``.

        Returns:
            The steps' values, in the form :meth:`read` returns.
        """
        return self.read().slice(start, stop - start)

    def read_numpy(self) -> np.ndarray:
        """Return the values as a read-only NumPy array, shaped ``(n_steps, *value_shape)``.

        The default converts :meth:`read`. A handle whose data is NumPy already returns it without
        building Arrow.

        Returns:
            The values, in the NumPy dtype of their Arrow type.

        Raises:
            TimeFNullValuesError: If the values contain nulls, which NumPy cannot hold. Read them
                with :meth:`read` instead.
        """
        values = self.read()
        if values.null_count:
            raise TimeFNullValuesError(f"{values.null_count} of {len(values)} values are null, which NumPy cannot hold")
        if isinstance(values, pa.FixedShapeTensorArray):
            array = tensor_to_numpy(values)
        else:
            array = values.to_numpy(zero_copy_only=False)
        # A zero-copy conversion is read-only already; a converted copy is made read-only too.
        array.flags.writeable = False
        return array


def tensor_to_numpy(values: pa.FixedShapeTensorArray) -> np.ndarray:
    """Convert a fixed-shape tensor array to NumPy, booleans included.

    ``to_numpy_ndarray()`` rejects boolean tensors, so this reshapes the flattened storage instead.

    Returns:
        The values shaped ``(n_steps, *value_shape)``.

    Raises:
        TimeFNullValuesError: If a step holds a null element, since a step can be null only as a
            whole.
    """
    flat = values.storage.flatten()
    if flat.null_count:
        raise TimeFNullValuesError("tensor values may have nulls only for whole timesteps")
    dense = flat.to_numpy(zero_copy_only=False).reshape((len(values), *values.type.shape))
    if values.type.permutation:
        dense = dense.transpose((0, *(axis + 1 for axis in values.type.permutation)))
    return dense


def _read_only(values: np.ndarray) -> np.ndarray:
    """Return a new array over the same buffer, which nobody can make writable.

    The array reads a read-only buffer, so ``flags.writeable = True`` fails on it. Each call returns
    its own array object, so changing one leaves the held array as it is.
    """
    return np.asarray(memoryview(values).toreadonly())


def _numpy_to_arrow(values: np.ndarray) -> pa.Array:
    """Convert non-empty NumPy values to Arrow, as a fixed-shape tensor array past one dimension.

    Returns:
        The Arrow array, sharing a contiguous numeric buffer.
    """
    if values.ndim > 1:
        return pa.FixedShapeTensorArray.from_numpy_ndarray(values)
    return pa.array(values)


@dataclass(frozen=True, eq=False, slots=True)
class NumpyArray(ArrayHandle):
    """Hold a boolean, integer, or float NumPy array in memory, one value per step along the first axis.

    The handle is always loaded and builds Arrow only when needed: :meth:`read_numpy` needs no Arrow,
    and :meth:`read_steps` converts only the steps it returns. Treat the array passed in, and every
    array a read returns, as read-only.
    """

    values: np.ndarray
    _arrow: pa.Array | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        values = self.values
        if not isinstance(values, np.ndarray):
            raise TimeFValidationError(f"a NumPy array handle holds a NumPy array, got {type(values).__name__}")
        if values.ndim == 0 or values.dtype.kind not in "biuf":
            raise TimeFValidationError(
                "NumPy values need one value per step along the first axis and a boolean, integer, or "
                f"float dtype; got shape {values.shape} and dtype {values.dtype}"
            )
        if values.dtype.kind == "b" or not values.flags.c_contiguous:
            values = values.copy()
        object.__setattr__(self, "values", _read_only(values))

    def __getstate__(self) -> tuple[np.ndarray]:
        # Pickle only the NumPy values. read() rebuilds Arrow when it is needed.
        return (self.values,)

    def __setstate__(self, state: tuple[np.ndarray]) -> None:
        (values,) = state
        object.__setattr__(self, "values", _read_only(values))
        object.__setattr__(self, "_arrow", None)
        object.__setattr__(self, "_lock", Lock())

    @property
    def is_loaded(self) -> bool:
        """Return True, since the array is held in memory."""
        return True

    def read(self) -> pa.Array:
        """Return the values as Arrow.

        The first call builds the Arrow array, a fixed-shape tensor array past one dimension, once
        across threads, and later calls return it.
        """
        arrow = self._arrow
        if arrow is None:
            with self._lock:
                arrow = self._arrow
                if arrow is None:
                    arrow = _numpy_to_arrow(self.values)
                    object.__setattr__(self, "_arrow", arrow)  # noqa: PLC2801 - fills the frozen handle's cache
        return arrow

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return the half-open step range ``[start, stop)``, converting only those steps to Arrow.

        Once :meth:`read` has built the Arrow array, this slices it instead.

        Returns:
            The steps' values, in the form :meth:`read` returns.
        """
        arrow = self._arrow
        if arrow is not None:
            return arrow.slice(start, stop - start)
        steps = self.values[start:stop]
        if len(steps) or steps.ndim == 1:
            return _numpy_to_arrow(steps)
        # Arrow builds no empty tensor array from NumPy, so an empty range of tensor steps converts
        # the first step and keeps none of it.
        return _numpy_to_arrow(self.values[:1]).slice(0, 0)

    def read_numpy(self) -> np.ndarray:
        """Return the NumPy values as a new read-only view, without building Arrow."""
        return _read_only(self.values)


@dataclass(frozen=True, eq=False, slots=True)
class ArrowArray(ArrayHandle):
    """Hold an Arrow array in memory. It is always loaded."""

    values: pa.Array

    def __post_init__(self) -> None:
        if not isinstance(self.values, pa.Array):
            raise TimeFValidationError(f"an Arrow array handle holds a pyarrow Array, got {type(self.values).__name__}")

    @property
    def is_loaded(self) -> bool:
        """Return True, since the array is held in memory."""
        return True

    def read(self) -> pa.Array:
        """Return the array."""
        return self.values


@dataclass(frozen=True, eq=False, slots=True)
class CallableArray(ArrayHandle):
    """Read an Arrow array through a function that takes no arguments.

    Every read calls the function, and the handle keeps nothing, so it is never loaded. Two handles
    over the same function object compare equal, since they read the same values. Equality and the
    hash use the function's identity and never call its own ``==``.
    """

    function: Callable[[], pa.Array]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TimeFValidationError(
                f"a callable array handle takes a function that returns Arrow values, got {type(self.function).__name__}"
            )

    @property
    def is_loaded(self) -> bool:
        """Return False, since every read calls the function."""
        return False

    def read(self) -> pa.Array:
        """Return the values from a new call to the function."""
        return self.function()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallableArray):
            return NotImplemented
        return self.function is other.function

    def __hash__(self) -> int:
        return hash(id(self.function))
