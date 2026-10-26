"""The read interface shared by values that may be held in memory or read from storage.

Text handles live in :mod:`timenet.types.text`. This module also holds :class:`ArrayHandle`, which
refers to an array with one value per step, and the classes that hold such an array in memory or
read it through a function: :class:`NumpyArray`, :class:`ArrowArray`, and :class:`CallableArray`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
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

    # Declared for type checkers only, so get_type_hints() needs no import of timenet.format.
    if TYPE_CHECKING:
        _location: PayloadLocation | None
    _location = None
    """Internal: where the value's bytes are stored, known without reading them.

    ``None`` for a value that exists only in memory or behind a function. A stored handle sets it.
    """

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
        """Return the values as NumPy, shaped ``(n_steps, *value_shape)``.

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
            raise TimeFNullValuesError(
                f"{values.null_count} of {len(values)} values are null, which NumPy cannot hold; read them as Arrow"
            )
        if isinstance(values, pa.FixedShapeTensorArray):
            return tensor_to_numpy(values)
        return values.to_numpy(zero_copy_only=False)


def tensor_to_numpy(values: pa.FixedShapeTensorArray) -> np.ndarray:
    """Convert a fixed-shape tensor array to NumPy, booleans included.

    ``to_numpy_ndarray()`` rejects boolean tensors, so this reshapes the flattened storage instead.

    Returns:
        The values shaped ``(n_steps, *value_shape)``.
    """
    dense = values.storage.flatten().to_numpy(zero_copy_only=False).reshape((len(values), *values.type.shape))
    if values.type.permutation:
        dense = dense.transpose((0, *(axis + 1 for axis in values.type.permutation)))
    return dense


def _read_only(values: np.ndarray) -> np.ndarray:
    """Return a read-only view, so a caller of ``read_numpy()`` cannot change the held values."""
    view = values.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, eq=False, slots=True)
class NumpyArray(ArrayHandle):
    """Hold a one-dimensional boolean, integer, or float NumPy array in memory. It is always loaded.

    :meth:`read_numpy` returns the array, read-only, without building Arrow. The first :meth:`read`
    builds the Arrow array and later reads return it. A contiguous numeric array is shared with the
    caller and with Arrow. Boolean and strided arrays, which Arrow copies, are copied here too, so a
    later change to the caller's array cannot make the two reads disagree.
    """

    values: np.ndarray
    _arrow: pa.Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        values = self.values
        if not isinstance(values, np.ndarray):
            raise TimeFValidationError(f"a NumPy array handle holds a NumPy array, got {type(values).__name__}")
        if values.ndim != 1 or values.dtype.kind not in "biuf":
            raise TimeFValidationError(
                "NumPy values must be one-dimensional, one value per step, with a boolean, integer, or "
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

    @property
    def is_loaded(self) -> bool:
        """Return True, since the array is held in memory."""
        return True

    def read(self) -> pa.Array:
        """Return the values as Arrow, built on the first call and kept."""
        arrow = self._arrow
        if arrow is None:
            arrow = pa.array(self.values)
            object.__setattr__(self, "_arrow", arrow)  # noqa: PLC2801 - fills the frozen handle's cache
        return arrow

    def read_numpy(self) -> np.ndarray:
        """Return the NumPy array without building Arrow."""
        return self.values


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


@dataclass(frozen=True, slots=True)
class CallableArray(ArrayHandle):
    """Read an Arrow array through a function that takes no arguments.

    Every read calls the function, and the handle keeps nothing, so it is never loaded. Two handles
    over the same function compare equal, since they read the same values.
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
