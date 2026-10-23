"""NumPy values retained until an Arrow read needs conversion."""

from threading import Lock

import numpy as np
import pyarrow as pa

from timenet.dataset.materialized_values.base import MaterializedValues
from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec


class NumpyMaterializedValues(MaterializedValues):
    """Retain numeric buffers and cache Arrow conversion across threads.

    Boolean inputs are copied because Arrow packs their bits into a separate buffer. NumPy reads
    return writable boolean copies, as Arrow does. Other dtypes share read-only views, and callers
    must treat the input buffers as read-only while this loader uses them.
    """

    def __init__(self, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values)
        self._values = values.copy() if values.dtype.kind == "b" else values.view()
        self._arrow: pa.Array | None = None
        self._lock = Lock()

    @classmethod
    def from_values(cls, spec: TimeSeriesSpec, values: np.ndarray) -> "NumpyMaterializedValues":
        """Normalize dtype and check the per-timestep shape.

        Returns:
            A loader for values converted with NumPy's casting rules.

        Raises:
            TimeFValidationError: If values do not match the spec's per-timestep shape.
        """
        if values.ndim == 0 or values.shape[1:] != spec.value_shape:
            raise TimeFValidationError(
                f"series for {spec.spec_type!r} needs shape (n_steps, *{spec.value_shape}), got {values.shape}"
            )
        return cls(np.ascontiguousarray(values, dtype=np.dtype(spec.dtype)))

    def __len__(self) -> int:
        return len(self._values)

    def to_arrow(self) -> pa.Array:
        """Return the cached Arrow array, converting once on the first call.

        Returns:
            The complete Arrow values.
        """
        if self._arrow is None:
            with self._lock:
                if self._arrow is None:
                    self._arrow = self._to_arrow(self._values)
        return self._arrow

    def __reduce__(self) -> tuple[type["NumpyMaterializedValues"], tuple[np.ndarray]]:
        # Rebuild the lock and discard the cache in pickle/deepcopy and spawned workers.
        return type(self), (self._values,)

    @staticmethod
    def _to_arrow(values: np.ndarray) -> pa.Array:
        if values.ndim > 1:
            return pa.FixedShapeTensorArray.from_numpy_ndarray(values)
        return pa.array(values)

    def to_numpy(self) -> np.ndarray:
        """Return a fresh view, or a writable copy for booleans.

        Returns:
            Values with independent shape metadata and the retained dtype.
        """
        if self._values.dtype.kind == "b":
            return self._values.copy()
        # A read-only buffer prevents setflags(write=True), including after unpickling. Each call
        # also gets its own array object, so changing its shape cannot reshape the stored values.
        return np.asarray(memoryview(self._values).toreadonly())

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Convert only the requested slice, leaving the full-series cache untouched.

        Returns:
            The requested Arrow values.
        """
        if self._arrow is not None:
            return self._arrow.slice(start, stop - start)
        return self._to_arrow(self._values[start:stop])
