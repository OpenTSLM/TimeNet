"""Adapt existing Arrow callables to the common values interface."""

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from timenet.dataset.materialized_values.arrow_values import PyArrowMaterializedValues
from timenet.dataset.materialized_values.base import ValuesLoader


class CallableValuesLoader(ValuesLoader):
    """Wrap a legacy loader without reading or caching its values.

    Adapters for the same callable compare equal, so shared offset streams keep one source identity.
    """

    def __init__(self, loader: Callable[[], pa.Array]) -> None:
        self._loader = loader

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallableValuesLoader):
            return NotImplemented
        return self._loader is other._loader

    def __hash__(self) -> int:
        return id(self._loader)

    def to_arrow(self) -> pa.Array:
        """Read the callable's Arrow values.

        Returns:
            The complete Arrow array.
        """
        return self._loader()

    def to_numpy(self) -> np.ndarray:
        """Read dense NumPy values through the callable's Arrow output.

        Returns:
            Values shaped ``(n_steps, *value_shape)``.
        """
        return PyArrowMaterializedValues(self.to_arrow()).to_numpy()

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Use the callable's range method when available.

        Returns:
            The requested Arrow values.
        """
        read_steps = getattr(self._loader, "read_steps", None)
        if read_steps is not None:
            return read_steps(start, stop)
        return super().read_steps(start, stop)
