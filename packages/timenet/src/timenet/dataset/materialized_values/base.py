"""Common interface for values stored in memory or loaded on demand."""

from abc import ABC, abstractmethod

import numpy as np
import pyarrow as pa


class ValuesLoader(ABC):
    """Values with explicit Arrow, NumPy, and step-range access."""

    def __call__(self) -> pa.Array:
        """Keep the legacy callable interface as an alias for :meth:`to_arrow`.

        Returns:
            The complete Arrow values.
        """
        return self.to_arrow()

    @abstractmethod
    def to_arrow(self) -> pa.Array:
        """Return the retained values as an Arrow array."""

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Return values without missing timesteps."""

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open step range, using a full read when no range loader exists.

        Returns:
            The requested Arrow values.
        """
        return self.to_arrow().slice(start, stop - start)


class MaterializedValues(ValuesLoader):
    """Values already in memory, with a known timestep count."""

    @abstractmethod
    def __len__(self) -> int: ...
