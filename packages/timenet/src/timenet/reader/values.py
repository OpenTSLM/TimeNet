"""Reader-side values backend seam: the abstract contract plus its factory.

The inverse of :mod:`timenet.writer.values`. A :class:`BaseValuesReader` takes the index rows for one
series (sorted by ``chunk_idx``, each carrying the backend's chunk locator) and returns the series' 1-D
float32 Arrow array. The :class:`TimeFReader` picks the backend from the manifest's ``values_backend``
tag and never imports a specific storage library itself. Concrete readers live in their own modules —
:mod:`timenet.reader.parquet_values` (the default).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend


class BaseValuesReader(ABC):
    """Reads a series' values from the version directory given its index rows."""

    @abstractmethod
    def load(self, root: Path, rows: list[dict]) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            The series' 1-D float32 values.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open handles or caches held for the reader's lifetime."""


def make_values_reader(name: str) -> BaseValuesReader:
    """Construct the reader-side values backend named ``name``.

    Args:
        name: The manifest ``values_backend`` tag.

    Returns:
        The constructed reader backend.

    Raises:
        TimeFValidationError: If ``name`` is not a known backend.
    """
    if name == ValuesBackend.PARQUET:
        from timenet.reader.parquet_values import ParquetValuesReader

        return ParquetValuesReader()
    raise TimeFValidationError(
        f"unknown values_backend {name!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
    )
