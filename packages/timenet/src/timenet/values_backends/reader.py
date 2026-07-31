"""Reader-side values backend seam: the abstract contract plus its factory.

The inverse of :mod:`timenet.values_backends.writer`. A :class:`BaseValuesReader` takes the index rows for one
series (sorted by ``chunk_idx``, each carrying the backend's chunk locator) and returns a primitive or
fixed-shape tensor Arrow array matching the spec. :class:`TimeFReader` picks the backend from the
manifest's ``values_backend`` tag and never imports a specific storage library itself. Concrete readers
live in their own modules — :mod:`timenet.values_backends.parquet.reader` (the default) and
:mod:`timenet.values_backends.zarr.reader`.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend


class BaseValuesReader(ABC):
    """Reads a series' values from the version directory given its index rows."""

    @abstractmethod
    def load(self, root: Path, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            root: The version directory.
            rows: The series' index rows, sorted by ``chunk_idx``; each holds ``chunk_file``,
                ``chunk_major_idx``, and ``chunk_minor_idx``.
            spec: The series' spec, for backends whose decoding depends on shape/dtype.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """

    @abstractmethod
    def load_range(self, root: Path, rows: list[dict], start: int, stop: int, spec: TimeSeriesSpec) -> pa.Array:
        """Read only the steps of one series in the half-open step range ``[start, stop)``.

        Returns:
            The requested steps in their canonical Arrow representation.
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
        from timenet.values_backends.parquet.reader import ParquetValuesReader

        return ParquetValuesReader()
    if name == ValuesBackend.ZARR:
        from timenet.values_backends.zarr.reader import ZarrValuesReader

        return ZarrValuesReader()
    raise TimeFValidationError(
        f"unknown values_backend {name!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
    )
