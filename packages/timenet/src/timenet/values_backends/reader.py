"""Reader-side values backend seam: the abstract contract plus its factory.

This module mirrors :mod:`timenet.values_backends.writer`. A :class:`BaseValuesReader` takes the index rows
for one series and returns a primitive or fixed-shape tensor Arrow array that matches the spec. The rows are
sorted by ``chunk_idx``, and each row carries the backend's chunk locator. :class:`TimeFReader` picks the
backend from the manifest's ``values_backend`` tag. It never imports a specific storage library itself.
Concrete readers live in their own modules: :mod:`timenet.values_backends.parquet.reader` (the default) and
:mod:`timenet.values_backends.zarr.reader`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend


if TYPE_CHECKING:
    from collections.abc import Sequence

    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec
    from timenet.values_backends.types import StoredChunk


class BaseValuesReader(ABC):
    """Reads a series' values from a storage handle given its index rows."""

    @abstractmethod
    def load(self, version: DatasetVersion, rows: Sequence[StoredChunk], spec: TimeSeriesSpec) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            version: The opened version handle. Reads flow through its filesystem/store.
            rows: The series' index rows, sorted by ``chunk_idx``. Each holds a file path
                and backend coordinates in ``data_index``.
            spec: The series' spec, for backends whose decoding depends on shape/dtype.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """

    @abstractmethod
    def load_range(
        self, version: DatasetVersion, rows: Sequence[StoredChunk], start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
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
        ImportError: If the selected optional backend is not installed.
        ModuleNotFoundError: If another dependency of the backend is missing.
    """
    if name == ValuesBackend.PARQUET:
        from timenet.values_backends.parquet.reader import ParquetValuesReader  # noqa: PLC0415

        return ParquetValuesReader()
    if name == ValuesBackend.ZARR:
        try:
            from timenet.values_backends.zarr.reader import ZarrValuesReader  # noqa: PLC0415
        except ModuleNotFoundError as error:
            if error.name != "zarr":
                raise
            raise ImportError("Zarr values require the 'timenet[zarr]' extra") from error

        return ZarrValuesReader()
    raise TimeFValidationError(
        f"unknown values_backend {name!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
    )
