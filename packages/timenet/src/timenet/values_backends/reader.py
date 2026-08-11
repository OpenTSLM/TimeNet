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
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec


class BaseValuesReader(ABC):
    """Reads a series' values from a storage handle given its index rows."""

    @abstractmethod
    def load(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            version: The opened version handle; reads flow through its filesystem/store.
            rows: The series' index rows, sorted by ``chunk_idx``; each holds ``chunk_file``,
                ``chunk_major_idx``, and ``chunk_minor_idx``.
            spec: The series' spec, for backends whose decoding depends on shape/dtype.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """

    @abstractmethod
    def load_range(
        self, version: DatasetVersion, rows: list[dict], start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
        """Read only the steps of one series in the half-open step range ``[start, stop)``.

        Returns:
            The requested steps in their canonical Arrow representation.
        """

    @abstractmethod
    def load_time_offsets(self, version: DatasetVersion, rows: list[dict]) -> pa.Array:
        """Read and concatenate one irregular series' per-value time offsets.

        Args:
            version: The opened version handle; reads flow through its filesystem/store.
            rows: The series' index rows, sorted by ``chunk_idx``.

        Returns:
            One int64 microsecond time offset per value.
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
        from timenet.values_backends.parquet.reader import ParquetValuesReader  # noqa: PLC0415

        return ParquetValuesReader()
    if name == ValuesBackend.ZARR:
        from timenet.values_backends.zarr.reader import ZarrValuesReader  # noqa: PLC0415

        return ZarrValuesReader()
    raise TimeFValidationError(
        f"unknown values_backend {name!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
    )
