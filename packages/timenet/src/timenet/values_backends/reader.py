"""Reader-side values backend seam: the abstract contract plus its factory.

This module mirrors :mod:`timenet.values_backends.writer`. A :class:`BaseValuesReader` takes the index rows
for one series and returns a primitive or fixed-shape tensor Arrow array that matches the spec. The rows are
sorted by ``chunk_idx``, and each row holds the backend's own address for one chunk. :class:`TimeFReader`
picks the backend from the manifest's ``values_backend`` tag. It never imports a specific storage library
itself. Concrete readers live in their own modules: :mod:`timenet.values_backends.parquet.reader` (the
default) and :mod:`timenet.values_backends.zarr.reader`.

A windowed read finds its chunks with :class:`SeriesChunks` and :func:`window_chunks`. Every backend shares
them, because the offsets come from the index rows and not from the stored values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxtyping import Int64
import numpy as np
import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec


@dataclass
class SeriesChunks:
    """One series' index rows, plus the chunk offsets a windowed read bisects.

    :class:`~timenet.reader.reader.TimeFReader` holds one of these per series for as long as it
    holds the rows. The offsets are built once, and every window of that series reuses them.

    A full read takes the rows alone. Only a windowed read needs the offsets.
    """

    rows: list[dict]
    """The series' index rows, sorted by ``chunk_idx``."""
    _offsets: Int64[np.ndarray, " boundary"] | None = None
    """The rows' chunk offsets, built on the first windowed read of this series."""

    def offsets(self) -> Int64[np.ndarray, " boundary"]:
        """Return the value index each chunk starts at, then the series' total.

        This method builds the offsets on the first call and keeps them for later windows.

        Returns:
            ``len(rows) + 1`` offsets. The last one is the series' length.
        """
        if self._offsets is None:
            offsets = np.zeros(len(self.rows) + 1, dtype=np.int64)
            counts = np.fromiter((row["n_values"] for row in self.rows), dtype=np.int64, count=len(self.rows))
            np.cumsum(counts, out=offsets[1:])
            self._offsets = offsets
        return self._offsets


def window_chunks(offsets: Int64[np.ndarray, " boundary"], start: int, stop: int) -> tuple[int, int, int]:
    """Find the chunks a step window crosses, by bisecting a series' chunk offsets.

    A ``stop`` past the last step clamps to it. A ``start`` at or past the last step crosses no
    chunk.

    Args:
        offsets: The series' chunk offsets, from :meth:`SeriesChunks.offsets`.
        start: First step of the window, inclusive. A negative one clamps to zero.
        stop: One past the window's last step.

    Returns:
        ``(first, last, bounded_stop)``: the half-open chunk range ``rows[first:last]`` that the
        window crosses, and ``stop`` clamped to the series' length. ``first == last`` when the
        window covers no value.
    """
    bounded_stop = min(stop, int(offsets[-1]))
    if start >= bounded_stop:
        return 0, 0, bounded_stop
    first = max(int(np.searchsorted(offsets, start, side="right")) - 1, 0)
    last = int(np.searchsorted(offsets, bounded_stop, side="left"))
    return first, last, bounded_stop


class BaseValuesReader(ABC):
    """Reads a series' values from a storage handle given its index rows."""

    @abstractmethod
    def load(self, version: DatasetVersion, rows: list[dict], spec: TimeSeriesSpec) -> pa.Array:
        """Read and concatenate one series' chunk values.

        Args:
            version: The opened version handle. Reads flow through its filesystem/store.
            rows: The series' index rows, sorted by ``chunk_idx``. Each holds ``chunk_file``,
                ``chunk_major_idx``, and ``chunk_minor_idx``.
            spec: The series' spec, for backends whose decoding depends on shape/dtype.

        Returns:
            The series values in the spec's canonical Arrow representation.
        """

    @abstractmethod
    def load_range(
        self, version: DatasetVersion, chunks: SeriesChunks, start: int, stop: int, spec: TimeSeriesSpec
    ) -> pa.Array:
        """Read only the steps of one series in the half-open step range ``[start, stop)``.

        Args:
            version: The opened version handle. Reads flow through its filesystem/store.
            chunks: The series' index rows and their chunk offsets. The read bisects the offsets to
                find the chunks it needs.
            start: First step of the window, inclusive.
            stop: One past the window's last step. A ``stop`` past the last step clamps to it.
            spec: The series' spec, for backends whose decoding depends on shape/dtype.

        Returns:
            The requested steps in their canonical Arrow representation.
        """

    @abstractmethod
    def load_time_offsets(self, version: DatasetVersion, rows: list[dict]) -> pa.Array:
        """Read and concatenate one irregular series' per-value time offsets.

        Args:
            version: The opened version handle. Reads flow through its filesystem/store.
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
