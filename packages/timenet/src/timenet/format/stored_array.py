"""Signal arrays stored as chunks in the Parquet or Zarr values files of a release."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Protocol, Self

import pyarrow as pa

from timenet.types import TimeSeriesSpec
from timenet.types.handles import ArrayHandle
from timenet.types.reference_keys import OwnerKey
from timenet.values_backends.types import StoredChunk


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkLocation:
    """Locate a signal array stored as chunks in the values files of one release.

    The catalog records each chunk's file, coordinates, and length, so the array's place in storage
    is known without opening a values file.
    """

    owner: OwnerKey | None = None
    """Release that stores the chunks, or ``None`` for the release whose catalog records them."""
    chunks: tuple[StoredChunk, ...]
    """The array's chunks in step order."""


class ChunkReader(Protocol):
    """Find and read the stored chunks of one release's signal arrays.

    A release's ``TimeFReader`` is one: it finds the chunks of a batch of signals at a time, and reads
    them from whichever release stores them.
    """

    _version: DatasetVersion
    """The release whose catalog records the chunk locations."""

    def _chunk_location(self, signal_key: int, signal_id: str) -> ChunkLocation:
        """Return where one signal's chunks are stored, without reading them."""
        ...

    def _read_array(self, location: ChunkLocation, spec: TimeSeriesSpec) -> pa.Array:
        """Return the values of every chunk at a location."""
        ...

    def _read_array_steps(self, location: ChunkLocation, spec: TimeSeriesSpec, start: int, stop: int) -> pa.Array:
        """Return the half-open step range ``[start, stop)`` of the chunks at a location."""
        ...


@dataclass(frozen=True, slots=True)
class StoredArray(ArrayHandle):
    """Refer to one signal's array, stored as chunks in the values files of a release.

    The handle holds the release's reader and the signal's catalog key, ID, and spec. A loaded
    handle also holds the array and the location of its chunks, so it reads, slices, and locates
    them without storage access. An unloaded handle holds no values: every read goes to the chunks,
    and ``read_steps`` reads only the chunks that cover the range. Handles from one reader for one
    signal are equal, whether loaded or not.
    """

    reader: ChunkReader = dataclass_field(repr=False)
    """Reader of the release whose catalog records the signal."""
    signal_key: int
    """Internal catalog key of the signal."""
    signal_id: str
    """ID of the signal."""
    spec: TimeSeriesSpec
    """Spec that the stored values decode to."""
    _array: pa.Array | None = dataclass_field(default=None, init=False, repr=False, compare=False)
    """The values, kept once a read loads them."""
    _chunks: ChunkLocation | None = dataclass_field(default=None, init=False, repr=False, compare=False)
    """Where the kept values are stored, captured when they were loaded."""

    @property
    def is_loaded(self) -> bool:
        """Return whether the handle keeps the values, so reads need no storage access."""
        return self._array is not None

    @property
    def _location(self) -> ChunkLocation:
        """Return where the chunks are stored, without reading them.

        A loaded handle keeps its location. An unloaded one asks its reader, which finds the
        locations of a batch of signals at a time and keeps them until it closes.
        """
        chunks = self._chunks
        return self.reader._chunk_location(self.signal_key, self.signal_id) if chunks is None else chunks

    def read(self) -> pa.Array:
        """Return the kept values, or the values of every chunk."""
        array = self._array
        return self.reader._read_array(self._location, self.spec) if array is None else array

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return the half-open step range ``[start, stop)``.

        A loaded handle slices the values it keeps. An unloaded one reads only the chunks that
        cover the range.

        Returns:
            The steps' values, in the form :meth:`read` returns.
        """
        array = self._array
        if array is not None:
            return array.slice(start, stop - start)
        return self.reader._read_array_steps(self._location, self.spec, start, stop)

    def _fill(self, array: pa.Array, location: ChunkLocation) -> None:
        """Hold the values and their location, so the handle is loaded from then on.

        Args:
            array: The values, in buffers of their own.
            location: Where the values are stored.
        """
        object.__setattr__(self, "_array", array)  # noqa: PLC2801 - fills the frozen handle before it is handed out
        object.__setattr__(self, "_chunks", location)  # noqa: PLC2801 - fills the frozen handle before it is handed out

    def __deepcopy__(self, memo: dict) -> Self:
        return self
