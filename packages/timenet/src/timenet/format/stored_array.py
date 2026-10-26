"""Signal arrays stored as chunks in the Parquet or Zarr values files of a release."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Protocol, Self

import pyarrow as pa

from timenet.types import TimeSeriesSpec
from timenet.types.handles import ArrayHandle
from timenet.types.reference_keys import SourceKey
from timenet.values_backends.types import StoredChunk


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkLocation:
    """Locate a signal array stored as chunks in the values files of one release.

    The catalog records each chunk's file, coordinates, and length, so the array's place in storage
    is known without opening a values file.
    """

    source: SourceKey | None = None
    """The payload source that stores the chunks, or ``None`` for the release whose catalog records them."""
    chunks: tuple[StoredChunk, ...]
    """The array's chunks in step order."""


class ChunkReader(Protocol):
    """Find and read the stored chunks of one release's signal arrays.

    A release's ``TimeFReader`` is one: it finds the chunks of a batch of signals at a time, and reads
    them from whichever release stores them.
    """

    _version: DatasetVersion
    """The release whose catalog records the chunk locations."""

    def _chunk_location(self, signal_id: int) -> ChunkLocation:
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

    The handle holds the release's reader and the signal's ID and spec, not the values, so it is
    never loaded and every read goes to the chunks. ``read_steps`` reads only the chunks that cover
    the range. Handles from one reader for one signal are equal.
    """

    reader: ChunkReader = dataclass_field(repr=False)
    """Reader of the release whose catalog records the signal."""
    signal_id: int
    """ID of the signal in that release's catalog."""
    spec: TimeSeriesSpec
    """Spec that the stored values decode to."""

    @property
    def is_loaded(self) -> bool:
        """Return False, since every read goes to the chunks."""
        return False

    @property
    def _location(self) -> ChunkLocation:
        """Return where the chunks are stored, without reading them.

        The reader finds them for a batch of signals at a time and keeps them until it closes. The
        handle keeps nothing.
        """
        return self.reader._chunk_location(self.signal_id)

    def read(self) -> pa.Array:
        """Return the values of every chunk."""
        return self.reader._read_array(self._location, self.spec)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return the half-open step range ``[start, stop)``, reading only the chunks that cover it.

        Returns:
            The steps' values, in the form :meth:`read` returns.
        """
        return self.reader._read_array_steps(self._location, self.spec, start, stop)

    def __deepcopy__(self, memo: dict) -> Self:
        return self
