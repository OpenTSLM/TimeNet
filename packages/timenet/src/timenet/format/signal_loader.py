"""Stored Signal loaders shared by the reader and writer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from timenet.values_backends.types import physical_owner


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader
    from timenet.registry.version import DatasetVersion
    from timenet.types import TimeSeriesSpec
    from timenet.values_backends.types import StoredChunk


@dataclass(frozen=True)
class StoredSignalLoader:
    """Picklable values loader for one DuckDB-indexed Signal."""

    reader: TimeFReader
    signal_key: int
    signal_id: str
    spec: TimeSeriesSpec

    def stored_chunks(self) -> tuple[DatasetVersion, list[StoredChunk]]:
        """Return direct physical locators without reading any waveform bytes."""
        rows = self.reader._chunk_rows(self.signal_key, self.signal_id)
        version = self.reader._version.resolve_payload_source(physical_owner(rows, self.signal_id))
        return version, rows

    def __deepcopy__(self, memo: dict) -> StoredSignalLoader:
        return self

    def __call__(self) -> pa.Array:
        """Load all Signal values.

        Returns:
            The complete canonical Arrow array.
        """
        return self.reader._load_signal(self.signal_key, self.signal_id, self.spec)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Load a half-open Signal step range.

        Returns:
            The requested canonical Arrow values.
        """
        return self.reader._load_signal_range(self.signal_key, self.signal_id, self.spec, start, stop)


@dataclass(frozen=True)
class MemoryArray:
    """Keep loaded values behind Signal's callable loader interface.

    The named wrapper is picklable and returns its retained array without storage access.
    The original loader preserves storage provenance, allowing writers to reuse stored offsets.
    """

    array: pa.Array
    source: Callable[[], pa.Array] | None = None

    def __call__(self) -> pa.Array:
        """Return the retained array without storage access."""
        return self.array


def resolve_storage_loader(loader: Callable[[], pa.Array] | None) -> Callable[[], pa.Array] | None:
    """Return the loader that reads from storage, behind any in-memory copies of its values.

    Eager reads keep values in :class:`MemoryArray` but remember the stored loader they came from.
    Writers follow that link to reuse stored chunks and timestamps instead of copying them.

    Returns:
        The innermost loader, or the given loader if it holds no stored source.
    """
    while isinstance(loader, MemoryArray) and loader.source is not None:
        loader = loader.source
    return loader
