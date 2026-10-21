"""Stored Signal loaders shared by the reader and writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa


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
        version = self.reader._version.resolve_payload_source(rows[0].owner_key)
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
