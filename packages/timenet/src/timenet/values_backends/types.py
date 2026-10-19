"""Storage backend identifiers and chunk locations shared by readers and writers."""

from dataclasses import dataclass
from enum import StrEnum


class ValuesBackend(StrEnum):
    """Storage backend for the time-series values plane, recorded as the manifest ``values_backend`` tag.

    Each member value is the on-disk tag. This class is a :class:`~enum.StrEnum`, so each member is
    also a plain ``str``. Each member compares and serializes like its bare tag.
    """

    PARQUET = "parquet"
    ZARR = "zarr"


SUPPORTED_VALUES_BACKENDS: frozenset[ValuesBackend] = frozenset(ValuesBackend)
"""Every backend a writer may target and a reader can resolve."""


@dataclass(frozen=True)
class ChunkDataIndex:
    """A backend-defined address locating one chunk's values inside its ``chunk_file``.

    Backends need a different number of coordinates, so ``minor_idx`` is optional:

    - Parquet (two-level): ``major_idx`` is the row-group index within the shard. ``minor_idx`` is the
      chunk's row within that row group. That row is its element in the ``values`` ``list<float32>``
      column.
    - Zarr (one-level): ``major_idx`` is the chunk's element-start index in the per-``spec_type`` array.
      ``minor_idx`` is ``None``. The chunk is the contiguous range ``[major_idx, major_idx + n_values)``.

    The catalog stores these as ``chunk_major_index`` and ``chunk_minor_index``.
    """

    major_idx: int
    """Coarse coordinate: which block/region of ``chunk_file`` holds the chunk."""
    minor_idx: int | None
    """Fine coordinate: the chunk's position within that block, or ``None`` for a one-level backend."""


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """Locate one stored signal chunk without depending on a catalog row dictionary."""

    chunk_idx: int
    """Position of this chunk within the signal."""
    chunk_file: str
    """Path of the values file, relative to the dataset version."""
    data_index: ChunkDataIndex
    """Backend coordinates within the file."""
    n_values: int
    """Number of values in the chunk."""
