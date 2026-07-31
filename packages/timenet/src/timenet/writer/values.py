"""Writer-side values backend seam: the abstract contract plus its shared types and factory.

The values plane (the float32 waveform of every series) is the one part of a TimeF version whose
on-disk representation is swappable. Everything else — samples, annotations, tasks, and the
time-series *index* that locates each chunk — is backend-agnostic. A :class:`BaseValuesBackend` takes
the deduped, sorted series and writes their values however it likes, returning a generic
:class:`ChunkPlacement` per chunk plus the list of value files to record in the manifest. The core
writer stays ignorant of shards, row groups, or arrays.

Concrete backends live in their own modules — :mod:`timenet.writer.parquet_values` (the default) — and
are constructed via :func:`make_values_backend`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.format.schemas import IdCodec, IdTypes


@dataclass(frozen=True)
class ChunkDataIndex:
    """A backend-defined address locating one chunk's values inside its ``chunk_file``.

    Backends need a different number of coordinates, so ``minor_idx`` is optional:

    - Parquet (two-level): ``major_idx`` is the row-group index within the shard, and ``minor_idx`` is
      the chunk's row within that row group (its element in the ``values`` ``list<float32>`` column).
    - Zarr (one-level): ``major_idx`` is the chunk's element-start index in the per-``spec_type`` array,
      and ``minor_idx`` is ``None`` — the chunk is the contiguous range
      ``[major_idx, major_idx + n_values)``.

    Persisted flat in the time-series index as the two integer columns ``chunk_major_idx`` and
    ``chunk_minor_idx``.
    """

    major_idx: int
    """Coarse coordinate: which block/region of ``chunk_file`` holds the chunk."""
    minor_idx: int | None
    """Fine coordinate: the chunk's position within that block, or ``None`` for a one-level backend."""


@dataclass
class ChunkPlacement:
    """Where one chunk of a series landed, plus the metadata the index needs."""

    chunk_file: str
    """Values file (relative to the staging directory) holding this chunk."""
    data_index: ChunkDataIndex
    """Backend-defined coordinates locating the chunk inside ``chunk_file``."""
    spec_type: str
    """Spec type of the source series."""
    channel: str
    """Channel name of the source series."""
    t_start_s: float
    """Start time of the chunk in seconds."""
    n_values: int
    """Number of values in the chunk."""
    sampling_rate_hz: float
    """Sampling rate of the series in Hz."""


@dataclass
class ValuesWriteResult:
    """The outcome of writing every series' values with a backend."""

    placements: dict[tuple[str, int], ChunkPlacement]
    """``(time_series_id, chunk_idx)`` -> its on-disk placement."""
    files: list[str] = field(default_factory=list)
    """Value files produced, relative to the staging directory, for ``manifest.files.time_series``."""


@dataclass(frozen=True)
class ParquetValuesConfig:
    """Typed construction options for the Parquet values backend."""

    staging_dir: Path
    """Version staging directory; shards are written beneath it."""
    id_types: IdTypes
    """Resolved logical-id storage types."""
    codec: IdCodec
    """Shared logical-id codec used by the rest of the TimeF writer."""
    shard_target_bytes: int
    """Target size for rotating shards."""
    row_group_target_bytes: int
    """Target size for flushing row groups."""
    chunk_max_bytes: int
    """Maximum uncompressed values size of one logical chunk."""
    compression: str
    """Parquet compression codec."""
    compression_level: int
    """Parquet compression level."""


ValuesBackendConfig = ParquetValuesConfig


class BaseValuesBackend(ABC):
    """Writes the values plane of a dataset and reports where each chunk landed.

    Concrete backends (e.g. :class:`~timenet.writer.parquet_values.ParquetValuesBackend`) implement
    :meth:`write_series`; the core writer stays ignorant of shards, row groups, or arrays.
    """

    name: ClassVar[str]
    """Manifest ``values_backend`` tag identifying this backend on read-back."""

    @abstractmethod
    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Write every series' values and return their placements plus the produced files.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series' float32 values (the writer's contract
                check).
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(files_finalized)`` after each value file closes.

        Returns:
            The chunk placements and the value files written.
        """


def make_values_backend(config: ValuesBackendConfig) -> BaseValuesBackend:
    """Construct the values backend described by ``config``.

    Args:
        config: Backend-specific typed construction options.

    Returns:
        The constructed backend.
    """
    from timenet.writer.parquet_values import ParquetValuesBackend

    return ParquetValuesBackend(config)
