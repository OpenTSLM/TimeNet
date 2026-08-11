"""Writer-side values backend seam: the abstract contract plus its shared types and factory.

The values plane is the float32 waveform of every series. It is the one part of a TimeF version whose
on-disk representation is swappable. Everything else is backend-agnostic: samples, annotations, tasks,
and the time-series index that locates each chunk. A :class:`BaseValuesBackend` takes the deduped,
sorted series and writes their values. It returns one :class:`ChunkPlacement` per chunk plus the list
of value files to record in the manifest. The core writer does not know about shards, row groups, or
arrays.

Concrete backends live in their own modules: :mod:`timenet.values_backends.parquet.writer` (the
default) and :mod:`timenet.values_backends.zarr.writer`. The :func:`make_values_backend` factory
constructs them.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, assert_never

import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.values_backends.parquet.config import ParquetValuesConfig
from timenet.values_backends.zarr.config import ZarrValuesConfig


@dataclass(frozen=True)
class ChunkDataIndex:
    """A backend-defined address locating one chunk's values inside its ``chunk_file``.

    Backends need a different number of coordinates, so ``minor_idx`` is optional:

    - Parquet (two-level): ``major_idx`` is the row-group index within the shard. ``minor_idx`` is the
      chunk's row within that row group. That row is its element in the ``values`` ``list<float32>``
      column.
    - Zarr (one-level): ``major_idx`` is the chunk's element-start index in the per-``spec_type`` array.
      ``minor_idx`` is ``None``. The chunk is the contiguous range ``[major_idx, major_idx + n_values)``.

    The index persists these flat as the two integer columns ``chunk_major_idx`` and ``chunk_minor_idx``.
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
    n_values: int
    """Number of values in the chunk."""


@dataclass
class ValuesWriteResult:
    """The outcome of writing every series' values with a backend."""

    placements: dict[tuple[str, int], ChunkPlacement]
    """``(time_series_id, chunk_idx)`` -> its on-disk placement."""
    files: list[str] = field(default_factory=list)
    """Value files produced, relative to the staging directory, for ``manifest.files.time_series``."""
    value_encoding: dict[str, str] = field(default_factory=dict)
    """``spec_type`` -> the values encoding the backend applied, for ``manifest.value_encoding``.

    Empty for a backend whose layout has no such choice (Zarr fixes its codec per array instead).
    """


ValuesBackendConfig = ParquetValuesConfig | ZarrValuesConfig


class BaseValuesBackend(ABC):
    """Write the values plane of a dataset and report where each chunk landed.

    Concrete backends implement :meth:`write_series`. One example is
    :class:`~timenet.values_backends.parquet.writer.ParquetValuesBackend`. The core writer does not know
    about shards, row groups, or arrays.
    """

    name: ClassVar[str]
    """Manifest ``values_backend`` tag identifying this backend on read-back."""

    @abstractmethod
    def write_series(
        self,
        unique_series: list[TimeSeries],
        *,
        read_and_validate: Callable[[TimeSeries], pa.Array],
        read_time_offsets: Callable[[TimeSeries], pa.Array | None],
        on_series_done: Callable[[int, int], None],
        on_file_done: Callable[[int], None],
    ) -> ValuesWriteResult:
        """Write every series' values and return their placements plus the produced files.

        Args:
            unique_series: The deduped, sorted series to serialize.
            read_and_validate: Loads and validates one series against its spec's values contract.
            read_time_offsets: Loads and validates the per-value time offsets of an irregular series.
                Returns ``None`` for a series that stores none. It stays separate from
                ``read_and_validate`` so the values seam holds one array per series for any axis shape.
            on_series_done: Progress callback invoked ``(completed, total)`` after each series.
            on_file_done: Progress callback invoked ``(files_finalized)`` after each value file closes.

        Returns:
            The chunk placements and the value files written.
        """


def make_values_backend(config: ValuesBackendConfig) -> BaseValuesBackend:
    """Construct the values backend described by ``config``.

    Each backend receives the subset of the writer's value options it understands. Parquet takes all of
    them. Zarr takes the chunk and shard byte targets and the codec. The Zarr store has no row groups or
    id columns.

    Args:
        config: Backend-specific typed construction options.

    Returns:
        The constructed backend.
    """
    if isinstance(config, ParquetValuesConfig):
        from timenet.values_backends.parquet.writer import ParquetValuesBackend  # noqa: PLC0415

        return ParquetValuesBackend(config)
    if isinstance(config, ZarrValuesConfig):
        from timenet.values_backends.zarr.writer import ZarrValuesBackend  # noqa: PLC0415

        return ZarrValuesBackend(config)
    assert_never(config)
