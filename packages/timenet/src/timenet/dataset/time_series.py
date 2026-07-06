"""The :class:`TimeSeries` reference type: one channel of values with a lazy Arrow loader."""

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
import pyarrow as pa

from timenet.types import TimeSeriesSpec, new_id


@dataclass(frozen=True, eq=False, kw_only=True)
class TimeSeries:
    """Reference to one channel of time-series data, with optional windowing and a lazy loader.

    Identity-based equality (``eq=False``): the writer dedupes by ``time_series_id``, not by value, so
    reusing one instance across samples (or giving two instances the same explicit id) shares one chunk
    on disk. Consumers read values through :meth:`to_arrow` / :meth:`to_numpy`; ``loader`` is plumbing
    supplied by the connector (raw source) at curation or by :class:`~timenet.reader.TimeFReader`
    (shard parquet) on read-back.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, name, and axis units."""
    channel: str
    """Name of this channel within the modality; must be non-empty."""
    sampling_rate_hz: float
    """Sampling rate in hertz; must be positive and finite."""
    loader: Callable[[], pa.Array]
    """Lazy callable returning the series' values as an Arrow array."""
    source_id: str | None = None
    """Optional identifier of the raw source recording."""
    time_series_id: str = field(default_factory=new_id)
    """Stable identity used to dedupe and share chunks; defaults to a UUIDv7."""
    t_start_s: float = 0.0
    """Start of the series window in seconds; must be non-negative."""
    t_end_s: float | None = None
    """Optional end of the series window in seconds; must exceed t_start_s."""

    def __post_init__(self) -> None:
        """Validate the intrinsic per-series invariants.

        Raises:
            ValueError: If ``channel`` is empty, ``sampling_rate_hz`` is not positive and finite,
                ``t_start_s`` is negative, or ``t_end_s`` is not greater than ``t_start_s``.
        """
        if not self.channel:
            raise ValueError("TimeSeries.channel must be non-empty")
        if not math.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            raise ValueError(f"TimeSeries.sampling_rate_hz must be positive and finite, got {self.sampling_rate_hz!r}")
        if self.t_start_s < 0:
            raise ValueError(f"TimeSeries.t_start_s must be >= 0, got {self.t_start_s}")
        if self.t_end_s is not None and self.t_end_s <= self.t_start_s:
            raise ValueError(f"TimeSeries.t_end_s ({self.t_end_s}) must be > t_start_s ({self.t_start_s})")

    def to_arrow(self) -> pa.Array:
        """Read the series' values as an Arrow array.

        Returns:
            The series' 1-D Arrow array of values, produced by ``loader``.
        """
        return self.loader()

    def to_numpy(self) -> np.ndarray:
        """Read the series' values as a NumPy array.

        Returns:
            The series' values as a 1-D ``np.ndarray``.
        """
        return self.to_arrow().to_numpy(zero_copy_only=False)
