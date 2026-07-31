"""The :class:`TimeSeries` reference type: one logical stream with a lazy Arrow loader."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec, new_id


@dataclass(frozen=True, eq=False, kw_only=True)
class TimeSeries:
    """Reference to one logical stream of time-series data, with optional windowing and a lazy loader.

    Identity-based equality (``eq=False``): the writer dedupes by ``time_series_id``, not by value, so
    reusing one instance across samples (or giving two instances the same explicit id) shares one chunk
    on disk. Consumers read values through :meth:`to_arrow` / :meth:`to_numpy`; ``loader`` is plumbing
    supplied by the connector (raw source) at curation or by :class:`~timenet.reader.TimeFReader`
    (the manifest-selected values backend) on read-back.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, units, dtype, and per-timestep shape."""
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

    @classmethod
    def from_values(
        cls,
        values: np.ndarray | Sequence[float],
        *,
        spec: TimeSeriesSpec,
        channel: str,
        sampling_rate_hz: float,
        source_id: str | None = None,
        time_series_id: str | None = None,
        t_start_s: float = 0.0,
        t_end_s: float | None = None,
    ) -> "TimeSeries":
        """Build a series from already-materialized values, wrapping them in a float32 loader.

        The convenience path for connectors that hold an in-memory array: it caches ``values`` as a
        float32 Arrow array behind the loader and derives ``t_end_s`` from the length when omitted. Use
        the ``loader=`` constructor directly for genuinely lazy sources (files, remote shards).

        Args:
            values: The channel's values (cast to float32).
            spec: The series' measurement-modality spec.
            channel: The channel name.
            sampling_rate_hz: Sampling rate in hertz.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.
            t_start_s: Start of the window in seconds.
            t_end_s: End of the window in seconds, or ``None`` to derive it from the length.

        Returns:
            The constructed :class:`TimeSeries`.
        """
        array = pa.array(np.asarray(values, dtype=np.float32))
        return cls(
            spec=spec,
            channel=channel,
            sampling_rate_hz=sampling_rate_hz,
            loader=lambda: array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            t_start_s=t_start_s,
            t_end_s=t_end_s if t_end_s is not None else t_start_s + len(array) / sampling_rate_hz,
        )

    def to_arrow(self) -> pa.Array:
        """Read the series' values as an Arrow array.

        Returns:
            A primitive array for scalar values or a fixed-shape tensor array for N-D values.
        """
        return self.loader()

    def to_numpy(self) -> Shaped[np.ndarray, " time *value"]:
        """Read the series' values as a NumPy array.

        Returns:
            The series' values with shape ``(n_steps, *spec.value_shape)``.
        """
        values = self.to_arrow()
        if isinstance(values, pa.FixedShapeTensorArray):
            return values.to_numpy_ndarray()
        return values.to_numpy(zero_copy_only=False)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open temporal step range as Arrow without forcing a NumPy conversion.

        Range-aware storage loaders read only the intersecting chunks. Connector loaders that only
        implement the original no-argument callable remain compatible through a full-read slice.

        Args:
            start: First temporal step, inclusive.
            stop: Last temporal step, exclusive.

        Returns:
            A primitive Arrow array for scalar series or a fixed-shape tensor array for N-D series.

        Raises:
            TimeFValidationError: If the range is negative or reversed.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError(f"expected 0 <= start <= stop, got start={start}, stop={stop}")
        read_steps = getattr(self.loader, "read_steps", None)
        if read_steps is not None:
            return read_steps(start, stop)
        return self.to_arrow().slice(start, stop - start)
