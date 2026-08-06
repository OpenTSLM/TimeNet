"""The :class:`TimeSeries` reference type: one logical stream with a lazy Arrow loader."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset.axis import OrdinalAxis, TimeAxis
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
    time_axis: TimeAxis
    """Where this series' values sit in time: a :class:`~timenet.dataset.axis.RegularAxis` for a
    cadence, or an :class:`~timenet.dataset.axis.OrdinalAxis` for a sequence with no time at all."""
    loader: Callable[[], pa.Array]
    """Lazy callable returning the series' values as an Arrow array."""
    source_id: str | None = None
    """Optional identifier of the raw source recording."""
    time_series_id: str = field(default_factory=new_id)
    """Stable identity used to dedupe and share chunks; defaults to a UUIDv7."""
    n_values: int
    """How many values the series holds, counting one per timestep. A series whose ``spec`` gives each
    timestep a shape contributes one value per timestep, not one per scalar, which is the same count
    a chunk's ``n_values`` reports.

    """

    def __post_init__(self) -> None:
        """Validate the intrinsic per-series invariants.

        Raises:
            TimeFValidationError: If ``n_values`` is not a positive integer.
            ValueError: If ``channel`` is empty. The axis validates itself.
        """
        if not self.channel:
            raise ValueError("TimeSeries.channel must be non-empty")
        if isinstance(self.n_values, bool) or not isinstance(self.n_values, int) or self.n_values <= 0:
            raise TimeFValidationError(f"TimeSeries.n_values must be a positive integer, got {self.n_values!r}")

    @property
    def span_us(self) -> tuple[int, int] | None:
        """The half-open microsecond window this series covers, or ``None`` if it has no timeline.

        Derived from the axis and the value count rather than stored, so the window and the count
        cannot disagree.

        Returns:
            ``(first time_offset, one past the last)`` in microseconds, or ``None`` for an ordinal series.
        """
        if isinstance(self.time_axis, OrdinalAxis):
            return None
        return (self.time_axis.time_offset_us(0), self.time_axis.time_offset_us(self.n_values))

    @classmethod
    def from_values(  # noqa: PLR0913
        cls,
        values: np.ndarray | Sequence[float],
        *,
        spec: TimeSeriesSpec,
        channel: str,
        time_axis: TimeAxis,
        source_id: str | None = None,
        time_series_id: str | None = None,
    ) -> "TimeSeries":
        """Build a series from already-materialized values, wrapping them in a float32 loader.

        The convenience path for connectors that hold an in-memory array: it caches ``values`` as a
        float32 Arrow array behind the loader and takes ``n_values`` from the array's own length. Use the
        ``loader=`` constructor directly for genuinely lazy sources (files, remote shards), where the
        length has to be stated because nothing has read the values yet.

        Args:
            values: The channel's values (cast to float32).
            spec: The series' measurement-modality spec.
            channel: The channel name.
            time_axis: Where the values sit in time.
            source_id: Optional id of the raw source recording.
            time_series_id: Explicit id, or ``None`` for an auto-generated UUIDv7.

        Returns:
            The constructed :class:`TimeSeries`.
        """
        array = pa.array(np.asarray(values, dtype=np.float32))
        return cls(
            spec=spec,
            channel=channel,
            time_axis=time_axis,
            loader=lambda: array,
            source_id=source_id,
            time_series_id=time_series_id or new_id(),
            n_values=len(array),
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
