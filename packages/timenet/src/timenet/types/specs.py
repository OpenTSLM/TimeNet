"""Modality and data-source descriptors.

A :class:`TimeSeriesSpec` describes one measurement *modality* (its type tag, display name, and units)
and a :class:`DataSource` the origin that produced it. Both are flat frozen dataclasses: connectors
build them directly (or subclass with field defaults for reuse), and :class:`~timenet.reader.TimeFReader`
reconstructs the identical instances from the manifest, so they round-trip and pickle without any
runtime class synthesis. The per-channel identifier lives on :class:`~timenet.dataset.TimeSeries`, not
here, so one spec is shared across every channel of a modality.
"""

from dataclasses import dataclass

import numpy as np
import pint


SUPPORTED_VALUE_DTYPES = frozenset(
    {"bool", "float32", "float64", "int8", "int16", "int32", "uint8", "uint16", "uint32"}
)


@dataclass(frozen=True)
class DataSource:
    """The origin that produced a modality: a device, an API feed, a model, an institution."""

    data_source_type: str
    """Type tag identifying the kind of source; keys the spec-to-source link in the manifest."""
    name: str
    """Human-readable display name of the source."""
    provider: str | None = None
    """Organization or platform behind the source, if any."""


@dataclass(frozen=True)
class TimeSeriesSpec:
    """The contract for a measurement modality: type tag, name, and the units of its axes."""

    spec_type: str
    """Type tag identifying the modality; used to filter datasets by spec type."""
    name: str
    """Human-readable display name of the modality."""
    unit_sampling_rate: pint.Unit
    """Unit of the sampling rate; must have frequency dimensionality."""
    unit_timestamp: pint.Unit
    """Unit of the timestamp axis; must have time dimensionality."""
    unit_value: pint.Unit
    """Unit of the measured values."""
    data_source: DataSource | None = None
    """Origin that produced this modality, if known."""
    dtype: str = "float32"
    """NumPy scalar dtype used for each value element."""
    value_shape: tuple[int, ...] = ()
    """Shape of one timestep, excluding the leading time axis; empty means scalar values."""
    dimension_names: tuple[str, ...] = ()
    """Optional names for the dimensions in :attr:`value_shape`."""

    def __post_init__(self) -> None:
        """Validate that the sampling-rate and timestamp units have the right dimensionality.

        Raises:
            ValueError: If ``unit_sampling_rate`` is not a frequency or ``unit_timestamp`` is not a time.
        """
        hertz = pint.Unit("hertz").dimensionality
        second = pint.Unit("second").dimensionality
        if not self.spec_type:
            raise ValueError("TimeSeriesSpec.spec_type must be non-empty")
        if self.unit_sampling_rate.dimensionality != hertz:
            raise ValueError(
                f"unit_sampling_rate must be a frequency (dimensionality {hertz}), got {self.unit_sampling_rate!r}"
            )
        if self.unit_timestamp.dimensionality != second:
            raise ValueError(f"unit_timestamp must be a time (dimensionality {second}), got {self.unit_timestamp!r}")
        try:
            normalized_dtype = np.dtype(self.dtype).name
        except TypeError as exc:
            raise ValueError(f"unsupported TimeSeriesSpec.dtype {self.dtype!r}") from exc
        if normalized_dtype not in SUPPORTED_VALUE_DTYPES or normalized_dtype != self.dtype:
            raise ValueError(
                f"TimeSeriesSpec.dtype must be one of {sorted(SUPPORTED_VALUE_DTYPES)}, got {self.dtype!r}"
            )
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.value_shape):
            raise ValueError(
                f"TimeSeriesSpec.value_shape dimensions must be positive integers, got {self.value_shape!r}"
            )
        if self.dimension_names and len(self.dimension_names) != len(self.value_shape):
            raise ValueError("TimeSeriesSpec.dimension_names must be empty or match value_shape length")
        if any(not name for name in self.dimension_names):
            raise ValueError("TimeSeriesSpec.dimension_names must not contain empty names")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("TimeSeriesSpec.dimension_names must be unique")
