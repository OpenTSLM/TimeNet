"""Modality and data-source descriptors.

A :class:`TimeSeriesSpec` describes one measurement *modality* (its type tag, display name, and units)
and a :class:`DataSource` the origin that produced it. Both are flat frozen dataclasses: connectors
build them directly (or subclass with field defaults for reuse), and :class:`~timenet.reader.TimeFReader`
reconstructs the identical instances from the manifest, so they round-trip and pickle without any
runtime class synthesis. The per-channel identifier lives on :class:`~timenet.dataset.TimeSeries`, not
here, so one spec is shared across every channel of a modality.
"""

from dataclasses import dataclass

import pint


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

    def __post_init__(self) -> None:
        """Validate that the sampling-rate and timestamp units have the right dimensionality.

        Raises:
            ValueError: If ``unit_sampling_rate`` is not a frequency or ``unit_timestamp`` is not a time.
        """
        hertz = pint.Unit("hertz").dimensionality
        second = pint.Unit("second").dimensionality
        if self.unit_sampling_rate.dimensionality != hertz:
            raise ValueError(
                f"unit_sampling_rate must be a frequency (dimensionality {hertz}), got {self.unit_sampling_rate!r}"
            )
        if self.unit_timestamp.dimensionality != second:
            raise ValueError(f"unit_timestamp must be a time (dimensionality {second}), got {self.unit_timestamp!r}")
