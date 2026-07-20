"""Modality and data-source descriptors.

A :class:`TimeSeriesSpec` describes one measurement *modality* (its type tag, display name, and units)
and a :class:`DataSource` the origin that produced it. Both are flat frozen dataclasses: connectors
build them directly (or subclass with field defaults for reuse), and :class:`~timenet.reader.TimeFReader`
reconstructs the identical instances from the manifest, so they round-trip and pickle without any
runtime class synthesis. The per-channel identifier lives on :class:`~timenet.dataset.TimeSeries`, not
here, so one spec is shared across every channel of a modality.
"""

from dataclasses import dataclass
from typing import cast

import pint

from timenet.errors import TimeFValidationError
from timenet.types.units import ureg


@dataclass(frozen=True)
class DataSource:
    """The origin that produced a modality: a device, an API feed, a model, an institution."""

    data_source_type: str
    name: str
    provider: str | None = None


@dataclass(frozen=True)
class TimeSeriesSpec:
    """The contract for a measurement modality: type tag, name, and the units of its axes."""

    spec_type: str
    name: str
    unit_sampling_rate: pint.Unit
    unit_timestamp: pint.Unit
    unit_value: pint.Unit
    data_source: DataSource | None = None

    def __post_init__(self) -> None:
        """Validate that the sampling-rate and timestamp units have the right dimensionality.

        Raises:
            TimeFValidationError: If ``unit_sampling_rate`` is not a frequency or ``unit_timestamp`` is
                not a time.
        """
        hertz = ureg.hertz.dimensionality
        second = ureg.second.dimensionality
        if self.unit_sampling_rate.dimensionality != hertz:
            raise TimeFValidationError(
                f"unit_sampling_rate must be a frequency (dimensionality {hertz}), got {self.unit_sampling_rate!r}"
            )
        if self.unit_timestamp.dimensionality != second:
            raise TimeFValidationError(
                f"unit_timestamp must be a time (dimensionality {second}), got {self.unit_timestamp!r}"
            )

    def __getstate__(self) -> dict[str, object]:
        """Pickle every unit by name rather than as a registry-bound object.

        A :class:`pint.Unit` unpickles against whatever registry is process-global at the time, so a
        spec pickled as-is would come back bound to a foreign registry, or fail outright on ``bpm`` and
        the other units only :data:`~timenet.types.units.ureg` defines. Storing names keeps a pickled
        spec self-describing, which is what multiprocessing DataLoaders need.

        Every ``pint.Unit`` attribute is converted, not a fixed list of three: connectors are
        encouraged to subclass this with field defaults, and a subclass that adds its own unit field
        would otherwise pickle it registry-bound and fail on a custom unit. The converted names are
        recorded in the state so :meth:`__setstate__` knows which strings to rebuild.

        Returns:
            The instance state with every unit field replaced by its name.
        """
        state: dict[str, object] = dict(self.__dict__)
        unit_fields = sorted(name for name, value in state.items() if isinstance(value, pint.Unit))
        for name in unit_fields:
            state[name] = str(state[name])
        state[_UNIT_FIELDS_KEY] = unit_fields
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Rebuild the recorded unit fields against the shared registry.

        Args:
            state: The pickled state produced by :meth:`__getstate__`.
        """
        restored = dict(state)
        for name in cast("list[str]", restored.pop(_UNIT_FIELDS_KEY, [])):
            restored[name] = ureg.Unit(str(restored[name]))
        for key, value in restored.items():
            object.__setattr__(self, key, value)  # frozen dataclass


#: Key under which :meth:`TimeSeriesSpec.__getstate__` records which attributes held units.
_UNIT_FIELDS_KEY = "__timenet_unit_fields__"
