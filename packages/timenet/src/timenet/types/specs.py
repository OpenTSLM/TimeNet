"""Modality and data-source descriptors.

A :class:`TimeSeriesSpec` describes one measurement *modality* (its tag, units, dtype, and value shape)
and a :class:`DataSource` the origin that produced it. Both are flat frozen dataclasses: connectors
build them directly (or subclass with field defaults for reuse), and :class:`~timenet.reader.TimeFReader`
reconstructs the identical instances from the manifest, so they round-trip and pickle without any
runtime class synthesis. The per-channel identifier lives on :class:`~timenet.dataset.TimeSeries`, not
here, so one spec is shared across every channel of a modality.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pint

from timenet.errors import TimeFValidationError
from timenet.types.units import ureg


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
    """The contract for a measurement modality: identity, units, dtype, and per-timestep shape."""

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
        """Validate units plus the per-timestep dtype and shape contract.

        Raises:
            TimeFValidationError: If the units, dtype, shape, or dimension names are invalid.
        """
        hertz = ureg.hertz.dimensionality
        second = ureg.second.dimensionality
        if not self.spec_type:
            raise TimeFValidationError("TimeSeriesSpec.spec_type must be non-empty")
        if self.unit_sampling_rate.dimensionality != hertz:
            raise TimeFValidationError(
                f"unit_sampling_rate must be a frequency (dimensionality {hertz}), got {self.unit_sampling_rate!r}"
            )
        if self.unit_timestamp.dimensionality != second:
            raise TimeFValidationError(
                f"unit_timestamp must be a time (dimensionality {second}), got {self.unit_timestamp!r}"
            )
        try:
            normalized_dtype = np.dtype(self.dtype).name
        except TypeError as exc:
            raise TimeFValidationError(f"unsupported TimeSeriesSpec.dtype {self.dtype!r}") from exc
        if normalized_dtype not in SUPPORTED_VALUE_DTYPES or normalized_dtype != self.dtype:
            raise TimeFValidationError(
                f"TimeSeriesSpec.dtype must be one of {sorted(SUPPORTED_VALUE_DTYPES)}, got {self.dtype!r}"
            )
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.value_shape):
            raise TimeFValidationError(
                f"TimeSeriesSpec.value_shape dimensions must be positive integers, got {self.value_shape!r}"
            )
        if self.dimension_names and len(self.dimension_names) != len(self.value_shape):
            raise TimeFValidationError("TimeSeriesSpec.dimension_names must be empty or match value_shape length")
        if any(not name for name in self.dimension_names):
            raise TimeFValidationError("TimeSeriesSpec.dimension_names must not contain empty names")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise TimeFValidationError("TimeSeriesSpec.dimension_names must be unique")

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
