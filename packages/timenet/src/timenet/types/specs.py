"""Modality and data-source descriptors.

A :class:`TimeSeriesSpec` describes one measurement *modality* (its tag, unit, dtype, and value shape)
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


#: Spec types the Zarr backend cannot encode as its own array path segment.
_RESERVED_SPEC_TYPES = frozenset({".", "..", "_irregular", "_time_offsets"})

SUPPORTED_VALUE_DTYPES = frozenset(
    {"bool", "float32", "float64", "int8", "int16", "int32", "uint8", "uint16", "uint32"}
)


@dataclass(frozen=True)
class DataSource:
    """The origin that produced a modality: a device, an API feed, a model, an institution."""

    data_source_type: str
    """Type tag identifying the kind of source."""
    name: str
    """Human-readable display name of the source."""
    provider: str | None = None
    """Organization or platform behind the source, if any."""

    def __post_init__(self) -> None:
        """Reject a non-string or empty identifier.

        Raises:
            TimeFValidationError: If ``data_source_type`` or ``name`` is not a non-empty string.
        """
        for field_name in ("data_source_type", "name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TimeFValidationError(f"DataSource.{field_name} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class TimeSeriesSpec:
    """The contract for a measurement modality: identity, value unit, dtype, and per-timestep shape.

    It carries no unit for time or for a sampling rate. Both were fixed by construction rather than
    declared: time offsets are integer microseconds and a cadence is a Fraction of them, so neither field
    could ever hold anything but ``second`` and ``hertz``, and neither was read.
    """

    spec_type: str
    """Type tag identifying the modality; used to filter datasets by spec type."""
    name: str
    """Human-readable display name of the modality."""
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
        """Validate the spec type tag plus the per-timestep dtype and shape contract.

        Raises:
            TimeFValidationError: If the spec type, data source, dtype, shape, or dimension names are
                invalid.
        """
        if not self.spec_type:
            raise TimeFValidationError("TimeSeriesSpec.spec_type must be non-empty")
        if self.data_source is not None and not isinstance(self.data_source, DataSource):
            raise TimeFValidationError(
                f"TimeSeriesSpec.data_source must be a DataSource or None, got {type(self.data_source).__name__}"
            )
        if self.spec_type in _RESERVED_SPEC_TYPES:
            # The Zarr backend derives a per-spec_type array path from spec_type, and percent-encoding
            # leaves all of these untouched: "." and ".." are filesystem-special, and the two
            # underscore names are the groups it puts irregular values and their time offsets under.
            raise TimeFValidationError(
                f"TimeSeriesSpec.spec_type must not be one of {sorted(_RESERVED_SPEC_TYPES)}, got {self.spec_type!r}"
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

        Every ``pint.Unit`` attribute is converted rather than a fixed list: connectors are
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
