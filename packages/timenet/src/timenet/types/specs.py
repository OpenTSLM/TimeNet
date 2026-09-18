"""Measurement modality descriptors used by Signals."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pint

from timenet.errors import TimeFValidationError
from timenet.types.units import normalize_unit, ureg


def _to_unit(value: str | pint.Unit) -> pint.Unit:
    """Return ``value`` as a :data:`ureg`-bound :class:`pint.Unit`, coercing strings and foreign registries.

    Returns:
        The resolved unit bound to :data:`ureg`.
    """
    if isinstance(value, str):
        return ureg.Unit(cast("str", normalize_unit(value)))
    if isinstance(value, pint.Unit) and value._REGISTRY is not ureg:
        return ureg.Unit(str(value))
    return value


#: Spec types the Zarr backend cannot encode as its own array path segment.
_RESERVED_SPEC_TYPES = frozenset({".", "..", "_irregular", "_time_offsets", "_validity"})

SUPPORTED_VALUE_DTYPES = frozenset(
    {"bool", "float32", "float64", "int8", "int16", "int32", "uint8", "uint16", "uint32", "str"}
)


def _validate_dtype(dtype: str, categories: tuple[str, ...]) -> None:
    """Validate a spec dtype tag and its categories codebook.

    ``"str"`` and ``"enum"`` are string-kind values, not NumPy dtypes. An ``"enum"`` dtype carries
    its ordered category labels in ``categories``: non-empty, unique, non-empty strings. Every other
    dtype must not declare categories.

    Args:
        dtype: The spec's dtype tag.
        categories: The ordered codebook, empty for every dtype except ``"enum"``.

    Raises:
        TimeFValidationError: If ``dtype`` is unsupported, or ``categories`` is unusable for an
            enum or set on a non-enum dtype.
    """
    if dtype == "enum":
        if not categories:
            raise TimeFValidationError(
                f"TimeSeriesSpec.categories must be non-empty for dtype {dtype!r}, got {categories!r}"
            )
        if any(not isinstance(name, str) or not name for name in categories):
            raise TimeFValidationError(f"TimeSeriesSpec.categories must be non-empty strings, got {categories!r}")
        if len(set(categories)) != len(categories):
            raise TimeFValidationError(f"TimeSeriesSpec.categories must be unique, got {categories!r}")
        return
    if categories:
        raise TimeFValidationError(f"TimeSeriesSpec.categories is only for dtype 'enum', got dtype={dtype!r}")
    if dtype == "str":
        return
    try:
        normalized_dtype = np.dtype(dtype).name
    except TypeError as exc:
        raise TimeFValidationError(f"unsupported TimeSeriesSpec.dtype {dtype!r}") from exc
    if normalized_dtype not in SUPPORTED_VALUE_DTYPES or normalized_dtype != dtype:
        raise TimeFValidationError(
            f"TimeSeriesSpec.dtype must be one of {sorted(SUPPORTED_VALUE_DTYPES)}, got {dtype!r}"
        )


@dataclass(frozen=True)
class TimeSeriesSpec:
    """Define the contract for a measurement modality: identity, value unit, dtype, and per-timestep shape.

    The spec carries no unit for time or for a sampling rate. Time offsets are integer microseconds and
    a cadence is a Fraction of them, so those units can only be ``second`` and ``hertz``.
    """

    spec_type: str
    """Type tag identifying the modality. Callers use it to filter datasets by spec type."""
    name: str
    """Human-readable display name of the modality."""
    unit_value: pint.Unit
    """Unit of the measured values."""
    dtype: str = "float32"
    """Value dtype: a NumPy scalar dtype, ``"str"`` for text, or ``"enum"`` for a categorical value."""
    categories: tuple[str, ...] = ()
    """Ordered codebook for ``dtype="enum"``. Empty for every other dtype."""
    value_shape: tuple[int, ...] = ()
    """Shape of one timestep, excluding the leading time axis. An empty shape means scalar values."""
    dimension_names: tuple[str, ...] = ()
    """Optional names for the dimensions in :attr:`value_shape`."""
    nullable: bool = False
    """Whether a whole timestep can be missing. An Arrow bit marks whether each timestep is present."""

    def __post_init__(self) -> None:
        """Coerce the unit and validate the spec contract.

        Raises:
            TimeFValidationError: If the unit, spec type, dtype, shape, or dimension names are invalid.
        """
        object.__setattr__(self, "unit_value", _to_unit(self.unit_value))
        if not isinstance(self.nullable, bool):
            raise TimeFValidationError("TimeSeriesSpec.nullable must be a bool")
        if not self.spec_type:
            raise TimeFValidationError("TimeSeriesSpec.spec_type must be non-empty")
        if self.spec_type in _RESERVED_SPEC_TYPES:
            # The Zarr backend builds a per-spec_type array path from spec_type. Percent-encoding
            # leaves all of these names untouched. "." and ".." are filesystem-special. The two
            # underscore names are the groups that hold irregular values and their time offsets.
            raise TimeFValidationError(
                f"TimeSeriesSpec.spec_type must not be one of {sorted(_RESERVED_SPEC_TYPES)}, got {self.spec_type!r}"
            )
        _validate_dtype(self.dtype, self.categories)
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

        A :class:`pint.Unit` unpickles against whatever registry is process-global at the time. A spec
        pickled as-is can come back bound to a foreign registry. It can also fail on ``bpm`` and the
        other units that only :data:`~timenet.types.units.ureg` defines. Storing names keeps a pickled
        spec self-describing, which multiprocessing DataLoaders need.

        This method converts every ``pint.Unit`` attribute, not a fixed list. Connectors can subclass
        this class with field defaults. A subclass that adds its own unit field otherwise pickles it
        registry-bound and fails on a custom unit. The state records the converted names so
        :meth:`__setstate__` knows which strings to rebuild.

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
            object.__setattr__(self, key, value)


#: Key under which :meth:`TimeSeriesSpec.__getstate__` records which attributes held units.
_UNIT_FIELDS_KEY = "__timenet_unit_fields__"
