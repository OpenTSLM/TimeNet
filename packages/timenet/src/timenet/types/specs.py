"""Measurement modality descriptors used by Signals."""

from typing import Annotated, Any, Self, cast

import pint
from pydantic import (
    PlainSerializer,
    PlainValidator,
    StrictBool,
    StrictStr,
    WithJsonSchema,
    model_validator,
)

from timenet.types._model import TimeFModel
from timenet.types._wire import NonEmptyString, PositiveInt, ValueDtype
from timenet.types.modalities import InputModality
from timenet.types.units import normalize_unit, ureg


def _to_unit(value: Any) -> pint.Unit | None:
    """Return ``value`` as a :data:`ureg`-bound :class:`pint.Unit`, coercing strings and foreign registries.

    Returns:
        The resolved unit bound to :data:`ureg`.

    Raises:
        ValueError: If the value is neither a unit nor ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return ureg.Unit(cast("str", normalize_unit(value)))
    if isinstance(value, pint.Unit):
        return ureg.Unit(str(value)) if value._REGISTRY is not ureg else value
    raise ValueError(f"TimeSeriesSpec.unit_value must be a unit or None, got {value!r}")


UnitValue = Annotated[
    pint.Unit | None,
    PlainValidator(_to_unit, json_schema_input_type=str | None),
    PlainSerializer(lambda value: None if value is None else str(value), return_type=str | None),
    WithJsonSchema({"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}),
]
"""A Pint unit represented by its canonical string on the wire."""


#: Spec types the Zarr backend cannot encode as its own array path segment.
_RESERVED_SPEC_TYPES = frozenset({".", "..", "_irregular", "_time_offsets", "_validity"})


class TimeSeriesSpec(TimeFModel):
    """Define the contract for a measurement modality: identity, value unit, dtype, and per-timestep shape.

    The spec carries no unit for time or for a sampling rate. Time offsets are integer microseconds and
    a cadence is a Fraction of them, so those units can only be ``second`` and ``hertz``.
    """

    spec_type: NonEmptyString
    """Type tag identifying the modality. Callers use it to filter datasets by spec type."""
    name: StrictStr
    """Human-readable display name of the modality."""
    unit_value: UnitValue
    """Unit of the measured values. ``None`` means unknown, not dimensionless."""
    dtype: ValueDtype = "float32"
    """Value dtype: a NumPy scalar dtype, ``"str"`` for text, or ``"enum"`` for a categorical value."""
    categories: tuple[NonEmptyString, ...] = ()
    """Ordered codebook for ``dtype="enum"``. Empty for every other dtype."""
    value_shape: tuple[PositiveInt, ...] = ()
    """Shape of one timestep, excluding the leading time axis. An empty shape means scalar values."""
    dimension_names: tuple[NonEmptyString, ...] = ()
    """Optional names for the dimensions in :attr:`value_shape`."""
    nullable: StrictBool = False
    """Whether a whole timestep can be missing. An Arrow bit marks whether each timestep is present."""
    modality: InputModality = InputModality.TIME_SERIES
    """Semantic input kind. Set IMAGE, AUDIO, or TEXT for Signals that carry those kinds of data."""

    @model_validator(mode="after")
    def _valid_contract(self) -> Self:
        if self.modality is InputModality.NO_INPUT:
            raise ValueError("a Signal cannot have the no_input modality")
        if self.spec_type in _RESERVED_SPEC_TYPES:
            # The Zarr backend builds a per-spec_type array path from spec_type. Percent-encoding
            # leaves all of these names untouched. "." and ".." are filesystem-special. The
            # underscore names remain reserved for compatibility with older Zarr layouts.
            raise ValueError(
                f"TimeSeriesSpec.spec_type must not be one of {sorted(_RESERVED_SPEC_TYPES)}, got {self.spec_type!r}"
            )
        if self.dtype == "enum":
            if not self.categories:
                raise ValueError("enum categories must be non-empty")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError("enum categories must be unique")
        elif self.categories:
            raise ValueError("categories are only valid for dtype 'enum'")
        if self.dimension_names and len(self.dimension_names) != len(self.value_shape):
            raise ValueError("TimeSeriesSpec.dimension_names must be empty or match value_shape length")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("TimeSeriesSpec.dimension_names must be unique")
        return self

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
        state = super().__getstate__()
        values = state["__dict__"]
        unit_fields = sorted(name for name, value in values.items() if isinstance(value, pint.Unit))
        for name in unit_fields:
            values[name] = str(values[name])
        values[_UNIT_FIELDS_KEY] = unit_fields
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Rebuild the recorded unit fields against the shared registry.

        Args:
            state: The pickled state produced by :meth:`__getstate__`.
        """
        values = cast("dict[str, object]", state["__dict__"])
        for name in cast("list[str]", values.pop(_UNIT_FIELDS_KEY, [])):
            values[name] = ureg.Unit(str(values[name]))
        super().__setstate__(state)


#: Key under which :meth:`TimeSeriesSpec.__getstate__` records which attributes held units.
_UNIT_FIELDS_KEY = "__timenet_unit_fields__"
