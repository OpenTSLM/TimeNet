"""Measurement modality descriptors used by Signals."""

from collections.abc import Callable, Mapping
from enum import Enum, unique
from typing import Annotated, Any, Self, cast

import pint
from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    StrictFloat,
    StrictStr,
    WithJsonSchema,
    model_validator,
)

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.modalities import InputModality
from timenet.types.units import normalize_unit, ureg
from timenet.types.wire import NonEmptyString, StrictPositiveInt, ValueDtype


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


def unit_metadata(
    *, validator: Callable[[Any], pint.Unit | None], nullable: bool
) -> tuple[PlainValidator, PlainSerializer, WithJsonSchema]:
    """Build Pydantic metadata for a unit encoded by its canonical string.

    Args:
        validator: Convert input to a unit bound to the shared registry.
        nullable: Whether the wire representation also accepts null.

    Returns:
        The validator, serializer, and JSON Schema metadata for the unit field.
    """
    wire_type = str | None if nullable else str
    serializer = (lambda value: None if value is None else str(value)) if nullable else str
    schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if nullable:
        schema = {"anyOf": [schema, {"type": "null"}]}
    return (
        PlainValidator(validator, json_schema_input_type=wire_type),
        PlainSerializer(serializer, return_type=wire_type),
        WithJsonSchema(schema),
    )


UnitValue = Annotated[pint.Unit | None, *unit_metadata(validator=_to_unit, nullable=True)]
"""A Pint unit represented by its canonical string on the wire."""


def _to_required_unit(value: Any) -> pint.Unit:
    """Return ``value`` as a :data:`ureg`-bound unit, rejecting ``None``.

    Returns:
        The resolved unit bound to :data:`ureg`.

    Raises:
        ValueError: If the value is ``None`` or not a unit.
    """
    unit = _to_unit(value)
    if unit is None:
        raise ValueError("a unit is required")
    return unit


Unit = Annotated[pint.Unit, *unit_metadata(validator=_to_required_unit, nullable=False)]
"""A required Pint unit represented by its canonical string on the wire."""


#: Spec types the Zarr backend cannot encode as its own array path segment.
_RESERVED_SPEC_TYPES = frozenset({".", "..", "_irregular", "_time_offsets", "_validity"})


@unique
class MissingReason(Enum):
    """Reasons why whole timesteps of a modality can be missing."""

    SENSOR_GAP = "sensor_gap"
    """The device recorded nothing for a while, for example because it lost contact with the body."""
    NOT_RECORDED = "not_recorded"
    """The source reported no value for the timestep, although the recording ran."""
    ALIGNMENT_PADDING = "alignment_padding"
    """The source aligned several series to one time grid, and this series covers only part of it."""
    NOT_APPLICABLE = "not_applicable"
    """The quantity has no meaning at the timestep, for example a glucose reading before the sensor warmed up."""
    REDACTED = "redacted"
    """The source removed the value on purpose, for example to protect a person's privacy."""
    UNSPECIFIED = "unspecified"
    """The release does not record a reason for missing timesteps.

    New series with missing timesteps must use another reason.
    """


class MissingValues(TimeFModel):
    """The reason for missing timesteps and the maximum fraction allowed in a series.

    A spec without this record allows no missing timesteps. An Arrow null marks a missing timestep.
    """

    reason: MissingReason
    """Why timesteps are missing."""
    description: NonEmptyString | None = None
    """An explanation of missing timesteps in the source, or ``None``."""
    max_fraction: Annotated[StrictFloat, Field(gt=0, le=1)]
    """The maximum fraction of timesteps that a series can have missing.

    The writer rejects a series with a larger fraction of missing timesteps. ``1.0`` allows a
    series that misses almost every timestep.
    """

    @classmethod
    def unspecified(cls) -> Self:
        """Create a missing-value record for a legacy release.

        Returns:
            A record with reason :attr:`MissingReason.UNSPECIFIED` that allows any fraction of missing timesteps.
        """
        return cls(reason=MissingReason.UNSPECIFIED, max_fraction=1.0)


def _add_legacy_nullable_to_schema(schema: dict[str, Any]) -> None:
    """Keep the legacy ``nullable`` flag in the JSON schema of a spec."""
    schema["properties"]["nullable"] = {
        "type": "boolean",
        "deprecated": True,
        "description": "Older flag. A reader turns true into missing with the reason 'unspecified'.",
    }


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
    value_shape: tuple[StrictPositiveInt, ...] = ()
    """Shape of one timestep, excluding the leading time axis. An empty shape means scalar values."""
    dimension_names: tuple[NonEmptyString, ...] = ()
    """Optional names for the dimensions in :attr:`value_shape`."""
    missing: MissingValues | None = None
    """The reason and maximum fraction for missing timesteps, or ``None`` if none are allowed."""
    modality: InputModality = InputModality.TIME_SERIES
    """Semantic input kind. Set IMAGE, AUDIO, or TEXT for Signals that carry those kinds of data."""

    model_config = ConfigDict(json_schema_extra=_add_legacy_nullable_to_schema)

    @property
    def nullable(self) -> bool:
        """Whether a whole timestep can be missing. This is true when :attr:`missing` is set."""
        return self.missing is not None

    @staticmethod
    def upgrade_legacy_nullable(data: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a legacy ``nullable`` flag to a ``missing`` record.

        If ``missing`` is present, ``nullable`` must agree with it.

        Args:
            data: The fields of one spec from a manifest.

        Returns:
            The fields without ``nullable``.

        Raises:
            ValueError: If ``nullable`` is not a bool or disagrees with ``missing``.
        """
        fields = dict(data)
        if "nullable" not in fields:
            return fields
        nullable = fields.pop("nullable")
        if not isinstance(nullable, bool):
            raise ValueError(f"TimeSeriesSpec.nullable must be a bool, got {nullable!r}")
        missing = fields.get("missing")
        if missing is None:
            fields["missing"] = MissingValues.unspecified() if nullable else None
        elif not nullable:
            raise ValueError("TimeSeriesSpec.nullable is false but missing says that timesteps can be missing")
        return fields

    def check_nulls_within_missing_contract(self, null_count: int, n_values: int, *, series: str) -> None:
        """Check whether the spec allows the missing timesteps.

        Args:
            null_count: The number of missing timesteps in the series.
            n_values: The number of timesteps in the series.
            series: The series name to use in an error message.

        Raises:
            TimeFValidationError: If missing timesteps are disallowed, unspecified, or exceed ``max_fraction``.
        """
        if not null_count:
            return
        if self.missing is None:
            raise TimeFValidationError(
                f"{series} has {null_count} null values but its spec {self.spec_type!r} allows no missing "
                "timestep. Set TimeSeriesSpec.missing to say why values are missing"
            )
        if self.missing.reason is MissingReason.UNSPECIFIED:
            raise TimeFValidationError(
                f"{series} has {null_count} null values but its spec {self.spec_type!r} gives no reason. "
                "The 'unspecified' reason is only for releases written before TimeF recorded one. "
                "Set TimeSeriesSpec.missing to say why values are missing"
            )
        fraction = null_count / n_values
        if fraction > self.missing.max_fraction:
            raise TimeFValidationError(
                f"{series} is missing {null_count} of {n_values} values ({fraction:.1%}), more than the "
                f"max_fraction {self.missing.max_fraction} of its spec {self.spec_type!r}"
            )

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
