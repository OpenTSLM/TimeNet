"""Arrow values and their dense NumPy representations."""

from collections.abc import Iterable, Sequence

import numpy as np
import pyarrow as pa

from timenet.dataset.materialized_values.base import MaterializedValues
from timenet.errors import TimeFValidationError
from timenet.types import TimeSeriesSpec


def validate_enum_values(spec: TimeSeriesSpec, values: Iterable[object]) -> None:
    """Reject enum labels outside the spec's codebook.

    Raises:
        TimeFValidationError: If any label is outside the spec's categories.
    """
    allowed = set(spec.categories)
    unknown = {value for value in values if value not in allowed and not (value is None and spec.nullable)}
    if unknown:
        raise TimeFValidationError(
            f"enum series for {spec.spec_type!r} has values outside its categories "
            f"({list(spec.categories)!r}): {sorted(unknown, key=str)}"
        )


def _scalar_values_to_arrow(
    spec: TimeSeriesSpec, values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array
) -> pa.Array:
    """Convert scalar inputs before checking nullability and enum labels.

    Returns:
        Values converted to the spec's Arrow element type, preserving missing values.
    """
    arrow_type = pa.string() if spec.dtype in {"str", "enum"} else pa.from_numpy_dtype(np.dtype(spec.dtype))
    if isinstance(values, pa.Array):
        return values.cast(arrow_type)
    if isinstance(values, np.ndarray) and values.dtype.kind != "O" and spec.dtype not in {"str", "enum"}:
        # Preserve NumPy casting rules until dense arrays get their own loader.
        return pa.array(np.asarray(values, dtype=np.dtype(spec.dtype)))
    if spec.dtype == "bool":
        # Preserve None while keeping NumPy's boolean conversion rules for present values.
        values = [None if value is None else bool(value) for value in values]
    return pa.array(values, type=arrow_type)


class PyArrowMaterializedValues(MaterializedValues):
    """Retain an Arrow array, including its null bitmap and tensor shape."""

    def __init__(self, values: pa.Array) -> None:
        self._values = values

    @classmethod
    def from_values(
        cls, spec: TimeSeriesSpec, values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array
    ) -> "PyArrowMaterializedValues":
        """Convert scalar values to the spec's dtype and validate nulls and enum labels.

        Returns:
            A loader for the validated Arrow array.

        Raises:
            TimeFValidationError: If a non-nullable spec receives nulls or an enum label is unknown.
        """
        array = _scalar_values_to_arrow(spec, values)
        if array.null_count and not spec.nullable:
            raise TimeFValidationError(f"series for {spec.spec_type!r} has null values but nullable=False")
        if spec.dtype == "enum":
            array = array.dictionary_encode()
            validate_enum_values(spec, array.dictionary.to_pylist())
        return cls(array)

    def __len__(self) -> int:
        return len(self._values)

    def to_arrow(self) -> pa.Array:
        """Return the retained Arrow values.

        Returns:
            The complete Arrow array.
        """
        return self._values

    def to_numpy(self) -> np.ndarray:
        """Read dense values, preserving tensor shape.

        Returns:
            Values shaped ``(n_steps, *value_shape)``.

        Raises:
            TimeFValidationError: If a timestep is missing or only partially present.
        """
        values = self._values
        if values.null_count:
            raise TimeFValidationError("series contains null values. Use to_arrow() to preserve missingness")
        if isinstance(values, pa.FixedShapeTensorArray):
            flat = values.storage.flatten()
            if flat.null_count:
                raise TimeFValidationError("series may have nulls only for whole timesteps")
            dense = flat.to_numpy(zero_copy_only=False).reshape((len(values), *values.type.shape))
            if values.type.permutation:
                time_axis, *value_axes = range(dense.ndim)
                permuted_value_axes = [value_axes[axis] for axis in values.type.permutation]
                dense = dense.transpose(time_axis, *permuted_value_axes)
            return dense
        return values.to_numpy(zero_copy_only=False)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return an Arrow slice of the stored values.

        Returns:
            The requested Arrow values.
        """
        return self._values.slice(start, stop - start)
