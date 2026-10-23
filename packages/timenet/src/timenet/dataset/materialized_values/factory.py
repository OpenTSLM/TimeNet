"""Select value storage and adapt callable loaders to the common interface."""

from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa

from timenet.dataset.materialized_values.arrow_values import PyArrowMaterializedValues
from timenet.dataset.materialized_values.base import MaterializedValues, ValuesLoader
from timenet.dataset.materialized_values.callable_values import CallableValuesLoader
from timenet.types import TimeSeriesSpec


def materialized_values(
    spec: TimeSeriesSpec,
    values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
    *,
    validate_arrow: bool = False,
) -> MaterializedValues:
    """Choose a loader, retaining already-canonical Arrow input unless validation is requested.

    Returns:
        A loader for materialized Arrow values.
    """
    if isinstance(values, pa.Array) and not validate_arrow:
        return PyArrowMaterializedValues(values)
    return PyArrowMaterializedValues.from_values(spec, values)


def as_values_loader(loader: ValuesLoader | Callable[[], pa.Array]) -> ValuesLoader:
    """Adapt a legacy callable once, without reading its values.

    Returns:
        A loader with explicit Arrow, NumPy, and step-range methods.
    """
    if isinstance(loader, ValuesLoader):
        return loader
    return CallableValuesLoader(loader)
