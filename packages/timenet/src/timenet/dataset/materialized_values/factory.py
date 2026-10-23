"""Select value storage and adapt callable loaders to the common interface."""

from collections.abc import Callable, Sequence
from typing import TypeGuard

import numpy as np
import pyarrow as pa

from timenet.dataset.materialized_values.arrow_values import PyArrowMaterializedValues
from timenet.dataset.materialized_values.base import MaterializedValues, ValuesLoader
from timenet.dataset.materialized_values.callable_values import CallableValuesLoader
from timenet.dataset.materialized_values.numpy_values import NumpyMaterializedValues
from timenet.types import TimeSeriesSpec


def _can_use_numpy_storage(
    values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
    spec: TimeSeriesSpec,
) -> TypeGuard[np.ndarray]:
    """Check whether the input can use the NumPy values loader.

    Object arrays use Arrow to preserve possible ``None`` values. Text and enum specs use Arrow
    for string and dictionary encoding. The NumPy loader checks shape and normalizes dtype and layout.

    Returns:
        Whether the input is an array eligible for NumPy storage under this spec.
    """
    return isinstance(values, np.ndarray) and values.dtype.kind != "O" and spec.dtype not in {"str", "enum"}


def materialized_values(
    spec: TimeSeriesSpec,
    values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
    *,
    validate_arrow: bool = False,
) -> MaterializedValues:
    """Choose a loader, retaining already-canonical Arrow input unless validation is requested.

    Returns:
        A NumPy loader for dense numeric arrays, otherwise an Arrow loader.
    """
    if isinstance(values, pa.Array) and not validate_arrow:
        return PyArrowMaterializedValues(values)
    if _can_use_numpy_storage(values, spec):
        return NumpyMaterializedValues.from_values(spec, values)
    return PyArrowMaterializedValues.from_values(spec, values)


def as_values_loader(loader: ValuesLoader | Callable[[], pa.Array]) -> ValuesLoader:
    """Adapt a legacy callable once, without reading its values.

    Returns:
        A loader with explicit Arrow, NumPy, and step-range methods.
    """
    if isinstance(loader, ValuesLoader):
        return loader
    return CallableValuesLoader(loader)
