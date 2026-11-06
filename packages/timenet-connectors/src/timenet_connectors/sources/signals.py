"""Validate decoded scalar samples before constructing eager or lazy Signals.

These helpers do not calibrate, reshape, repair, or infer missing values. Use
``errors.error_context`` in the reader to attach a source file or row location.
"""

import math
from typing import Literal

import numpy as np
import pyarrow as pa

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.types import TimeSeriesSpec


def _count(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise TimeFValidationError(f"{name} must be a nonnegative integer or None")


def _samples(values: object, name: str) -> list:
    """Snapshot supported scalar inputs without coercing mixed Python values.

    Returns:
        Independent Python samples with explicit nulls preserved.

    Raises:
        TimeNetConnectorError: If the input has an unsupported representation.
    """
    if isinstance(values, (list, tuple)):
        return list(values)
    if (
        isinstance(values, np.ndarray)
        and not isinstance(values, np.ma.MaskedArray)
        and values.ndim == 1
        and values.dtype.kind in "iuf"
        and values.dtype.itemsize <= np.dtype("float64").itemsize
    ):
        return values.tolist()
    if isinstance(values, pa.Array) and (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
        return values.to_pylist()
    raise TimeNetConnectorError(f"{name}: expected a flat numeric list, NumPy array, or Arrow array")


def _integer_sample(original: int | float, dtype: np.dtype) -> tuple[int | None, str | None]:
    """Return an exact integer conversion or its failure reason."""
    limits = np.iinfo(dtype)
    if original < int(limits.min) or original > int(limits.max) or int(original) != original:
        return None, f"cannot be represented exactly as {dtype}"
    return int(original), None


def _float_sample(original: int | float, dtype: np.dtype, casting: str) -> tuple[float | None, str | None]:
    """Return a float conversion with the requested precision or its failure reason."""
    try:
        with np.errstate(over="ignore", under="ignore"):
            result = float(dtype.type(original))
    except OverflowError:
        return None, f"overflows {dtype}"
    if not math.isfinite(result):
        return result, f"overflows {dtype}"
    if result != original and (casting == "exact" or isinstance(original, int)):
        return result, f"cannot be represented exactly as {dtype}"
    return result, None


def _convert_sample(
    sample: object, dtype: np.dtype, nullable: bool, casting: str
) -> tuple[int | float | None, str | None]:
    """Return a converted sample and an optional failure reason."""
    if sample is None:
        return None, None if nullable else "null requires a nullable spec"
    if isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, float, np.integer, np.floating)):
        return None, "expected a finite number"
    # Keep integers as Python ints: mixed NumPy comparisons can round a large
    # integer to the very float whose loss of precision we need to detect.
    original = int(sample) if isinstance(sample, (int, np.integer)) else float(sample)
    if isinstance(original, float) and not math.isfinite(original):
        return None, "expected a finite number"
    if dtype.kind in "iu":
        return _integer_sample(original, dtype)
    return _float_sample(original, dtype, casting)


def _float_array(values: object, dtype: np.dtype, casting: str, name: str, n_values: int | None) -> pa.Array | None:
    """Return a validated NumPy float snapshot, or None to use scalar diagnostics.

    Raises:
        TimeNetConnectorError: If a supported float array has the wrong sample count.
    """
    if not (
        isinstance(values, np.ndarray)
        and not isinstance(values, np.ma.MaskedArray)
        and values.ndim == 1
        and values.dtype in {np.dtype("float32"), np.dtype("float64")}
        and dtype.kind == "f"
    ):
        return None
    if n_values is not None and len(values) != n_values:
        raise TimeNetConnectorError(f"{name}: expected {n_values} samples, got {len(values)}")
    with np.errstate(over="ignore", under="ignore"):
        snapshot = values.astype(dtype, copy=True)
    if (
        np.isfinite(values).all()
        and np.isfinite(snapshot).all()
        and (casting == "float_rounding" or np.array_equal(snapshot, values))
    ):
        return pa.array(snapshot)
    return None


def numeric_signal(
    values: object,
    name: str,
    *,
    spec: TimeSeriesSpec,
    n_values: int | None = None,
    casting: Literal["exact", "float_rounding"] = "exact",
) -> pa.Array:
    """Return scalar samples in the spec's Arrow dtype, with independent storage.

    Nulls require a nullable spec. NaN, infinity, booleans, nested samples, and
    nonnumeric specs are rejected. Empty inputs are allowed. Exact casting is
    value-based; float_rounding permits only float-to-float precision loss,
    including underflow to zero. Integer conversions remain exact. Python inputs
    are checked before coercion can erase boolean or integer precision evidence.

    Args:
        values: A flat list/tuple, numeric NumPy array, or primitive numeric Arrow array.
        name: Field or channel name for diagnostics; sample indices are zero-based.
        spec: The scalar numeric signal contract.
        n_values: Expected sample count, if known.
        casting: Whether floating-point rounding is explicitly permitted.

    Returns:
        An Arrow array whose values cannot change through mutation of the input.

    Raises:
        TimeFValidationError: If the casting policy or expected count is invalid.
        TimeNetConnectorError: If the source or spec is unsupported or a sample is invalid.
    """
    _count(n_values, "n_values")
    if casting not in {"exact", "float_rounding"}:
        raise TimeFValidationError(f"unsupported casting policy {casting!r}")
    if spec.value_shape or spec.dtype in {"bool", "str", "enum"}:
        raise TimeNetConnectorError(f"{name}: expected a scalar numeric spec")
    dtype = np.dtype(spec.dtype)
    # Invalid arrays fall through to scalar diagnostics for the earliest failing sample.
    snapshot = _float_array(values, dtype, casting, name, n_values)
    if snapshot is not None:
        return snapshot
    samples = _samples(values, name)
    if n_values is not None and len(samples) != n_values:
        raise TimeNetConnectorError(f"{name}: expected {n_values} samples, got {len(samples)}")
    converted: list[int | float | None] = []
    for index, sample in enumerate(samples):
        result, reason = _convert_sample(sample, dtype, spec.nullable, casting)
        if reason is not None:
            raise TimeNetConnectorError(f"{name}[{index}]: {reason}")
        converted.append(result)
    return pa.array(converted, type=pa.from_numpy_dtype(dtype))


def check_channel_matrix(
    values: np.ndarray,
    name: str,
    *,
    sample_axis: Literal[0, 1],
    n_channels: int,
    n_values: int | None = None,
) -> None:
    """Check declared matrix layout without scanning, casting, or transposing values.

    Raises:
        TimeFValidationError: If an axis or count argument is invalid.
        TimeNetConnectorError: If dimensions or counts disagree with the source contract.
    """
    _count(n_values, "n_values")
    _count(n_channels, "n_channels")
    if n_channels is None or n_channels == 0:
        raise TimeFValidationError("n_channels must be a positive integer")
    if type(sample_axis) is not int or sample_axis not in {0, 1}:
        raise TimeFValidationError("sample_axis must be 0 or 1")
    if not isinstance(values, np.ndarray) or len(values.shape) != len((0, 1)):
        raise TimeNetConnectorError(f"{name}: expected a two-dimensional channel matrix")
    if values.shape[1 - sample_axis] != n_channels:
        raise TimeNetConnectorError(f"{name}: expected {n_channels} channels, got {values.shape[1 - sample_axis]}")
    if n_values is not None and values.shape[sample_axis] != n_values:
        raise TimeNetConnectorError(f"{name}: expected {n_values} samples, got {values.shape[sample_axis]}")
