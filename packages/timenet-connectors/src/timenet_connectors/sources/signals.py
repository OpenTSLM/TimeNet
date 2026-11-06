"""Validate decoded scalar samples before constructing eager or lazy Signals.

These helpers do not calibrate, reshape, repair, or infer missing values. Use
``fields.error_context`` in the reader to attach a source file or row location.
"""

import math
from typing import Literal

import numpy as np
import pyarrow as pa

from timenet.errors import TimeNetConnectorError
from timenet.types import TimeSeriesSpec


def _count(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise TypeError(f"{name} must be a nonnegative integer or None")


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


def _convert_sample(
    sample: object, dtype: np.dtype, nullable: bool, casting: str
) -> tuple[int | float | None, str | None]:
    """Return a converted sample and an optional failure reason."""
    reason = None
    result = None
    if sample is None:
        reason = None if nullable else "null requires a nullable spec"
    elif isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, float, np.integer, np.floating)):
        reason = "expected a finite number"
    else:
        # Keep integers as Python ints: mixed NumPy comparisons can round a large
        # integer to the very float whose loss of precision we need to detect.
        original = int(sample) if isinstance(sample, (int, np.integer)) else float(sample)
        if isinstance(original, float) and not math.isfinite(original):
            reason = "expected a finite number"
        elif dtype.kind in "iu":
            limits = np.iinfo(dtype)
            if original < int(limits.min) or original > int(limits.max) or int(original) != original:
                reason = f"cannot be represented exactly as {dtype}"
            else:
                result = int(original)
        else:
            try:
                with np.errstate(over="ignore", under="ignore"):
                    result = float(dtype.type(original))
            except OverflowError:
                reason = f"overflows {dtype}"
            else:
                if not math.isfinite(result):
                    reason = f"overflows {dtype}"
                elif result != original and (casting == "exact" or isinstance(original, int)):
                    reason = f"cannot be represented exactly as {dtype}"
    return result, reason


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
        TypeError: If the casting policy or expected count is invalid.
        TimeNetConnectorError: If the source or spec is unsupported or a sample is invalid.
    """
    _count(n_values, "n_values")
    if casting not in {"exact", "float_rounding"}:
        raise TypeError(f"unsupported casting policy {casting!r}")
    if spec.value_shape or spec.dtype in {"bool", "str", "enum"}:
        raise TimeNetConnectorError(f"{name}: expected a scalar numeric spec")
    dtype = np.dtype(spec.dtype)
    # The common waveform path stays vectorized. Invalid arrays fall through to
    # the scalar path to report the earliest failing sample consistently.
    if (
        isinstance(values, np.ndarray)
        and not isinstance(values, np.ma.MaskedArray)
        and values.ndim == 1
        and values.dtype in {np.dtype("float32"), np.dtype("float64")}
        and dtype.kind == "f"
    ):
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
        TypeError: If an axis or count argument is invalid.
        TimeNetConnectorError: If dimensions or counts disagree with the source contract.
    """
    _count(n_values, "n_values")
    _count(n_channels, "n_channels")
    if n_channels is None or n_channels == 0:
        raise TypeError("n_channels must be a positive integer")
    if type(sample_axis) is not int or sample_axis not in {0, 1}:
        raise TypeError("sample_axis must be 0 or 1")
    if not isinstance(values, np.ndarray) or len(values.shape) != len((0, 1)):
        raise TimeNetConnectorError(f"{name}: expected a two-dimensional channel matrix")
    if values.shape[1 - sample_axis] != n_channels:
        raise TimeNetConnectorError(f"{name}: expected {n_channels} channels, got {values.shape[1 - sample_axis]}")
    if n_values is not None and values.shape[sample_axis] != n_values:
        raise TimeNetConnectorError(f"{name}: expected {n_values} samples, got {values.shape[sample_axis]}")
