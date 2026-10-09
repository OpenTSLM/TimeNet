"""Bounded signal windows, exact scalar encoding, and extrema reduction."""

# ruff: noqa: DOC201
from __future__ import annotations

import math
from typing import Any

import numpy as np

from timenet.dataset import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.inspection.requests import WindowQuery
from timenet.viewer.jobs import checkpoint


_MAX_RAW_STEPS = 200
_MAX_PLOT_STEPS = 20_000
_NUMERIC_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"}
)


def window_result(inspection: ViewerInspection, query: WindowQuery) -> dict[str, object]:  # noqa: PLR0912, PLR0914, PLR0915
    """Read bounded chunks and serialize a raw page or a reduced plot.

    Raises:
        TimeFValidationError: If the selected component or window is invalid.
    """
    signal = inspection.signal(query.record_id, query.signal_id)
    if query.mode == "plot" and signal.spec.dtype not in _NUMERIC_DTYPES:
        raise TimeFValidationError("select the raw table for nonnumeric signals")
    if len(query.component) != len(signal.spec.value_shape):
        raise TimeFValidationError("select exactly one index for each tensor component dimension")
    if any(index < 0 or index >= size for index, size in zip(query.component, signal.spec.value_shape, strict=True)):
        raise TimeFValidationError("tensor component index is outside the signal shape")
    start, stop = _window_bounds(inspection, signal, query)
    requested_stop = stop
    if query.mode == "raw":
        stop = min(stop, start + _MAX_RAW_STEPS)
    counts = dict.fromkeys(("null", "nan", "positive_infinity", "negative_infinity", "finite"), 0)
    samples: list[tuple[int, int, object]] = []
    buckets: list[dict[str, object]] = []
    reduced = query.mode == "plot" and stop - start > 2 * query.width
    bucket_size = max(1, math.ceil((stop - start) / query.width)) if reduced else 1
    bucket: dict[str, Any] = {}

    def finish_bucket() -> None:
        if not bucket:
            return
        selected = {item[0]: item for item in bucket["points"].values()}
        samples.extend(selected[index] for index in sorted(selected))
        buckets.append(
            {
                "start": str(bucket["first"]),
                "stop": str(bucket["last"] + 1),
                "coverage": {key: str(value) for key, value in bucket["counts"].items()},
            }
        )
        bucket.clear()

    chunk_size = max(1, min(_MAX_PLOT_STEPS, (8 * 2**20) // (8 * math.prod(signal.spec.value_shape))))
    for chunk_start in range(start, stop, chunk_size):
        checkpoint(chunk_start - start, stop - start)
        chunk_stop = min(stop, chunk_start + chunk_size)
        values = _component_values(signal.read_steps(chunk_start, chunk_stop), query.component)
        xs = _x_coordinates(inspection, query.record_id, query.signal_id, signal.time_axis, chunk_start, chunk_stop)
        for index, x, value in zip(range(chunk_start, chunk_stop), xs, values, strict=True):
            kind = _value_kind(value)
            counts[kind] += 1
            sample = (index, x, value)
            if not reduced:
                samples.append(sample)
                continue
            number = (index - start) // bucket_size
            if bucket and bucket["number"] != number:
                finish_bucket()
            if not bucket:
                bucket.update(number=number, first=index, counts=dict.fromkeys(counts, 0), points={"first": sample})
            bucket["last"] = index
            bucket["counts"][kind] += 1
            points = bucket["points"]
            points["last"] = sample
            if _finite(value):
                if "min" not in points or value < points["min"][2]:
                    points["min"] = sample
                if "max" not in points or value > points["max"][2]:
                    points["max"] = sample
            elif "gap" not in points:
                points["gap"] = sample
    finish_bucket()
    checkpoint(stop - start, stop - start)
    return {
        "api_version": 1,
        "record_id": query.record_id,
        "signal_id": query.signal_id,
        "signal_name": signal.name,
        "spec_name": signal.spec.name,
        "unit": None if signal.spec.unit_value is None else str(signal.spec.unit_value),
        "dtype": signal.spec.dtype,
        "component": list(query.component),
        "axis_kind": signal.time_axis.axis_type.value,
        "mode": query.mode,
        "requested": {"start": str(start), "stop": str(requested_stop)},
        "scanned_count": str(stop - start),
        "coverage": {key: str(value) for key, value in counts.items()},
        "reduced": reduced,
        "buckets": buckets,
        "next_start": str(stop) if stop < requested_stop else None,
        "items": [
            {
                "index": str(index),
                "x": str(x),
                "value": _value(value),
                "kind": _value_kind(value),
                "display": _display_value(value),
            }
            for index, x, value in samples
        ],
    }


def _window_bounds(inspection: Any, signal: Any, query: WindowQuery) -> tuple[int, int]:
    """Locate a common time window independently on each selected signal.

    Raises:
        TimeFValidationError: If the window is reversed or time is used with an ordinal axis.
    """
    start = query.start
    stop = query.stop
    axis = signal.time_axis
    if len(query.component) != len(signal.spec.value_shape) or any(
        index < 0 or index >= size for index, size in zip(query.component, signal.spec.value_shape, strict=True)
    ):
        raise TimeFValidationError("select a valid tensor component")

    def lower_bound(target: int) -> int:
        low, high = 0, signal.n_values
        while low < high:
            checkpoint()
            middle = (low + high) // 2
            value = _x_coordinates(inspection, query.record_id, query.signal_id, axis, middle, middle + 1)[0]
            if value < target:
                low = middle + 1
            else:
                high = middle
        return low

    if query.start_us is not None or query.end_us is not None:
        if isinstance(axis, OrdinalAxis):
            raise TimeFValidationError("ordinal signals require a step window")
        start = lower_bound(query.start_us) if query.start_us is not None else 0
        stop = lower_bound(query.end_us) if query.end_us is not None else signal.n_values
    elif stop is None:
        if query.full:
            stop = signal.n_values
        elif isinstance(axis, OrdinalAxis):
            stop = min(signal.n_values, start + 2000)
        elif start < signal.n_values:
            first = _x_coordinates(inspection, query.record_id, query.signal_id, axis, start, start + 1)[0]
            stop = lower_bound(first + 10_000_000)
        else:
            stop = start
    if start > stop or stop > signal.n_values or start > signal.n_values:
        raise TimeFValidationError("window must satisfy 0 <= start <= stop <= signal length")
    return start, stop


def _value_kind(value: object) -> str:
    """Classify nulls and nonfinite values without conflating them."""
    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == math.inf:
            return "positive_infinity"
        if value == -math.inf:
            return "negative_infinity"
    return "finite"


def _display_value(value: object) -> str | None:
    """Return exact raw scalar text, including nonfinite values."""
    return None if value is None else str(value)


def _component_values(values: Any, component: tuple[int, ...]) -> list[object]:
    """Extract one scalar component while preserving missing timesteps.

    Raises:
        TimeFValidationError: If component selection does not match tensor values.
    """
    if not component:
        return values.to_pylist()
    if not hasattr(values, "to_numpy_ndarray"):
        raise TimeFValidationError("tensor component selection needs fixed-shape tensor values")
    valid = values.is_valid().to_pylist()
    dense = values.to_numpy_ndarray()
    selected = np.asarray(dense[slice(None), *component]).tolist()
    return [value if observed else None for value, observed in zip(selected, valid, strict=True)]


def _x_coordinates(  # noqa: PLR0913, PLR0917
    inspection: Any, record_id: str, signal_id: str, axis: Any, start: int, stop: int
) -> tuple[int, ...]:
    """Return exact step or time coordinates for a bounded window.

    Raises:
        TimeFValidationError: If the axis shape is unsupported.
    """
    if isinstance(axis, RegularAxis):
        return tuple(axis.time_offset_us(index) for index in range(start, stop))
    if isinstance(axis, IrregularAxis):
        return tuple(inspection.offsets(record_id, signal_id, start, stop).to_pylist())
    if isinstance(axis, OrdinalAxis):
        return tuple(range(start, stop))
    raise TimeFValidationError("unsupported signal axis")


def _reduce(samples: tuple[tuple[int, int, object], ...], width: int) -> tuple[tuple[int, int, object], ...]:
    """Keep extrema in bounded display buckets without silently stride-sampling."""
    if len(samples) <= 2 * width:
        return samples
    reduced: list[tuple[int, int, object]] = []
    bucket_size = math.ceil(len(samples) / width)
    for start in range(0, len(samples), bucket_size):
        bucket = samples[start : start + bucket_size]
        finite = [item for item in bucket if _finite(item[2])]
        if not finite:
            reduced.append(bucket[0])
            continue
        minimum, maximum = (
            min(finite, key=lambda item: _as_float(item[2])),
            max(finite, key=lambda item: _as_float(item[2])),
        )
        selected = {item[0]: item for item in (bucket[0], minimum, maximum, bucket[-1])}
        gap = next((item for item in bucket if not _finite(item[2])), None)
        if gap is not None:
            selected[gap[0]] = gap
        reduced.extend(selected[index] for index in sorted(selected))
    return tuple(reduced)


def _value(value: object) -> str | None:
    """Encode scalar values as JSON-safe exact display strings."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return str(value)


def _coverage(values: list[object]) -> dict[str, str]:
    """Count missing and non-finite values without collapsing their meanings."""
    counts = {"null": 0, "nan": 0, "positive_infinity": 0, "negative_infinity": 0, "finite": 0}
    for value in values:
        if value is None:
            counts["null"] += 1
        elif isinstance(value, float) and math.isnan(value):
            counts["nan"] += 1
        elif isinstance(value, float) and value == math.inf:
            counts["positive_infinity"] += 1
        elif isinstance(value, float) and value == -math.inf:
            counts["negative_infinity"] += 1
        else:
            counts["finite"] += 1
    return {name: str(count) for name, count in counts.items()}


def _finite(value: object) -> bool:
    """Return whether an Arrow scalar value is finite and numeric."""
    return isinstance(value, int | float) and math.isfinite(value)


def _as_float(value: object) -> float:
    """Return a numeric Arrow scalar as a display float.

    Raises:
        TimeFValidationError: If a non-numeric value reaches numeric reduction.
    """
    if not isinstance(value, int | float):
        raise TimeFValidationError("plot reduction received a non-numeric value")
    return float(value)
