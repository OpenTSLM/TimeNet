"""Bounded signal windows, exact scalar encoding, and extrema reduction."""

# ruff: noqa: DOC201
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import math
from typing import TypeGuard

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timenet.dataset import Signal
from timenet.errors import TimeFValidationError
from timenet.reader.inspection import ReaderInspection
from timenet.reader.inspection.requests import WindowQuery
from timenet.reader.inspection.results import WindowBucket, WindowResult, WindowSample
from timenet.reader.inspection.window_bounds import window_bounds, x_coordinates


_MAX_RAW_STEPS = 200
_MAX_PLOT_STEPS = 20_000
_MAX_NUMERIC_BYTES = 8 * 2**20
_MAX_RAW_TEXT_CHARACTERS = 1024
_NUMERIC_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"}
)


@dataclass(frozen=True)
class Sample:
    """One scalar sample before JSON encoding."""

    index: int
    x: int
    value: object
    original_bytes: int | None = None
    truncated: bool = False


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(("null", "nan", "positive_infinity", "negative_infinity", "finite"), 0)


@dataclass
class ReductionBucket:
    """Retain first, last, extrema and one gap with exact coverage counts."""

    number: int
    first: int
    last: int
    counts: dict[str, int] = field(default_factory=_empty_counts)
    points: dict[str, Sample] = field(default_factory=dict)

    def add(self, sample: Sample) -> None:
        """Account for a sample while retaining at most five markers."""
        self.last = sample.index
        self.counts[_value_kind(sample.value)] += 1
        self.points.setdefault("first", sample)
        self.points["last"] = sample
        if _finite(sample.value):
            for name, choose_minimum in (("min", True), ("max", False)):
                previous = self.points.get(name)
                if previous is None or (
                    _finite(previous.value)
                    and ((sample.value < previous.value) if choose_minimum else (sample.value > previous.value))
                ):
                    self.points[name] = sample
        else:
            self.points.setdefault("gap", sample)

    def finish(self, samples: list[Sample], buckets: list[WindowBucket]) -> None:
        """Append chronologically ordered markers and the bucket coverage."""
        selected = {point.index: point for point in self.points.values()}
        samples.extend(selected[index] for index in sorted(selected))
        buckets.append(
            {
                "start": str(self.first),
                "stop": str(self.last + 1),
                "coverage": {key: str(value) for key, value in self.counts.items()},
            }
        )


def _validate_component(signal: Signal, query: WindowQuery) -> None:
    """Reject incompatible scalar selections before reading values.

    Raises:
        TimeFValidationError: If plotting or component selection is invalid.
    """
    if query.mode == "plot" and signal.spec.dtype not in _NUMERIC_DTYPES:
        raise TimeFValidationError("select the raw table for nonnumeric signals")
    if len(query.component) != len(signal.spec.value_shape):
        raise TimeFValidationError("select exactly one index for each tensor component dimension")
    if any(index < 0 or index >= size for index, size in zip(query.component, signal.spec.value_shape, strict=True)):
        raise TimeFValidationError("tensor component index is outside the signal shape")


def _samples(
    inspection: ReaderInspection, signal: Signal, query: WindowQuery, bounds: tuple[int, int]
) -> Iterator[Sample]:
    """Yield selected scalar samples from bounded Arrow reads.

    Yields:
        The samples in step order.
    """
    start, stop = bounds
    chunk_size = max(1, min(_MAX_PLOT_STEPS, _MAX_NUMERIC_BYTES // (8 * math.prod(signal.spec.value_shape))))
    if signal.spec.dtype in {"str", "enum"}:
        # Decode one variable-width value at a time; trim in Arrow before creating Python strings.
        chunk_size = 1
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(stop, chunk_start + chunk_size)
        array = signal.read_steps(chunk_start, chunk_stop)
        text_sizes: list[int | None] | None = None
        if signal.spec.dtype in {"str", "enum"}:
            text = array.cast(pa.string())
            text_sizes = pc.binary_length(text).to_pylist()  # ty: ignore[unresolved-attribute]
            array = pc.utf8_slice_codeunits(text, 0, _MAX_RAW_TEXT_CHARACTERS)  # ty: ignore[unresolved-attribute]
        values = _component_values(array, query.component)
        xs = x_coordinates(inspection, query.record_id, query.signal_id, signal.time_axis, chunk_start, chunk_stop)
        for offset, (index, x, value) in enumerate(zip(range(chunk_start, chunk_stop), xs, values, strict=True)):
            size = None if text_sizes is None else text_sizes[offset]
            truncated = isinstance(value, str) and size is not None and len(value.encode("utf-8")) < size
            yield Sample(index, x, value, size, truncated)


def _reduce(
    samples: Iterator[Sample], bucket_size: int | None, start: int
) -> tuple[list[Sample], list[WindowBucket], dict[str, int]]:
    """Reduce a stream without retaining the scanned signal."""
    selected: list[Sample] = []
    buckets: list[WindowBucket] = []
    counts = _empty_counts()
    bucket: ReductionBucket | None = None
    for sample in samples:
        counts[_value_kind(sample.value)] += 1
        if bucket_size is None:
            selected.append(sample)
            continue
        number = (sample.index - start) // bucket_size
        if bucket is not None and bucket.number != number:
            bucket.finish(selected, buckets)
            bucket = None
        if bucket is None:
            bucket = ReductionBucket(number, sample.index, sample.index)
        bucket.add(sample)
    if bucket is not None:
        bucket.finish(selected, buckets)
    return selected, buckets, counts


def _encode_sample(sample: Sample) -> WindowSample:
    """Encode exact values separately from the plot's numeric approximation."""
    return {
        "index": str(sample.index),
        "x": str(sample.x),
        "value": _value(sample.value),
        "kind": _value_kind(sample.value),
        "display": None if isinstance(sample.value, str) else _display_value(sample.value),
        "original_bytes": None if sample.original_bytes is None else str(sample.original_bytes),
        "value_truncated": sample.truncated,
    }


def window_result(inspection: ReaderInspection, query: WindowQuery) -> WindowResult:
    """Resolve, read, reduce, and encode one bounded scalar window.

    Raises:
        TimeFValidationError: If the selected window exceeds the read limits.
    """
    signal = inspection.signal(query.record_id, query.signal_id)
    _validate_component(signal, query)
    start, requested_stop = window_bounds(inspection, signal, query)
    stop = min(requested_stop, start + _MAX_RAW_STEPS) if query.mode == "raw" else requested_stop
    if query.mode == "plot" and stop - start > _MAX_PLOT_STEPS:
        raise TimeFValidationError("plot windows are limited to 20,000 steps; choose a smaller window")
    if signal.spec.dtype in _NUMERIC_DTYPES or signal.spec.dtype == "bool":
        # Tensor reads materialize every component, even when displaying only one.
        estimated_bytes = (stop - start) * 8 * math.prod(signal.spec.value_shape)
        if estimated_bytes > _MAX_NUMERIC_BYTES:
            raise TimeFValidationError(
                "numeric windows are limited to 8 MiB of estimated values; choose a smaller window"
            )
    reduced = query.mode == "plot" and stop - start > 2 * query.width
    bucket_size = max(1, math.ceil((stop - start) / query.width)) if reduced else None
    samples, buckets, counts = _reduce(_samples(inspection, signal, query, (start, stop)), bucket_size, start)
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
        "items": [_encode_sample(sample) for sample in samples],
    }


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


def _component_values(values: pa.Array, component: tuple[int, ...]) -> list[object]:
    """Extract one scalar component while preserving missing timesteps.

    Raises:
        TimeFValidationError: If component selection does not match tensor values.
    """
    if not component:
        return values.to_pylist()
    if not isinstance(values, pa.FixedShapeTensorArray):
        raise TimeFValidationError("tensor component selection needs fixed-shape tensor values")
    valid = values.is_valid().to_pylist()
    dense = values.to_numpy_ndarray()
    selected = np.asarray(dense[slice(None), *component]).tolist()
    return [value if observed else None for value, observed in zip(selected, valid, strict=True)]


def _value(value: object) -> str | None:
    """Encode scalar values as JSON-safe exact display strings."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return str(value)


def _finite(value: object) -> TypeGuard[int | float]:
    """Return whether an Arrow scalar value is finite and numeric."""
    return isinstance(value, int | float) and math.isfinite(value)
