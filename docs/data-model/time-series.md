---
icon: lucide/waypoints
description: "Signals hold typed values on regular, irregular, or ordinal axes."
tags:
  - guide
  - concepts
---

# Signals and time series

A `Signal` is one named stream of values. Its `TimeSeriesSpec` describes each value, and its
`TimeAxis` describes the sequence.

```python
import numpy as np

from timenet.dataset import Signal
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec

temperature = Signal(
    name="Temperature",
    spec=TimeSeriesSpec(
        spec_type="temperature",
        name="Temperature",
        unit_value="degree_Celsius",
        dtype="float32",
    ),
    time_axis=RegularAxis.from_rate_hz(1),
    data=np.array([20.1, 20.3, 20.2], dtype=np.float32),
)
```

A Signal has a read-only int64 `id` and a human-readable `name`. When you add the Record of the
Signal, the dataset sets the `id`. A [Source](sources.md) owns the Signal, and a Record reaches it
through that Source hierarchy.

## Choose an axis

TimeNet provides three axis types:

| Axis | Use it when | Position unit |
| --- | --- | --- |
| `RegularAxis` | Values have a fixed sampling rate. | Microseconds derived from the rate. |
| `IrregularAxis` | The source stores one time offset per value. | Stored microseconds. |
| `OrdinalAxis` | Values have an order but no clock. | Integer steps. |

A time offset is relative to the Record origin. It does not identify a calendar date.
`Record.start_time.timestamp` supplies the absolute timestamp of that origin when it is known.
Every timed Signal in the Record uses this common relative zero.

Do not invent timestamps for an ordinal sequence or a recording with no wall-clock source.

## Describe one timestep

`TimeSeriesSpec` declares the value dtype, unit, nullability, and optional per-timestep shape.
Set its required `unit_value` to `None` for an unknown unit. Dimensionless values use
`ureg.dimensionless`.

Scalar Signals use `value_shape=()`. An image sequence can use a shape such as
`(height, width, channels)`.

The complete array shape is `(n_steps, *value_shape)`.

## Read values

Signals expose three common array methods:

```python
arrow_values = temperature.to_arrow()
numpy_values = temperature.to_numpy()
window = temperature.read_steps(0, 2)
```

`to_arrow()` preserves nulls. `to_numpy()` raises `TimeFValidationError` when the loaded values
contain nulls.

For data with nulls, Arrow provides the validity mask:

```python
values = temperature.to_arrow()
present = values.is_valid()
```

The mask marks present timesteps. The PyTorch adapter raises `TimeFValidationError` for actual nulls,
even when `nullable=True`. Consumers must handle missing data before tensor conversion.

## Cut a window

`slice()` returns a new Signal that holds only the steps a span covers. The span can be a
`TimeInterval` on the Record timeline or a `StepInterval` that names this Signal.

Which Signals a span applies to and which values it holds are separate questions:

- `is_selected_by()` answers which Signals. A `StepInterval` selects only the Signal it names. A
  `TimeInterval` with no `time_series` selects every Signal, and one with `time_series` set selects
  only the Signals it lists. Signals compare by identity, so a span does not select a slice of a
  Signal it names: the slice is a new Signal.
- `overlaps_with()` and `slice()` answer which values, by time only. They ignore a time span's
  `time_series`, so a slice can be cut again with the span it was cut with. A `StepInterval` must
  still name the Signal it slices, because its steps mean nothing on another one.

By default the span must lie inside the Signal: a `TimeInterval` inside the window `span_us`
reports, and a `StepInterval` that ends at or before `n_values`. A span that reaches past either
end raises `TimeFValidationError`. Pass `clip=True` to keep only the part of the span inside the
Signal instead.

`overlaps_with()` tells you whether a `TimeInterval` or `TimePoint` holds at least one value of a
Signal, so you can skip the Signals it misses, such as other segments of one recording. It uses the
same rule as `slice()`: a value is inside when its time offset falls in the span. An interval
between two values, or after the last one, holds no value. On an irregular Signal,
`overlaps_with()` reads the time offsets, unless the span lies outside the window `span_us`
reports.

An ordinal Signal has no timeline. `overlaps_with()` and `slice()` raise `TimeFValidationError` for
a time span on it, so test `is_ordinal` first and scope an ordinal Signal in steps. `slice()` also
raises for a point span.

To cut every Signal of a Record that an interval selects, ask `is_selected_by()` first, and cut
with `clip=True`, because an interval can overlap a Signal only in part:

```python
from timenet.types import TimeInterval

interval = TimeInterval.seconds(1, 3)
parts = []
for signal in record.walk_signals():
    if not signal.is_selected_by(interval):
        continue
    if signal.is_ordinal and interval.time_series is None:
        # Unscoped: keep the ordinal Signal whole (the caller's choice).
        parts.append(signal)
        continue
    # Raises for an ordinal Signal that the interval names: it has no timeline.
    if signal.overlaps_with(interval):
        parts.append(signal.slice(interval, clip=True))
```

The loop skips a Signal that the interval selects but holds no value of. If every Signal that the
interval names must hold a value, raise when `overlaps_with()` returns `False` for one of them.

The slice keeps the name, spec, metadata, and place on the timeline of the original Signal: a
regular axis starts at the slice's first step, an irregular axis keeps the sliced time offsets, and
an ordinal axis stays as it is. It drops the annotations of the original, because they can fall
outside the window. The slice is a new Signal that belongs to no Source, so it has no `id`.

The slice reads no values when you make it. When you read it, it asks the original for its own
steps only. That reads less than the whole original only when its values support range reads, as
for `read_steps()`: stored values read only the chunks that cover the steps, but a plain loader
callable reads all its values first. Slicing an irregular Signal reads its time offsets, and the
slice keeps a copy of its own offsets only. On in-memory NumPy values, `to_numpy()` on the
slice returns a read-only view, with no copy.

## Lazy values

A connector can construct a Signal from in-memory `data`. It can also use `Signal.from_loader()` for
a source that must remain lazy.

The TimeF reader leaves stored values behind lazy handles unless the read's
[load plan](../timef-reader.md#loading-values-at-read-time) loads them. The reader from
`TimeNet.open()` therefore reads a dataset without loading its Signal arrays, while
`TimeNet.load()` loads every value.

## Ownership and reuse

Each Signal belongs to [one Source](sources.md#ownership). Do not attach one Signal instance to
several Sources or Records.

Several Signals can share the same immutable `TimeAxis` or `TimeSeriesSpec`. TimeF stores the axis
definition once. For an irregular axis, this includes its complete offset stream. Signal values
remain independent.
