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

A Signal has a read-only int64 `id` and a human-readable `name`. The dataset sets the `id` when you
add the Record of the Signal. A Source owns the Signal, and a Record reaches it through that Source
hierarchy.

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

By default the span must lie inside the Signal: a `TimeInterval` inside the window `span_us`
reports, and a `StepInterval` that ends at or before `n_values`. A span that reaches past either
end raises `TimeFValidationError`. Pass `clip=True` to keep only the part of the span inside the
Signal instead.

`overlaps_with()` tells you whether a `TimeInterval` reaches a Signal at all, so you can skip the
Signals it misses, such as other segments of one recording. An interval can overlap a Signal only
in part, so cut it with `clip=True`:

```python
from timenet.types import TimeInterval

interval = TimeInterval.seconds(1, 3)
if temperature.overlaps_with(interval):
    part = temperature.slice(interval, clip=True)
```

The span must still hold at least one value. An interval between two values of the Signal
overlaps it, but `slice()` raises `TimeFValidationError` for it, with or without `clip`.

The slice keeps the name, spec, metadata, and place on the timeline of the original Signal. It
drops the annotations of the original, because they can fall outside the window. The slice reads
no values when you make it: each read of the slice reads only its own steps from the original.

An ordinal Signal has no timeline. `overlaps_with()` and a `TimeInterval` passed to `slice()` raise
`TimeFValidationError` on it, so scope an ordinal Signal in steps. `slice()` also raises for a
point span.

## Lazy values

A connector can construct a Signal from in-memory `data`. It can also use `Signal.from_loader()` for
a source that must remain lazy.

The TimeF reader leaves stored values behind lazy handles unless the read's
[load plan](../timef-reader.md#loading-values-at-read-time) loads them. The reader from
`TimeNet.open()` therefore reads a dataset without loading its Signal arrays, while
`TimeNet.load()` loads every value.

## Ownership and reuse

Each Signal belongs to one Source. Do not attach one Signal instance, or one Signal ID, to several
Sources or Records.

Several Signals can share the same immutable `TimeAxis` or `TimeSeriesSpec`. TimeF stores the axis
definition once. For an irregular axis, this includes its complete offset stream. Signal values
remain independent.
