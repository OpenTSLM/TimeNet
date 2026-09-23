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
    id="temperature",
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

A Signal has a stable `id` and a human-readable `name`. A Source owns it, and a Record reaches it
through that Source hierarchy.

## Choose an axis

TimeNet provides three axis types:

| Axis | Use it when | Position unit |
| --- | --- | --- |
| `RegularAxis` | Values have a fixed sampling rate. | Microseconds derived from the rate. |
| `IrregularAxis` | The source stores one time offset per value. | Stored microseconds. |
| `OrdinalAxis` | Values have an order but no clock. | Integer steps. |

A time offset is relative to the Record timeline. It does not identify a calendar date. The optional
`Record.start_time` supplies that wall-clock anchor.

Do not invent timestamps for an ordinal sequence or a recording with no wall-clock source.

## Describe one timestep

`TimeSeriesSpec` declares the value dtype, unit, nullability, and optional per-timestep shape.

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

Use `to_numpy_and_mask()` for nullable values:

```python
values, present = temperature.to_numpy_and_mask()
```

The mask marks present timesteps. A placeholder at a missing position is not an observation.

## Lazy values

A connector can construct a Signal from in-memory `data`. It can also use `Signal.from_loader()` for
a source that must remain lazy.

The TimeF reader always supplies lazy storage loaders. Reading a dataset therefore does not load
every Signal array.

## Ownership and reuse

Each Signal belongs to one Source. Do not attach one Signal instance, or one Signal ID, to several
Sources or Records.

Several Signals can share the same immutable `TimeAxis` or `TimeSeriesSpec`. This reuse represents a
shared contract, not shared values.
