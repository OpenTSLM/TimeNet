---
icon: lucide/waypoints
description: "Time series: one channel of a sample, float32 values over time, typed by a spec."
tags:
  - guide
  - concepts
---

# Time series

A time series is one channel of a [sample](samples.md): `float32` values over time. A sample carries one
or more of them, each identified by a `time_series_id` (the vibration and temperature channels of a
machine, for example). Its type, the units of its value, timestamp, and sampling-rate axes, comes from a
[`TimeSeriesSpec`](../types.md), so a g-scale accelerometer channel and a °C temperature channel read
through the same API.

<figure markdown="span">
  ![One channel labelled with its time_series_id, spec, and units](../assets/figures/time-series-example.svg)
</figure>

## Reading values

Values load lazily, backed by Apache Arrow, so opening a dataset does not pull every array into memory.
Read a channel with `to_numpy()` or `to_arrow()`:

```python
from timenet.client import TimeNet

dataset = TimeNet().load("chengsenwang/tsqa")
series = dataset.samples[0].time_series[0]
values = series.to_numpy()   # a float32 numpy array
```

## Sharing across samples

To share one channel across several samples, attach the same `TimeSeries` instance (or two instances
with the same explicit `time_series_id`) to each. The [writer](../timef-writer.md) dedupes by
`time_series_id`, so the bytes are stored once no matter how many samples reference them.
