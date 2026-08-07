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
machine, for example). Its type, the units of its value, time, and sampling-rate axes, comes from a
[`TimeSeriesSpec`](../types.md), so a g-scale accelerometer channel and a °C temperature channel read
through the same API.

<figure markdown="span">
  ![One channel labelled with its time_series_id, spec, and units](../assets/figures/time-series-example.svg)
</figure>

## Time offsets and timestamps

The docs and the API use two words for time, and they do not mean the same thing.

An **time offset** is a position on a series' own axis: microseconds counted from the sample's relative
zero. Every axis quantity is one, and so are a [span](annotations.md)'s bounds. A time offset says where
a value sits inside its recording, and nothing about what day that was.

A **timestamp** is an absolute point on the wall clock, in Unix microseconds. Exactly one field
carries one: a sample's `start_time`, which is what its relative zero refers to.

Wall clock therefore enters a dataset once and composes by addition:

```text
value timestamp = sample.start_time + value time offset
```

A recording with no known date has time offsets and no timestamps, and that is a supported case rather
than missing data. A 500 Hz ECG whose source gives no `base_date` is placed exactly on its own axis;
asking what calendar day it fell on has no answer, and the format prefers no answer to a fabricated
one.

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
