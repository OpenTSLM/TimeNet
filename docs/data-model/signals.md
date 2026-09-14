---
icon: lucide/waypoints
description: "Signals: one sequence of values on one time axis, typed by a spec."
tags:
  - guide
  - concepts
---

# Signals

A signal is the leaf of the hierarchy: one sequence of values, one time axis, one spec. A
[source](records.md) produces one or more, such as the vibration and temperature signals of a
machine. The [`TimeSeriesSpec`](../types.md) gives the modality and the unit of the values, so a
g-scale accelerometer signal and a °C temperature signal read through the same API.

<figure markdown="span">
  ![One signal labelled with its id, spec, and units](../assets/figures/time-series-example.svg)
</figure>

```python
from fractions import Fraction

from timenet.control_plane import Signal
from timenet.dataset.axis import RegularAxis

vibration = Signal(
    id="run-1-vibration",
    name="vibration",
    values=values,                                        # a 1-D numpy array
    time_axis=RegularAxis(period_us=Fraction(1_000_000, 500)),
    spec=vibration_spec,
)
```

`values` is a 1-D NumPy array, and its dtype is what comes back out. Nothing pads, resamples, or
casts it. Two signals of one record need share neither a length nor a dtype.

## Time axes

TimeF has three axis shapes, and a signal declares one.

| Axis | What it says | What it stores |
| --- | --- | --- |
| `RegularAxis` | Value `k` sits at `(start_index + k) * period_us`. | A period and an origin index. Nothing per value. |
| `IrregularAxis` | A placement no formula produces. | Every time offset, beside the values in the values plane. The axis itself keeps only the first and last. |
| `OrdinalAxis` | Order only: no cadence, no offsets, no place on a timeline. | Nothing. |

`period_us` is a `Fraction` of microseconds, not a float rate. Not every real rate is a whole number
of them: 360 Hz is 25000/9 µs and 256 Hz is 15625/4. A Fraction keeps placing a value and locating a
time offset exactly inverse, with no tolerance band.

`start_index` is an index into the cadence rather than a time, because a window cut from a longer
recording rarely starts on a whole microsecond. At 44.1 kHz only 3 of 1000 window starts do. An
index is exact for all of them.

## Time offsets and timestamps

The docs and the API use two words for time, and they do not mean the same thing.

A **time offset** is a position on the signal's own axis: microseconds from the record's relative
zero. Every axis quantity is a time offset, and so are the bounds of an
[annotation](annotations.md). A time offset says where a value sits inside its recording. It says
nothing about the calendar day.

A **timestamp** is an absolute point on the wall clock, in Unix microseconds. Exactly one field
carries one: `start_time_us` on a [record](records.md).

The wall clock therefore enters a dataset once, and it composes by addition:

```text
value timestamp = record.start_time_us + value time offset
```

A recording with no known date has time offsets and no timestamps. That is a supported case, not
missing data. A 500 Hz ECG whose source gives no base date sits exactly on its own axis; which
calendar day it fell on has no answer, and the format prefers no answer to a fabricated one.

## Reading values

Values live in the values plane, outside the control plane, and are read by signal id.

```python
with TimeNet("./local_registry").open("demo/machine") as reader:
    record = reader.record("run-1")
    signals = record.signals()
    values = reader.values_for([s.signal_id for s in signals])
    vibration = values[signals[0].signal_id]  # numpy, in the stored dtype
```

`values_for()` reads a whole batch in one pass. `values()` reads a single signal, and is the slow
way round when you want more than one: it re-opens the shard and re-decodes the row group per call.
To read part of a signal, `values_window(signal_id, start, stop)` reads only the chunks the window
crosses.

## Sharing across records

To share one signal between several records, put the same `Signal` object in each source that
produces it. The link between a source and a signal is a table row, so the values are written once
however many records reference them. Real corpora rely on this: 7,013 of ARFBench's 9,187 series are
referenced by more than one record.
