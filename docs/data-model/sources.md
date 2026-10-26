---
icon: lucide/cpu
description: "A Source groups the Signals of one device, sensor, or subsystem inside a Record."
tags:
  - guide
  - concepts
---

# Sources

A `Source` is a device, sensor, or subsystem inside a [Record](records.md). It sits between the
Record and its [Signals](time-series.md), and it groups the Signals that one device produces. For
example, a Record can keep a chest strap apart from a wrist watch. It can also keep an
accelerometer apart from a gyroscope inside one IMU.

A Source can hold Signals, child Sources, or both. A Signal is always a leaf.

## Create a Source

Build the tree from the leaves upward. The `vibration()` helper builds a 10 Hz Signal with 50
values, as in [Time series](time-series.md).

```python
import numpy as np

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec


def vibration(name: str) -> Signal:
    return Signal(
        name=name,
        spec=TimeSeriesSpec(
            spec_type="acceleration",
            name="Acceleration",
            unit_value="m/s**2",
            dtype="float32",
        ),
        time_axis=RegularAxis.from_rate_hz(10),
        data=np.zeros(50, dtype=np.float32),
    )


accelerometer = Source(
    name="Accelerometer",
    signals=(vibration("x"), vibration("y")),
    metadata={"model": "ADXL355"},
)
gearbox = Source(name="Gearbox", sources=(accelerometer,))
record = Record(sources=(gearbox,))

print([source.name for source in record.walk_sources()])
print([signal.name for signal in record.signals])
```

```text
['Gearbox', 'Accelerometer']
['x', 'y']
```

All fields are keyword-only:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Human-readable name. It must not be empty. |
| `sources` | `tuple[Source, ...]` | `()` | Direct child Sources. Their order has no meaning. |
| `signals` | `tuple[Signal, ...]` | `()` | Signals that this Source produces directly. |
| `annotations` | `tuple[Annotation, ...]` | `()` | Annotation occurrences on this Source. |
| `metadata` | JSON-compatible mapping | `{}` | Untyped device details. |

## Nest Sources

`walk_sources()` yields the Source and every descendant. It walks depth-first and sorts by name at
each level. `walk_signals()` yields every Signal below the Source, sorted by name within each
Source. `Record.signals` and `Record.walk_sources()` use the same order. Thus the order in which
you pass children has no effect.

Names do not have to be unique. If two Signals below one Source have the same name,
`select(signal_names=...)` rejects that name as ambiguous. In that case, select the objects.

## Annotate a Source

`source.annotate(annotation)` attaches one occurrence and returns it. A static annotation
describes the whole device, for example its firmware version:

```python
from timenet.types import Annotation, TimeInterval

gearbox.annotate(Annotation(key="firmware", value="2.4.1"))
gearbox.annotate(
    Annotation(
        key="fault",
        value="bearing",
        span=TimeInterval.seconds(1.0, 2.0),
    )
)
```

A timed annotation covers the subtree of the Source. TimeNet compares its span with the union of
the windows of all Signals below the Source. If the span is outside that window, TimeNet gives a
`SpanOutsideWindowWarning` and keeps the span as given. A Source with no timed Signals rejects a
timed annotation. The span cannot set `time_series`. To annotate a subset of Signals, use
`source.select(...)`. See [Annotations](annotations.md).

## Sources have no ID

A Source has no `id`. The dataset numbers Records, Signals, and Tasks, but not Sources. Refer to a
Source by the object, for example from `record.sources` or `walk_sources()`. Keep a serial number
or an upstream device code in `metadata` or in a static annotation.

## Ownership

Each Source has one parent. The parent is a Source, or the Record that lists it as a root. Each
Signal belongs to one Source. If two Records use the same device, build a separate Source for each
Record.

`walk_sources()` rejects a Source under two parents, and it rejects a cycle in the tree. When you
construct a Record, it does the same checks. `add_record()` rejects a Source or Signal that
another Record already owns. It also rejects a Signal that one Record attaches twice.

```python
from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata

dataset = TimeFDataset(
    metadata=DatasetMetadata(
        dataset_id="demo/gearbox",
        dataset_version="1.0.0",
        name="Gearbox demo",
        description="One gearbox run.",
        license="CC-BY-4.0",
    )
)
dataset.add_record(record=record)
dataset.add_record(record=Record(sources=(accelerometer,)))
```

```text
TimeFValidationError: Sources ['Accelerometer'] already belong to another
record; each Source has one owner
```

## Storage and read-back

The writer stores each Source as one row of the `sources` table in `control.duckdb`. The row has
`source_id`, `record_id`, `parent_source_id` (`NULL` for a root), `name`, and `metadata`. The
writer assigns `source_id` again on each write. This number is internal to storage. Each `signals`
row and each Source annotation occurrence refers to it. The writer rejects a Record without
Sources, because a Record stores its Signals under Sources. See
[On-disk format](../timef-format.md).

When you read a version, TimeNet builds new Source objects from these rows. The objects have their
annotations and metadata. Signal values stay lazy until you read them.

```python
version = dataset.write(path="./registry")
(stored,) = TimeFDataset.open(path=version).records
(device,) = stored.sources
print(device.name, [a.key for a in device.annotations])
print(device.sources[0].metadata)
```

```text
Gearbox ['firmware', 'fault']
{'model': 'ADXL355'}
```

## Validation errors

| Cause | Raised by | Error |
| --- | --- | --- |
| Empty `name` | `Source(...)` | `TimeFValidationError` |
| Source under two parents, or a cycle | `walk_sources()`, `Record(...)` | `TimeFValidationError` |
| Source or Signal already in another Record | `add_record()` | `TimeFValidationError` |
| Signal attached twice in one Record | `add_record()` | `TimeFValidationError` |
| Timed annotation with no timed Signals below | `annotate()` | `TimeFValidationError` |
| Span with `time_series` set | `Annotation(...)` | `TimeFValidationError` |
| Span outside the windows of the subtree | `annotate()` | `SpanOutsideWindowWarning` |
| Unknown, ambiguous, or duplicate selection | `select()` | `TimeFValidationError` |
| Record with no Sources | `write()` | `TimeFValidationError` |
