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

Build the tree from the leaves upward. This example is one night of a sleep study. The
participant wears a Polar H10 chest strap, which records ECG at 130 Hz. They also wear an
Empatica EmbracePlus watch. Its accelerometer records three axes at 64 Hz.

```python
import numpy as np

from timenet.dataset import Record, RegularAxis, Signal, Source
from timenet.types import TimeSeriesSpec, ureg

# Stand-in for 60 s of a device export. A connector reads these values.
rng = np.random.default_rng(17)
ecg_values = rng.normal(0.0, 0.2, 60 * 130).astype(np.float32)
wrist_values = rng.normal(0.0, 0.05, (3, 60 * 64)).astype(np.float32)

ecg = TimeSeriesSpec(
    spec_type="ecg",
    name="ECG",
    unit_value=ureg.millivolt,
    dtype="float32",
)
acceleration = TimeSeriesSpec(
    spec_type="acceleration",
    name="Acceleration",
    unit_value=ureg.standard_gravity,
    dtype="float32",
)

chest_strap = Source(
    "Chest strap",
    signals=(
        Signal(
            name="ECG",
            spec=ecg,
            time_axis=RegularAxis.from_rate_hz(130),
            data=ecg_values,
        ),
    ),
    metadata={"model": "Polar H10"},
)
accelerometer = Source(
    "Accelerometer",
    signals=[
        Signal(
            name=axis,
            spec=acceleration,
            time_axis=RegularAxis.from_rate_hz(64),
            data=values,
        )
        for axis, values in zip("xyz", wrist_values)
    ],
)
watch = Source(
    "Wrist watch",
    sources=(accelerometer,),
    metadata={"model": "Empatica EmbracePlus"},
)
record = Record(
    sources=(chest_strap, watch),
    metadata={"participant": "P017", "session": "night-1"},
)

print([source.name for source in record.walk_sources()])
print([signal.name for signal in record.signals])
```

```text
['Chest strap', 'Wrist watch', 'Accelerometer']
['ECG', 'x', 'y', 'z']
```

The device model goes in the `metadata` of its Source. The participant and the session go in the
`metadata` of the Record.

`name` can be positional, as in `Source("IMU", signals=[...])`. All other fields are keyword-only. The
collection fields accept any iterable, such as a list, and store a tuple. A single item or a string
raises `TimeFValidationError`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Human-readable name. It must not be empty. |
| `sources` | `Iterable[Source]`, stored as a tuple | `()` | Direct child Sources. Their order has no meaning. |
| `signals` | `Iterable[Signal]`, stored as a tuple | `()` | Signals that this Source produces directly. |
| `annotations` | `Iterable[Annotation]`, stored as a tuple | `()` | Annotation occurrences on this Source. |
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
describes the whole device, for example its firmware version. A timed annotation covers an
interval, for example a time when an electrode lost contact with the skin:

```python
from timenet.types import Annotation, TimeInterval

chest_strap.annotate(Annotation("firmware", value="3.1.1"))
chest_strap.annotate(
    Annotation(
        key="electrode_contact",
        value="lost",
        span=TimeInterval.seconds(12.0, 15.5),
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
Signal belongs to one Source. If the participant wears the same chest strap on the next night,
build a new Source for the new Record.

`walk_sources()` rejects a Source under two parents, and it rejects a cycle in the tree. When you
construct a Record, it does the same checks. `add_record()` rejects a Source or Signal that
another Record already owns. It also rejects a Signal that one Record attaches twice.

```python
from timenet.dataset import TimeFDataset
from timenet.types import DatasetMetadata

dataset = TimeFDataset(
    metadata=DatasetMetadata(
        dataset_id="demo/sleep-wearables",
        dataset_version="1.0.0",
        name="Sleep wearables demo",
        description="One night of one participant.",
        license="CC-BY-4.0",
    )
)
dataset.add_record(record)
night_2 = Record(
    sources=(chest_strap,),
    metadata={"participant": "P017", "session": "night-2"},
)
dataset.add_record(night_2)
```

```text
TimeFValidationError: Sources ['Chest strap'] already belong to another
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
from timenet.reader import TimeFReader

version = dataset.write(path="./registry")
with TimeFReader.open(version) as reader:
    (stored,) = reader.read().records
strap, stored_watch = stored.sources
print(strap.name, [a.key for a in strap.annotations])
print(stored_watch.metadata, stored_watch.sources[0].name)
```

```text
Chest strap ['firmware', 'electrode_contact']
{'model': 'Empatica EmbracePlus'} Accelerometer
```
