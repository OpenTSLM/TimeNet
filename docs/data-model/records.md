---
icon: lucide/activity
description: "A Record groups Sources, Signals, annotations, and tasks for one observation."
tags:
  - guide
  - concepts
---

# Records

A `Record` is one observation unit. It can represent a patient visit, a machine run, or a market
window.

Each Record owns a hierarchy of Sources and Signals:

```mermaid
flowchart TB
    R["Record"]
    D["Source: device"]
    S1["Source: sensor"]
    S2["Source: sensor"]
    A["Signal"]
    B["Signal"]
    R --> D
    D --> S1
    D --> S2
    S1 --> A
    S2 --> B
```

A Source represents a device, sensor, or subsystem. It can contain child Sources and direct Signals.
A Signal is always a leaf.

## Create a Record

Build the hierarchy from the leaves upward:

```python
from timenet.dataset import Record, Source

accelerometer = Source(
    id="accelerometer",
    name="Accelerometer",
    signals=(acceleration_x, acceleration_y, acceleration_z),
)
imu = Source(
    id="imu",
    name="IMU",
    sources=(accelerometer,),
)
record = Record(
    record_id="run-001",
    sources=(imu,),
    metadata={"site": "lab-a"},
)
```

`record.signals` returns every Signal in deterministic hierarchy order. Use `walk_sources()` or
`walk_signals()` when you need an iterator.

Each Source and Signal has one owner. TimeNet rejects cycles, duplicate IDs, and objects attached in
more than one place.

## Store context

Use `metadata` for JSON-compatible details that do not need a typed schema. Examples include an
internal site code or source-specific import data.

Use an [Annotation](annotations.md) for a typed, searchable fact. Examples include age, condition,
firmware version, or an event on the timeline.

```python
from timenet.types import Annotation

record.annotate(Annotation(key="condition", value="healthy"))
```

Sources and Signals also support `metadata` and `annotate()`.

## Add time information

`start_time` anchors the relative recording timeline to Unix time. Pass a timezone-aware `datetime`
or whole Unix microseconds.

```python
from datetime import datetime, timezone

record = Record(
    record_id="run-001",
    sources=(imu,),
    start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
```

Leave `start_time=None` when the source has no reliable wall-clock time.

`time_span` can declare a session interval that contains all Signal windows. This field is useful
when every sensor is off during part of a session.

## Connect tasks

A [Task](tasks.md) refers directly to its input Records. Registering a Task adds its ID to each input
Record.

```python
from timenet.types import ClassificationTask

task = ClassificationTask(
    inputs=(record,),
    targets=("healthy",),
)
dataset.add_task(task=task)
```

Use `dataset.tasks_for(record)` to resolve the Task objects for one Record.
