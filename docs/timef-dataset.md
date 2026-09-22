---
icon: lucide/table-2
description: "The in-memory TimeFDataset hierarchy: Tasks, Records, Sources, and Signals."
tags:
  - reference
  - dataset
---

# TimeFDataset

`TimeFDataset` is the in-memory TimeF model. A connector builds ordinary Python objects, registers
them with the dataset, and writes one immutable version. The hierarchy is:

```
Task ──N:M── Record ──1:N── Source ──1:N── Signal ──N:1── TimeAxis
                                  └──1:N── Source
```

`Source` is recursive. `Signal` is the leaf that owns the values, a `TimeAxis`, and a
`TimeSeriesSpec`. Several Signals can reference the same immutable axis or specification object.

## Signal

A `Signal` is one named, one-dimensional stream. Give it eager `data`, or a lazy `loader` and an
explicit `n_values`:

```python
from timenet.dataset import Signal
from timenet.dataset.axis import RegularAxis

axis = RegularAxis.from_rate_hz(500)

lead_i = Signal(
    id="lead-i",
    name="I",
    data=lead_i_values,
    time_axis=axis,
    spec=ecg_spec,
)
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Stable identity. The default is a UUIDv7. |
| `name` | `str` | Human-readable name within the Source. |
| `spec` | `TimeSeriesSpec` | Unit, dtype, shape, and modality contract. |
| `time_axis` | `TimeAxis` | Regular, irregular, or ordinal placement of values. |
| `data` | array-like | Eager values. Mutually exclusive with `loader`. |
| `loader` | callable | Lazy Arrow value loader. Mutually exclusive with `data`. |
| `n_values` | `int` | Required with a lazy loader and inferred from eager data. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Signal. |
| `metadata` | `dict[str, object]` | Optional JSON-compatible metadata. |

`to_arrow()`, `to_numpy()`, and `read_steps()` materialize values. The TimeF reader supplies a lazy
storage loader, so reading the hierarchy does not load the arrays.

Irregular Signals also provide `time_offsets_loader`. Regular axes calculate their offsets from
their cadence, and ordinal axes describe order without a clock.

Call `signal.annotate(annotation)` to attach one occurrence to a Signal.

## Source

A `Source` represents a device, sensor, or subsystem. It can contain both direct Signals and child
Sources:

```python
from timenet.dataset import Source

accelerometer = Source(
    id="accelerometer",
    name="Accelerometer",
    signals=(acceleration_x, acceleration_y, acceleration_z),
)

imu = Source(
    id="imu",
    name="IMU",
    sources=(accelerometer, gyroscope, magnetometer),
)
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Stable identity. The default is a UUIDv7. |
| `name` | `str` | Human-readable Source name. |
| `sources` | `tuple[Source, ...]` | Direct child Sources. |
| `signals` | `tuple[Signal, ...]` | Signals produced directly by this Source. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Source. |
| `metadata` | `dict[str, object]` | Optional device, sensor, or subsystem metadata. |

`walk_sources()` and `walk_signals()` traverse a complete subtree. TimeNet rejects cycles, a Source
with two parents, and a Signal with two owners.

Call `source.annotate(annotation)` for a Source annotation. A selection attaches annotation content
to several Signals in one operation:

```python
source.select(signals=(lead_i, lead_ii)).annotate(annotation)
source.select(signal_names=("I", "II")).annotate(annotation)
```

Object selection is the safer form. Name selection is useful when only names are available and
fails if a name is absent or ambiguous.

## Record

A `Record` is one recording session. It owns one or more root Sources and can carry session-level
annotations and metadata:

```python
from timenet.dataset import Record

record = Record(
    record_id="record-123",
    sources=(imu,),
    metadata={"site": "lab-a"},
)
```

| Field | Type | Description |
| --- | --- | --- |
| `record_id` | `str` | Stable identity. `record.id` is the public alias. |
| `sources` | `tuple[Source, ...]` | Root Sources in the recording hierarchy. |
| `subject_ids` | `tuple[str, ...]` | Optional subject identifiers. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Record. |
| `metadata` | `dict[str, object]` | Optional session metadata. |
| `start_time` | `datetime \| int \| None` | Optional wall-clock anchor in Unix microseconds. |
| `time_span` | `TimeInterval \| None` | Optional overall session interval. |

`record.signals` flattens every Signal in the hierarchy. `walk_sources()` and `walk_signals()` expose
the same deterministic traversal used by the writer.

Call `record.annotate(annotation)` to attach an occurrence. Temporal annotations are validated
against the relevant Signal windows. `time_point()` and `time_interval()` convert wall-clock values
against an anchored Record.

## Task

`Task` is abstract. A concrete task such as `AnswerTask` or `ClassificationTask` refers directly to
one or more input Records:

```python
from timenet.types import AnswerTask

task = AnswerTask(
    id="diagnosis-123",
    inputs=(record,),
    prompt="Diagnose this patient.",
    target="The recording shows no cardiac activity.",
)
```

The Task-to-Record relationship is many-to-many. One Task can compare several Records, and one
Record can appear in many Tasks. Concrete task classes define their own target fields; TimeNet does
not wrap targets in generic text or mixed-target containers.

Tasks can also hold input or target Annotation references, references to parent Tasks, task-level
annotations, and optional metadata. The DuckDB storage layer turns these object references into
normalized ID relationships and restores the objects when it reads the dataset.

## Building a dataset

Register complete Records independently of Tasks. This supports unlabelled datasets as well as
supervised ones:

```python
from timenet.dataset import TimeFDataset

dataset = TimeFDataset(metadata=metadata)
dataset.add_record(record=record)
dataset.add_task(task=task)
```

`add_record()` validates hierarchy ownership and ID uniqueness. `add_task()` requires at least one
registered input Record and validates object references, answer shape, and temporal scopes. A
validation error leaves the dataset unchanged.

For a large corpus, `set_task_stream()` accepts a re-iterable task source. The writer validates and
inserts that stream once without retaining every Task in memory.

## Annotations

The same `Annotation` type can annotate a Dataset, Task, Record, Source, or Signal. Annotation
content is immutable and reusable. Every call to `annotate()` creates a separate occurrence, so a
long payload can be stored once while its occurrences carry independent placement, confidence, and
provenance.

```python
from timenet.types import Annotation, TimePoint

male = Annotation.static(name="patient_sex", value="male")
record.annotate(male)

lead_i.annotate(
    Annotation.point(
        name="lead_status",
        value="Lead fell off",
        at=TimePoint.seconds(6),
    )
)
```

## Schema and persistence

`derive_schema()` collects the TimeSeries specifications, annotation descriptors, and concrete Task
types found in the dataset. It does not load Signal values.

Write a version with the declarative helper:

```python
from timenet.values_backends import ValuesBackend

version_path = dataset.write(
    path="./registry",
    values_backend=ValuesBackend.PARQUET,
)
```

Open a local version with lazy Signal values:

```python
dataset = TimeFDataset.open(path=version_path)
```

The stored version contains `manifest.json`, an immutable `control.duckdb`, and a Parquet or Zarr
values plane. See [TimeF format](timef-format.md) and [TimeFWriter](timef-writer.md) for the physical
layout.

See the [API reference for `timenet.dataset`](api/dataset.md) for the full symbol listing.
