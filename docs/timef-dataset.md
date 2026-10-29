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
Task ──N:M── Record ──1:N── Source ──1:N── Signal
                         └──1:N── Source
Signal ──N:1── TimeAxis
```

`Source` is recursive. `Signal` is the leaf that owns the values, a `TimeAxis`, and a
`TimeSeriesSpec`. Several Signals can reference the same immutable axis or specification object.

Records, Signals, and Tasks have a read-only, positive int64 `id`. The dataset sets it when you add
the object. Sources, axes, and annotations have no ID. The dataset ID is the exception: it stays the
`org/name` string.

IDs carry no meaning. Keep upstream identifiers, such as a patient or ECG number, in `metadata` or
in an annotation. To refer to another object, use the object itself. See [IDs](types.md#ids) for
the rules.

## Signal

A `Signal` is one named stream. Pass `data` to the constructor for eager values. Use
`Signal.from_loader()` with `loader` and `n_values` for lazy values.

```python
from timenet.dataset import Signal
from timenet.dataset.axis import RegularAxis

axis = RegularAxis.from_rate_hz(500)

lead_i = Signal(
    name="I",
    data=lead_i_values,
    time_axis=axis,
    spec=ecg_spec,
)
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | `int` | Read-only identity. Set when the Record joins a dataset. |
| `name` | `str` | Human-readable name within the Source. |
| `spec` | `TimeSeriesSpec` | Unit, dtype, shape, and modality contract. |
| `time_axis` | `TimeAxis` | Regular, irregular, or ordinal placement of values. |
| `data` | array-like | Eager values. Mutually exclusive with `loader`. |
| `loader` | `ArrayHandle` or callable | Lazy values, passed to `Signal.from_loader()`. |
| `n_values` | `int` | Required for a lazy Signal and inferred from eager data. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Signal. |
| `metadata` | `dict[str, object]` | Optional JSON-compatible metadata. |

Read values through `to_arrow()`, `to_numpy()`, and `read_steps()`. These accessors apply the
spec: `to_numpy()` rejects nulls, and tensor values keep their trailing shape.

A Signal keeps its values behind an internal `ArrayHandle` from `timenet.types`, which library code
such as the writer reads directly. The constructor holds eager data in memory. Numeric and boolean
NumPy values stay NumPy, so `to_numpy()` builds no Arrow, and the first Arrow read converts them.
Other values become an Arrow array. `from_loader()` keeps a handle as it is and calls a plain
callable on every read. A loader that reads a step range on its own subclasses `ArrayHandle` and
overrides `read_steps()`; `from_loader()` rejects a plain callable that has its own `read_steps()`.
Two handles over the same function object are equal, so the writer reads time offsets shared
through one function once. The TimeF reader supplies a stored handle, so reading the hierarchy
does not load the arrays.

An irregular Signal also needs its time offsets. Pass `time_offsets_us` to
`Signal.from_irregular()` or `time_offsets_loader` to `Signal.from_loader()`, and read them with
`time_offsets_us()`. Regular axes calculate their offsets from their cadence, and ordinal axes
describe order without a clock.

Call `signal.annotate(annotation)` to attach one occurrence to a Signal.

## Source

A `Source` represents a device, sensor, or subsystem. It can contain both direct Signals and child
Sources:

```python
from timenet.dataset import Source

accelerometer = Source(
    name="Accelerometer",
    signals=(acceleration_x, acceleration_y, acceleration_z),
)

imu = Source(
    name="IMU",
    sources=(accelerometer, gyroscope, magnetometer),
)
```

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Human-readable Source name. |
| `sources` | `tuple[Source, ...]` | Direct child Sources. |
| `signals` | `tuple[Signal, ...]` | Signals produced directly by this Source. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Source. |
| `metadata` | `dict[str, object]` | Optional device, sensor, or subsystem metadata. |

`walk_sources()` and `walk_signals()` traverse a complete subtree. TimeNet rejects cycles, a Source
with two parents, and a Signal with two owners.

Sources group Signals. All timed Signals in a Record use its common relative origin.

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
    sources=(imu,),
    metadata={"site": "lab-a", "session": "record-123"},
)
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | `int` | Read-only identity. Set when the Record joins a dataset. |
| `sources` | `tuple[Source, ...]` | Root Sources in the recording hierarchy. |
| `annotations` | `tuple[Annotation, ...]` | Annotation occurrences attached to this Record. |
| `metadata` | `dict[str, object]` | Optional session metadata. |
| `start_time` | `TimeOrigin` | Common relative zero with an optional absolute timestamp. |
| `time_span` | `TimeInterval \| None` | Optional overall session interval. |

`record.signals` flattens every Signal in the hierarchy. `walk_sources()` and `walk_signals()` expose
the same deterministic traversal used by the writer.

Call `record.annotate(annotation)` to attach an occurrence. `time_point()` and `time_interval()`
convert wall-clock values against the Record origin. They require a known absolute timestamp.

Store subject identifiers as a list-valued record annotation. Omit it for a record with no subjects:

```python
from timenet.types import Annotation

record.annotate(
    Annotation(key="subject_ids", value=["patient-1", "patient-2"])
)
```

## Task

`Task` is the base class of the concrete tasks and cannot be built itself: each concrete class
sets its `task_type`. A concrete task such as `AnswerTask` or `ClassificationTask` refers directly
to one or more input Records:

```python
from timenet.types import AnswerTask

task = AnswerTask(
    inputs=(record,),
    prompt="Diagnose this patient.",
    targets=("The recording shows no cardiac activity.",),
)
```

The Task-to-Record relationship is many-to-many. One Task can compare several Records, and one
Record can appear in many Tasks. The ordered `targets` tuple accepts text, numbers, booleans,
Records, Signals, and spans. `targets=None` means that `target_annotations` supplies the answer. An
empty tuple is an explicit empty answer.

Tasks can also hold input or target Annotation references, references to parent Tasks, task-level
annotations, and optional metadata. The DuckDB storage layer turns these object references into
normalized ID relationships and restores the objects when it reads the dataset.

A Task may name its `split`: `train`, `validation`, or `test`. `get_train()`, `get_validation()`,
and `get_test()` return the Tasks of one split. `get_splits(Split.TRAIN, Split.VALIDATION)` selects
their union in dataset order without changing assignments. `get_all()` returns every Task;
`iter_tasks(split=...)` filters a stream by one split or an iterable of splits.
See [Tasks](data-model/tasks.md#splits).

## Building a dataset

Register complete Records independently of Tasks. This supports datasets with or without labels:

```python
from timenet.dataset import TimeFDataset

dataset = TimeFDataset(metadata=metadata)
dataset.add_record(record=record)
dataset.add_task(task=task)
```

`add_record()` validates hierarchy ownership and ID uniqueness. `add_task()` validates referenced
objects, the answer representation, and temporal scopes. A validation error leaves the dataset
unchanged.

For a large corpus, `set_task_stream()` accepts a re-iterable task source. The writer validates and
inserts that stream once without retaining every Task in memory.

## Annotations

The same `Annotation` type can annotate a Dataset, Task, Record, Source, or Signal. Annotation
content is immutable and reusable. Every call to `annotate()` creates a separate occurrence, so a
long payload can be stored once while its occurrences carry independent placement, confidence, and
provenance. `Annotation` is final: reuse content through a constant, or a builder function for
content that varies, as [Annotations](data-model/annotations.md#reuse-content) shows.

```python
from timenet.types import Annotation, TimePoint

male = Annotation(key="patient_sex", value="male")
record.annotate(male)

lead_i.annotate(
    Annotation(
        key="lead_status",
        value="Lead fell off",
        span=TimePoint.seconds(6),
    )
)
```

## Schema and persistence

`derive_schema()` collects the time-series specifications, annotation descriptors, and concrete Task
types found in the dataset. It does not load Signal values.

Write a version with the declarative helper:

```python
from timenet.values_backends import ValuesBackend

version_path = dataset.write(
    path="./registry",
    values_backend=ValuesBackend.PARQUET,
)
```

Open a local version with [`TimeFReader.open()`](timef-reader.md). The reader's `read()` returns
the dataset with lazy Signal values:

```python
from timenet.reader import TimeFReader

with TimeFReader.open(version_path) as reader:
    dataset = reader.read()
```

The stored version contains `manifest.json`, an immutable `control.duckdb`, and a Parquet or Zarr
values plane. See [TimeF format](timef-format.md) and [TimeFWriter](timef-writer.md) for the physical
layout.

See the [API reference for `timenet.dataset`](api/dataset.md) for the full symbol listing.

## Importing from a parent

A child dataset copies objects from a parent release with `import_from()`. A parent release is one
version of another dataset. The child must declare the parent in `metadata.parents`, under a parent
alias. A parent alias is the name that the child gives the parent.

```python
from timenet.reader import TimeFReader

with TimeFReader.open(parent_path) as parent:
    imported = child.import_from(parent, records=[12, 13])
copy = imported.records[12]
```

The unit of import is an object with an ID. Records, Signals, and tasks have IDs. An import makes
sure that each selected object, and each object that it needs, is in the child. If the child holds a
copy of the object, the import uses that copy as it is, with the edits of the child. Otherwise, the
import copies the object under a new ID. This includes an object that the child removed.

Sources and annotations have no IDs. They come only with the object that holds them. An import never
adds, restores, or changes them later.

### The parent

Pass a reader of the parent release. The child must declare that exact release. If several aliases
declare it, pass `alias=` to choose one. The child imports a release under one alias only, and an
alias imports one release only.

Each import records the parent in the child. `parent_cards` holds a copy of the card of each
parent, by parent alias. `lineage` holds the card of each ancestor release. An ancestor release is a
parent, a parent of a parent, and so on.

An import does not change the card of the child. Each card keeps the `access` level and the
`access_url` that its author declared. A parent can be credentialed, and the card of its child can
still say `open`. The effective access level of a dataset is the strictest level of its own card and
of the cards in its `lineage`. A user must have this level of access to use the data of the dataset.

`effective_access(metadata, lineage)` in `timenet.types.access` returns the effective access level.
`access_requirements(metadata, lineage)` returns each card whose `access` is not `open`, the card of
the dataset first. Each item names the release, its access level, and its `access_url`. The command
`timenet info` shows the effective access level and this list.

### Records

`records=` copies each Record whole: its Sources, its Signals, and the annotations on each of them.
Each copy is a Record of the child. The copy and each of its Signals get the next ID of their kind in
the child, as `add_record()` gives one. The copy has its own metadata and its own annotations, so an
edit of the copy does not change the parent.

`records=` does not copy the annotations of the parent dataset itself. If the child holds a copy of
a selected Record, the import returns that copy as it is.

A copy shares the time origins and the time axes of the parent. These objects are frozen, so no
copy can change them. Copies of Records that share a time origin in the parent thus share it in the
child too.

Each copied Signal keeps the value handles of the parent. A value handle points to values that a
release stores. So a copied Signal reads the stored values of the parent, unless the import loaded
them. A write of the child stores no values for the copies. The writer uses the payload of the
parent instead, which is the set of files that hold the values of the parent. So a written child
needs the parent release.

Each call reads the selected Records from the parent in one batch. Select many Records in one call,
not one Record per call. By default, the import loads no stored values. To load them during the
import, pass `load=`, as for `TimeFReader.get_many()`. With `load=LoadPlan.ALL_EAGER`, the copies
share the loaded values. You can then read them after the parent reader closes.

If the copy of one Record fails, the child does not change for that Record. The Records before it
in the call stay copied, and the parent stays recorded. You can repeat an import without harm. If a
call fails partway, fix the cause and repeat the call to complete it.

### The import map

The import map of the child links each copy to its parent object. The key of an entry is the
parent alias, the object kind, and the parent ID. The value is the ID of the copy. The map is the
only link between a copy and its parent object. No rule derives the ID of a copy from its ID in the
parent.

The writer stores the alias of each parent in the `imports` table. It also stores an entry in the
`imported_objects` table for each copy that the child holds. A reader of the written child then
finds a copy by its parent ID:

```python
from timenet.types.reference_keys import ParentAlias

with TimeFReader.open(child_path) as reader:
    copy = reader.get(Record, 12, parent=ParentAlias("ptbxl"))
```

The import state of a dataset is its parent cards, its lineage, and its import map. `read()` of a
written child restores the import state. It takes the parent cards and the lineage from the
manifest, and the import map from the catalog. A later import into the dataset that `read()`
returns uses the copies that the dataset holds. An import into the child in memory does the same.

If a later import selects or needs a removed object, the import copies the object again, under a
new ID. So remove imported objects after your last import.
