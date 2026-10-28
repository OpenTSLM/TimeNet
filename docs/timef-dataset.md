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

A child dataset can copy objects from a parent release with `import_from()`. The child must declare
the parent in `metadata.parents`, under a parent alias. A parent alias is the name that the child
gives the parent.

```python
from timenet.reader import TimeFReader

with TimeFReader.open(parent_path) as parent:
    imported = child.import_from(parent, records=[12, 13])
copy = imported.records[12]
```

The unit of import is an object with an ID. Records, Signals, and tasks have IDs. An import makes
sure that each selected object, and each object that it needs, exists in the child. It reuses a copy
that the child holds, as it is, with the edits of the child. It copies an object that the child does
not hold under a new ID. This includes an object that the child removed. Sources and annotations
have no IDs. They come only with the object that carries them. An import never adds, restores, or
changes them later.

### The parent

Pass a reader of the parent release. The child must declare that exact release. If several aliases
declare it, pass `alias=` to choose one. The child imports a release under one alias only, and an
alias imports one release only.

Each import records the parent in the child:

- `parent_cards` holds a copy of the card of each parent, by parent alias.
- `lineage` holds the card of each ancestor release, the parents included.
- The child takes the access requirement and the `access_url` of the parent when the parent
  requires stricter access. A parent that is not stricter keeps its own in its card.

### Records

`records=` copies each Record whole: its Sources, its Signals, and the annotations on each of them.
Each copy is a Record of the child. The copy and each of its Signals get the next ID of their kind in
the child, as `add_record()` gives one. The copy has its own metadata and its own annotations, so an
edit of the copy does not change the parent. Annotations on the parent dataset itself are not
copied. An import that selects a Record whose copy the child holds reuses that copy. It adds only
what the copy lacks, as [Later imports](#later-imports) describes.

A copy shares the frozen time origins and axes of the parent. So copies of Records that share a time
origin in the parent also share it in the child. Each copied Signal keeps the value handles of the
parent, and reads the stored values of the parent unless the import loaded them. A write of the
child stores no values for the copies. The writer reuses the payloads of the parent instead, so a
written child needs the parent release.

Each call reads the selected Records, and the Records that hold the selected Signals, from the
parent in one batch. Select many objects in one call, not one object per call. By default, the
import loads no stored values. Pass `load=` to load them during the import, as
`TimeFReader.get_many()` does. With `load=LoadPlan.ALL_EAGER`, the copies share the loaded values,
and you can read them after the parent reader closes.

If the copy of one Record fails, the child does not change for that Record. The Records before it
in the call stay copied, and the parent stays recorded. Imports are safe to repeat. If a call fails
partway, fix the cause and repeat the call to complete it.

### Signals

`signals=` copies each selected Signal with its annotations. The Signal goes into a copy of its
Record and of each Source above it. Such a partial copy holds only the Signals that an import
selected or needed. The window of a Signal is the time range that its values cover.

A partial copy keeps the annotations of its Record and its Sources that its Signals place. The owner
of an annotation is the Record or Source that carries it. A copy places all annotations of an owner
when it holds every Signal below that owner. Otherwise it places
an annotation that has no span, or whose span lies inside the windows of the Signals that it holds
below the owner. For a Record with a `time_span`, that session span replaces the windows. So a
partial copy of such a Record keeps every Record annotation inside the session.

### Tasks

`tasks=` copies each selected task with the objects that it references:

- Its input Records, its candidate Records, and its Record targets come whole, as `records=` copies
  them.
- Each Signal target comes in a partial copy of its Record, as `signals=` copies it.
- The tasks that it derives from through `from_tasks` come too, with the objects that they
  reference. So do the tasks that those tasks derive from, and so on.

Each task copy references the copies in the child, not the objects of the parent. It holds copies of
the annotations of the task itself. It gets the next task ID of the child, and the import map links
it to its parent ID. `imported.tasks` holds the copy of each selected task, by its parent ID.

An import reuses a task copy that the child holds as it is, with the edits of the child. It does not
read that task from the parent again. If an edit removed some references of the copy, they stay
removed. A new task copy that derives from a held copy references the held copy.

The import refuses a task that references annotations in `input_annotations` or
`target_annotations`, and raises `TimeFValidationError`. It also refuses a `TSCorrespondenceTask`
without `candidate_records`, because the candidates of such a task are all Records of the parent.
`tasks=` also raises `TimeFValidationError` for a child that streams its tasks.

Each call reads the selected tasks, the tasks that they derive from, and the Records that they
reference in one batch, before the other Records. `load=` applies to this read too. The task copies
register after every Record, all together or not at all. If they fail, the Records that the call
copied stay, and the child holds no task copy of the call.

### Later imports

A later import reuses each copy that the child holds and grows it in place. If the call selects or
needs a Signal that a held copy lacks, the import adds the Signal to that copy. It also adds the
Sources above the Signal that the copy lacks. It attaches the annotations that the copy places now
and did not place before the call. The other annotations and the edits of the copy stay as they
are. So an import of a whole Record into a partial copy completes the copy. If the growth of a copy
fails, the import restores the copy as it was.

A Signal that an import adds to a held copy goes below the copy of its Source. The import finds the
copy of the Source through the Signals that it imported below that Source before. If there are
none, it finds the copy by the name of the Source. If sibling Sources share that name, in the parent
or in the copy, the import cannot tell them apart and raises `TimeFValidationError`. If the child
restructured the Sources of the copy, the name can find another Source, or the import raises.

Do not assign `Source.signals` or `Source.sources` on a copy yourself. This is not supported. A
later import can then attach an annotation twice.

### The import map

The child keeps a map from the parent alias, the object kind, and the parent ID to the ID of the
copy. The map is the only link between a copy and its parent object. Nothing derives the ID of a
copy from its ID in the parent. The writer stores the alias of each parent in the `imports` table.
It also stores an entry in the `imported_objects` table for each copy that the child holds. A
reader of the written child then finds a copy by its parent ID:

```python
from timenet.types.reference_keys import ParentAlias

with TimeFReader.open(child_path) as reader:
    copy = reader.get(Record, 12, parent=ParentAlias("ptbxl"))
```

`read()` of a written child restores the import state. It takes the parent cards and the lineage
from the manifest, and the import map from the catalog. A later import into the dataset that
`read()` returns reuses the copies that it holds, as an import into the child in memory does.

A removed object comes back when a later import selects it or needs it, under a new ID. So remove
imported objects after your last import.
