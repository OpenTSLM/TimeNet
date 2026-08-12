---
icon: lucide/table-2
description: "The in-memory TimeFDataset model: samples, tasks, and time series."
tags:
  - reference
  - dataset
---

# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their tasks as Python
objects. No I/O: persistence is the [`TimeFWriter`](timef-writer.md) concern. Lives in `timenet.dataset`.

---

## TimeSeries

Reference to one logical stream of time-series data, with optional windowing and a lazy Arrow loader.

```python
from timenet.dataset import TimeSeries
from timenet.dataset.axis import RegularAxis

TimeSeries(
    spec=vibration,           # a TimeSeriesSpec (the modality)
    channel="axial",          # the channel this series carries
    time_axis=RegularAxis.from_rate_hz(500),
    # Callable[[], pa.Array] matching the spec's dtype and value_shape
    loader=load_axial,
    source_id="rec_001",      # optional
    n_values=5000,            # how many values the loader will return
)
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `spec` | `TimeSeriesSpec` | yes | The modality (shared across channels). |
| `channel` | `str` | yes | The logical stream name (e.g. `"axial"` or `"rgb_frames"`). |
| `time_axis` | `TimeAxis` | yes | Where the values sit in time: `RegularAxis`, `IrregularAxis`, or `OrdinalAxis`. |
| `n_values` | `int` | yes | How many values the series holds; the writer checks the loader against it. |
| `loader` | `Callable[[], pa.Array]` | yes | Lazy loader returning scalar values or an Arrow fixed-shape tensor array. |
| `source_id` | `str \| None` | no | Identifier of the raw recording this series came from. |
| `time_series_id` | `str` | no | Persistent handle (auto uuid7). The writer dedupes by it. |
| `time_offsets_loader` | `Callable[[], pa.Array] \| None` | no | One int64 microsecond time offset per value; required for `IrregularAxis` and rejected otherwise. |

`TimeSeries` is frozen with identity equality (`eq=False`): the writer dedupes by `time_series_id`, so
reusing one instance (or giving two instances the same explicit id) collapses to one chunk on disk.
Consumers read values through `to_arrow()` (Arrow, zero-copy) or `to_numpy()`; `loader` is plumbing
supplied by the connector at curation and by [`TimeFReader`](timef-reader.md) on read-back, and remains
the lazy boundary. `read_steps(start, stop)` lets range-aware storage loaders select a
temporal subsection before returning Arrow; older connector callables fall back to a full-read slice.

`TimeSeriesSpec.dtype`, `value_shape`, and `dimension_names` describe one timestep. Scalar series keep
the defaults `float32`, `()`, and `()`. For an RGB camera, for example, use `dtype="uint8"`,
`value_shape=(height, width, 3)`, and names `("height", "width", "color")`; the full logical shape is
always `(n_steps, *value_shape)`.

When a connector already holds the values in memory, use the classmethod
`TimeSeries.from_values(values, *, spec, channel, time_axis, source_id=None, time_series_id=None)`: it
wraps them in a `float32` loader and takes `n_values` from the array's own length. For a series whose
time offsets are stored rather than computed, use `TimeSeries.from_irregular(values, *, time_offsets_us, ...)`,
which derives the axis from the stream so the two cannot disagree. Reach for the `loader=` constructor
above only for genuinely lazy sources (files, remote shards).

An **ordinal** series has positions but no clock — an ordered sequence like `TSQA`, whose values carry
an order but no calendar time. Build it with an `OrdinalAxis`: there is no rate and no timeline, so a
task on it is scoped and forecast in steps rather than seconds (see [tasks](data-model/tasks.md)).

```python
from timenet.dataset import TimeSeries
from timenet.dataset.axis import OrdinalAxis

# TSQA: an ordered sequence of values with no wall clock
series = TimeSeries.from_values(
    values,                       # the ordered values
    spec=tsqa_spec,
    channel="series",
    time_axis=OrdinalAxis(),
    time_series_id="tsqa",
)
```

---

## Sample

One logical unit of time-series data: a recording, a session, a sensor bundle, a market window. Created
via `TimeFDataset.add_sample`.

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Auto uuid7 (or explicit, for deterministic output). |
| `time_series` | `tuple[TimeSeries, ...]` | The sample's logical streams. |
| `subject_ids` | `tuple[str, ...]` | Subjects (empty for subject-less domains). |
| `task_ids` | `tuple[str, ...]` | Ids of tasks attached via `add_task` (populated after construction). |
| `annotations` | `tuple[Annotation, ...]` | Attached via `add_annotation`. |
| `start_time` | `datetime \| int \| None` | Wall-clock anchor that relative time zero refers to, for every series and annotation on the sample. Pass a timezone-aware `datetime` or whole Unix microseconds; construction normalizes either one to microseconds. A bare float is refused, since seconds and microseconds are both plausible readings of it. `None` means no wall-clock reference exists (e.g. de-identified or synthetic data) — never fabricate one. |
| `time_span` | `TimeInterval \| None` | The session's overall span on the source recording timeline, for a recording whose series leave gaps an unscoped span may fall in (a note taken while every sensor was briefly off). Must carry no `time_series_ids` and must contain every series' window. When set, an unscoped span is checked against it instead of against the union of the series' windows. |

All series and annotations in an anchored sample share this clock and relative-time coordinate system.
Use `sample.has_absolute_time` to check whether the anchor is known.

`add_annotation(annotation)` attaches and returns it, validating a temporal annotation's span by the
same rule a task's `scope` uses. A span scoped to named `time_series_ids` must lie inside the
*intersection* of those series' windows. An unscoped span is checked against the sample's `time_span`
when it declares one, and otherwise against the *union* of the series' windows, so an event landing in
an unrecorded gap between series is rejected unless a `time_span` says the session spanned it.

`to_arrow()` / `to_numpy()` return the sole channel's 1-D values (Arrow / NumPy) for the common
single-channel sample, raising `ValueError` for a multi-channel sample (index `time_series` yourself
then).

---

## TimeFDataset

```python
from timenet.dataset import TimeFDataset

dataset = TimeFDataset(metadata=metadata)
sample = dataset.add_sample(time_series=(...), subject_ids=("p1",))
dataset.add_task(sample, ClassificationTask(target="faulty"))
dataset.derive_schema()
```

### `add_sample()`

```python
add_sample(
    *, time_series, subject_ids=(),
    sample_id=None, start_time=None, time_span=None,
) -> Sample
```

Creates a sample, registers it, returns it. Raises `TimeFValidationError` if `time_series` is empty.
Pass `sample_id` for deterministic output (e.g. golden fixtures), and `start_time` to anchor the
sample's relative timeline to wall-clock time so samples can be synchronized across datasets and
devices.

### `add_task()`

```python
add_task(samples, task, *, scope=None, from_tasks=()) -> Task
```

Registers a task and links it to its samples: populates `task.sample_ids` and appends `task.id` to each
sample's `task_ids`. `scope`, when passed, is stamped onto `task.scope` (equivalent to constructing the
task with it, and rejected if the task already has one). `from_tasks` overrides the task's own value only
when non-empty, so a task built with `from_tasks=` is never clobbered.

This is where a task is checked against the samples it is attached to, since this is the first point that
has both. Raises `ValueError` on empty `samples`; on a task that sets both `target` and
`target_annotation_ids`, or neither unless its answer is a produced series; on a
[`Span`](types.md#span) — the `scope` or a localization target — whose `time_series_ids` do not resolve on
every target sample or that falls outside a sample's covered span; and on an `input_annotation_ids` /
`target_annotation_ids` entry that no target sample carries.

### `derive_schema()`

```python
derive_schema() -> DatasetSchema
```

Walks the dataset's instances and builds its [`DatasetSchema`](types.md#datasetschema): the distinct
specs, annotation descriptors, and task types (order-preserving dedupe). Stores the result
(`dataset.schema`) and returns it. Never reads series values. The engine calls it after `convert()`,
before the writer runs.

### Properties

`metadata`, `samples` (tuple, read-only), `tasks` (tuple, read-only), and `schema`
(`DatasetSchema | None`, `None` until `derive_schema()` runs or the reader populates it).

### `tasks_of()` / `tasks_for()`

```python
tasks_of(task_type) -> tuple[Task, ...]
tasks_for(sample, task_type=Task) -> tuple[Task, ...]
```

`tasks_of` returns every task of a type across the dataset; `tasks_for` resolves one sample's `task_ids`
back to task objects, optionally filtered by type.

### `to_features_and_targets()`

```python
to_features_and_targets(*, task=None, output="arrow", features="timestep")
    -> tuple[pa.Array, pa.Array] | tuple[np.ndarray, np.ndarray]
```

Builds an `(X, y)` training pair, **deferring materialization by default**. `features` picks the shape of
`X`: `"timestep"` (default) gives one feature per point — a rectangular `FixedSizeListArray[T]` / `(n, T)`
matrix that needs equal-length samples; `"series"` gives one sequence per sample — a `ListArray` / `(n,)`
object array that also handles variable-length series. `output="arrow"` (default) builds these straight
from the loaders with no NumPy copy; `output="numpy"` materializes them. `task` is inferred when the
dataset has exactly one target-bearing type (pass it explicitly otherwise; `ForecastingTask` has no
scalar target).

### `describe()`

```python
describe(*, rows=5, file=None) -> None
```

Prints a plain-text summary, like pandas' `describe`/`info`: identity, counts, per-spec columns (name,
units, and the value dtype sampled from one series), and a preview of the first `rows` samples. The
preview reads only span metadata, so it never loads series values. Works before `derive_schema()` (all
figures are computed from the samples), needs no CLI or `rich` dependency, and writes to `file`
(default `sys.stdout`).

```python
TimeNet().load("chengsenwang/tsqa").describe()
```

```text
chengsenwang/tsqa @ 1.0.0
  name     TSQA
  license  Apache-2.0

counts
  samples      48000
  series       tsqa_series=48000
  annotations  48000
  tasks        answer=48000

specs
  spec         name         value          dtype
  tsqa_series  TSQA Series  dimensionless  float

samples (first 5 of 48000)
  sample_id  channels  length  tasks  annotations
  row-0      1         64      1      1
```

---

See the [API reference for `timenet.dataset`](api/dataset.md) for the full symbol listing.
