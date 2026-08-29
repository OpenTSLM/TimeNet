---
icon: lucide/table-2
description: "The in-memory TimeFDataset model: samples, tasks, and time series."
tags:
  - reference
  - dataset
---

# TimeFDataset

`TimeFDataset` is the in-memory model that a connector populates during `convert()`. It holds samples
and their tasks as Python objects. `TimeFDataset` does no I/O. The [`TimeFWriter`](timef-writer.md)
handles persistence. `TimeFDataset` lives in `timenet.dataset`.

---

## TimeSeries

A `TimeSeries` is a reference to one logical stream of time-series data. It supports optional
windowing and a lazy Arrow loader.

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
| `channel` | `str` | yes | The logical stream name (for example `"axial"` or `"rgb_frames"`). |
| `time_axis` | `TimeAxis` | yes | Where the values sit in time: `RegularAxis`, `IrregularAxis`, or `OrdinalAxis`. |
| `n_values` | `int` | yes | How many values the series holds. The writer validates the loader against this number. |
| `loader` | `Callable[[], pa.Array]` | yes | A lazy loader that returns scalar values or an Arrow fixed-shape tensor array. |
| `source_id` | `str \| None` | no | Identifier of the raw recording that this series came from. |
| `time_series_id` | `str` | no | Persistent handle (auto uuid7). The writer dedupes by it. |
| `time_offsets_loader` | `Callable[[], pa.Array] \| None` | no | One int64 microsecond time offset per value. `IrregularAxis` requires this field. Other axis types reject it. |

`TimeSeries` is frozen and uses identity equality (`eq=False`). The writer dedupes by
`time_series_id`. If a connector reuses one instance, or gives two instances the same explicit ID,
both collapse to one chunk on disk.
Consumers read values through `to_arrow()` (Arrow, zero-copy) or `to_numpy()`. The connector supplies
`loader` at build. [`TimeFReader`](timef-reader.md) supplies `loader` again on read-back. In both
cases, `loader` remains the lazy boundary. `read_steps(start, stop)` lets range-aware storage loaders
select a temporal subsection, then return it as Arrow. Older connector callables do not support this.
They fall back to a full-read slice.

`TimeSeriesSpec.dtype`, `value_shape`, and `dimension_names` describe one timestep. Scalar series keep
the defaults `float32`, `()`, and `()`. An RGB camera, for example, uses `dtype="uint8"`,
`value_shape=(height, width, 3)`, and names `("height", "width", "color")`. The full logical shape is
always `(n_steps, *value_shape)`.

If a connector already holds the values in memory, use the classmethod
`TimeSeries.from_values(values, *, spec, channel, time_axis, source_id=None, time_series_id=None)`.
This method wraps the values in a `float32` loader. It takes `n_values` from the length of the array.
If a series stores time offsets instead of computing them, use
`TimeSeries.from_irregular(values, *, time_offsets_us, ...)`. This method derives the axis from the
stream, so the two values cannot disagree. Use the `loader=` constructor above only for lazy sources
(files, remote shards).

An **ordinal** series has positions but no clock. It is an ordered sequence, for example `TSQA`, whose
values carry an order but no calendar time. Build it with an `OrdinalAxis`. An ordinal series has no
rate and no timeline. As a result, a task on this series uses steps instead of seconds for its scope
and forecast (see [tasks](data-model/tasks.md)).

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

A `Sample` is one logical unit of time-series data, for example a recording, a session, a sensor
bundle, or a market window. A connector creates it with `TimeFDataset.add_sample`.

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Auto uuid7 (or explicit, for deterministic output). |
| `time_series` | `tuple[TimeSeries, ...]` | The sample's logical streams. |
| `subject_ids` | `tuple[str, ...]` | Subjects (empty for subject-less domains). |
| `task_ids` | `tuple[str, ...]` | Ids of tasks attached via `add_task` (populated after construction). |
| `annotations` | `tuple[Annotation, ...]` | Attached via `add_annotation`. |
| `start_time` | `datetime \| int \| None` | The wall-clock anchor that relative time zero refers to, for every series and annotation in the sample. Pass a timezone-aware `datetime` or a whole number of Unix microseconds. Construction normalizes either value to microseconds. `Sample` rejects a bare float, because both seconds and microseconds are plausible readings of it. `None` means no wall-clock reference exists, for example for de-identified or synthetic data. If no wall-clock reference exists, do not invent one. |
| `time_span` | `TimeInterval \| None` | The overall span of the session on the source recording timeline. Some recordings have series with gaps between them. An unscoped span, for example a note taken while every sensor was briefly off, can fall inside such a gap. `time_span` must carry no `time_series_ids`. It must contain every series' window. When `time_span` is set, `Sample` validates an unscoped span against it instead of against the union of the series' windows. |

All series and annotations in an anchored sample share this clock and relative-time coordinate system.
Use `sample.has_absolute_time` to find out whether the anchor is known.

`add_annotation(annotation)` attaches the annotation and returns it. It validates a temporal
annotation's span with the same rule that a task's `scope` uses. A span scoped to named
`time_series_ids` must lie inside the *intersection* of those series' windows. If an unscoped span
declares a `time_span`, `add_annotation` validates the span against the sample's `time_span`.
Otherwise, `add_annotation` validates the span against the *union* of the series' windows. A span
that leaves the window this rule selects warns with `SpanOutsideWindowWarning` and is kept as it was
given. Some sources state a region that reaches past the signals it was written for, and a connector
records what the source says. Pass `warn_when_outside=False` to raise `TimeFValidationError` instead.
The reader passes that argument, so a stored span outside its window is still a format error.

`add_annotations([...])` attaches an iterable the same way, but as one all-or-nothing operation. It
validates the whole batch first. It leaves the sample untouched if any annotation fails.

`to_arrow()` and `to_numpy()` return the sole channel's 1-D values (Arrow or NumPy) for the common
single-channel sample. For a multi-channel sample, both methods raise `ValueError`. In that case, index
`time_series` directly.

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

`add_sample` creates a sample, registers it, and returns it. It raises `TimeFValidationError` if
`time_series` is empty. Pass `sample_id` for deterministic output, for example for golden fixtures.
Pass `start_time` to anchor the sample's relative timeline to wall-clock time. With this anchor, a
caller can synchronize samples across datasets and devices.

### `add_task()` / `add_tasks()`

```python
add_task(samples, task) -> Task
add_tasks(samples, tasks) -> tuple[Task, ...]
```

`add_task` and `add_tasks` register a task, or a batch of tasks, against the same sample or samples.
`add_task` populates each task's `sample_ids`. It appends each task's `id` to every target sample's
`task_ids`. Put `scope` and `from_tasks` on the task itself. These fields describe that one task, not
the call.

`add_task` validates a task against its samples at this point, because it is the first point that has
both a task and its samples. `add_task` raises `TimeFValidationError` in these cases:

- `samples` is empty.
- A task sets both `target` and `target_annotation_ids`, or sets neither, unless its answer is a
  produced series.
- A [`Span`](types.md#span) (the `scope` or a localization target) has `time_series_ids` that do not
  resolve on every target sample. A span that falls outside a sample's covered span warns with
  `SpanOutsideWindowWarning` and is kept, the same way an annotation's span is.
- An `input_annotation_ids` or `target_annotation_ids` entry names an annotation that no target sample
  carries.

`add_tasks` is all-or-nothing. It validates the whole batch before it attaches any task. If one task is
bad, `add_tasks` raises an error and leaves the dataset untouched. To keep tasks that were added before
a failure, call `add_task` in a loop instead.

Validation of the whole batch also enforces relationships across the batch:

- Task ids stay unique against the batch and the dataset.
- Every `from_tasks` parent already exists in the dataset, or the batch includes it.
- No task derives from itself or closes a cycle.

As a result, a task can derive from another task in the same call, regardless of order.

### `derive_schema()`

```python
derive_schema() -> DatasetSchema
```

`derive_schema` walks the dataset's instances and builds its
[`DatasetSchema`](types.md#datasetschema). The schema holds the distinct specs, annotation
descriptors, and task types, deduplicated in the original order. `derive_schema` stores the result in
`dataset.schema` and returns it. It never reads series values. The engine calls `derive_schema` after
`convert()` and before the writer runs.

### Properties

`TimeFDataset` exposes these properties:

- `metadata`
- `samples` (tuple, read-only)
- `tasks` (tuple, read-only)
- `schema` (`DatasetSchema | None`). This is `None` until `derive_schema()` runs, or until the reader
  populates it.

### `tasks_of()` / `tasks_for()`

```python
tasks_of(task_type) -> tuple[Task, ...]
tasks_for(sample, task_type=Task) -> tuple[Task, ...]
```

`tasks_of` returns every task of a type across the dataset. `tasks_for` resolves one sample's
`task_ids` back to task objects. `tasks_for` can filter the result by type.

### `to_features_and_targets()`

```python
to_features_and_targets(*, task=None, output="arrow", features="timestep")
    -> tuple[pa.Array, pa.Array] | tuple[np.ndarray, np.ndarray]
```

`to_features_and_targets` builds an `(X, y)` training pair. **By default, it defers
materialization.**

The `features` parameter picks the shape of `X`:

- `"timestep"` (the default) gives one feature per point. This is a rectangular
  `FixedSizeListArray[T]` (or `(n, T)`) matrix, and it needs samples of equal length.
- `"series"` gives one sequence per sample. This is a `ListArray` (or `(n,)`) object array, and it
  also handles series of variable length.

The `output` parameter picks how `to_features_and_targets` builds the arrays:

- `output="arrow"` (the default) builds the arrays directly from the loaders, with no NumPy copy.
- `output="numpy"` materializes the arrays.

`to_features_and_targets` infers `task` when the dataset has exactly one target-bearing type.
Otherwise, pass `task` explicitly. Note that `ForecastingTask` has no scalar target.

### `describe()`

```python
describe(*, rows=5, file=None) -> None
```

`describe` prints a plain-text summary, similar to the `describe` and `info` methods in pandas. The
summary includes identity, counts, per-spec columns (name, units, and the value dtype sampled from
one series), and a preview of the first `rows` samples. The preview reads only span metadata, so it
never loads series values. `describe` works before `derive_schema()` runs, because it computes all
figures from the samples. `describe` needs no CLI or `rich` dependency. It writes to `file`
(`sys.stdout` by default).

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
