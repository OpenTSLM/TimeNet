# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their tasks as Python
objects. No I/O: persistence is the [`TimeFWriter`](timef-writer.md) concern. Lives in `timenet.dataset`.

---

## TimeSeries

Reference to one channel of time-series data, with optional windowing and a lazy Arrow loader.

```python
from timenet.dataset import TimeSeries

TimeSeries(
    spec=ecg_lead,            # a TimeSeriesSpec (the modality)
    channel="II",             # the channel this series carries
    sampling_rate_hz=500.0,
    loader=load_lead_ii,      # Callable[[], pa.Array] returning float32 values
    source_id="rec_001",      # optional
    t_start_s=0.0,
    t_end_s=None,             # None = to end of source
)
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `spec` | `TimeSeriesSpec` | yes | The modality (shared across channels). |
| `channel` | `str` | yes | The single channel this series carries (e.g. `"II"`). |
| `sampling_rate_hz` | `float` | yes | Sampling rate in canonical Hz; positive and finite. |
| `loader` | `Callable[[], pa.Array]` | yes | Lazy loader returning the 1-D `float32` values. |
| `source_id` | `str \| None` | no | Identifier of the raw recording this series came from. |
| `time_series_id` | `str` | no | Persistent handle (auto uuid7). The writer dedupes by it. |
| `t_start_s` | `float` | no | Window start in the source timeline (default `0.0`). |
| `t_end_s` | `float \| None` | no | Window end, or `None` for end of source. Must exceed `t_start_s`. |

`TimeSeries` is frozen with identity equality (`eq=False`): the writer dedupes by `time_series_id`, so
reusing one instance (or giving two instances the same explicit id) collapses to one chunk on disk.
Consumers read values through `to_arrow()` (Arrow, zero-copy) or `to_numpy()`; `loader` is plumbing
supplied by the connector at curation and by [`TimeFReader`](timef-reader.md) on read-back.

---

## Sample

One logical unit of time-series data: a recording, a session, a sensor bundle, a market window. Created
via `TimeFDataset.add_sample`.

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Auto uuid7 (or explicit, for deterministic output). |
| `time_series` | `tuple[TimeSeries, ...]` | One `TimeSeries` per channel. |
| `view` | `View` | Which slice of the source this sample represents. |
| `subject_ids` | `tuple[str, ...]` | Subjects (empty for subject-less domains). |
| `task_ids` | `tuple[str, ...]` | Ids of tasks attached via `add_task` (populated after construction). |
| `annotations` | `tuple[Annotation, ...]` | Attached via `add_annotation`. |

`add_annotation(annotation)` attaches and returns it, validating that a temporal annotation's
`time_series_ids` resolve to series on the sample, and that a trial-level `IntervalAnnotation` is only
added when the sample's series share a common `(t_start_s, t_end_s)` span.

---

## TimeFDataset

```python
from timenet.dataset import TimeFDataset

dataset = TimeFDataset(metadata=metadata)
sample = dataset.add_sample(time_series=(...), view=View.FULL, subject_ids=("p1",))
dataset.add_task(sample, ClassificationTask(label="afib"))
dataset.derive_schema()
```

### `add_sample(*, time_series, view, subject_ids=(), sample_id=None) -> Sample`

Creates a sample, registers it, returns it. Raises `ValueError` if `time_series` is empty. Pass
`sample_id` for deterministic output (e.g. golden fixtures).

### `add_task(samples, task, *, from_tasks=()) -> Task`

Registers a task and links it to its samples: populates `task.sample_ids` and appends `task.id` to each
sample's `task_ids`. `from_tasks` overrides the task's own value only when non-empty, so a task built
with `from_tasks=` is never clobbered. Raises `ValueError` on empty `samples` or a `LabelingTask` whose
`time_series_ids` do not resolve to every target sample.

### `derive_schema() -> DatasetSchema`

Walks the dataset's instances and builds its [`DatasetSchema`](types.md#datasetschema): the distinct
specs, data sources, annotation descriptors, and task types (order-preserving dedupe). Stores the result
(`dataset.schema`) and returns it. Never reads series values. The engine calls it after `convert()`,
before the writer runs.

### Properties

`metadata`, `samples` (tuple, read-only), `tasks` (tuple, read-only), and `schema`
(`DatasetSchema | None`, `None` until `derive_schema()` runs or the reader populates it).

### `describe(*, rows=5, file=None) -> None`

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
  tasks        question_and_answer=48000

specs
  spec         name         value          rate   dtype
  tsqa_series  TSQA Series  dimensionless  hertz  float

samples (first 5 of 48000)
  sample_id  view  channels  length  tasks  annotations
  row-0      full  1         64      1      1
```
