# TimeFReader

Deserializes a TimeF version directory into an in-memory `TimeFDataset`. Inverse of `TimeFWriter`.

One public type lives in `timenet/timef/reader.py`:

```
timenet/timef/reader.py
    TimeFReader
```

The reader holds the manifest, the tasks, and the time-series index in memory. Per-series values and per-sample annotations are read lazily — annotations are constructed when `read()` / `iter_samples()` materializes each `Sample`; per-series values are pulled through closures attached to each `TimeSeries`.

---

## At a glance

```python
from collections.abc import Iterator
from pathlib import Path

from timenet.timef.dataset import Sample, TimeFDataset
from timenet.timef.metadata import DatasetMetadata, DatasetSchema
from timenet.tasks import Task


class TimeFReader:

    def __init__(self, root: Path) -> None: ...

    def read(self) -> TimeFDataset: ...
    def iter_samples(self) -> Iterator[Sample]: ...

    @property
    def metadata(self) -> DatasetMetadata: ...
    @property
    def schema(self) -> DatasetSchema: ...
    @property
    def tasks(self) -> tuple[Task, ...]: ...
```

```python
reader = TimeFReader(root / "ecg_dataset" / "1.0.0")
dataset = reader.read()
for sample in dataset.samples:
    values = sample.time_series[0].reader()
```

---

## On `TimeSeries.reader`

The `reader` (and `timestamps`) callable on each `TimeSeries` returned by `TimeFReader` is **not** the closure the connector supplied during conversion. The connector's closure pulled bytes from the original raw source (EDF, S3, HDF5, …) for the writer to consume. The closure attached here is constructed by `TimeFReader` and pulls bytes from the shard parquet files in `root/`.

The interface is identical, `Callable[[], np.ndarray]` returning the 1-D `float32` values for the series.

---

## `TimeFReader`

### `__init__()`

```python
def __init__(self, root: Path) -> None: ...
```

Opens the dataset. Reads `manifest.json`; synthesizes `TimeSeriesSpec`, `Device`, and `Annotation` subclasses (`StaticAnnotation` / `PointAnnotation` / `IntervalAnnotation`) from the manifest `schema` catalogs (see [Subclass synthesis](#subclass-synthesis)). Reads every `tasks/task=*/part-*.parquet`, constructs `Task` instances, resolves `from_tasks`. Reads `annotations.parquet` and constructs the `id → Annotation` lookup using the synthesized `Annotation` subclasses. Loads `time_series_index.parquet` into a lookup dict. After the constructor returns, `metadata`, `schema`, and `tasks` are populated and `read()` / `iter_samples()` can be called. Probably this will be explited into multiple methods

**Parameters**

| Name   | Type   | Default  | Description                                                                             |
| ------ | ------ | -------- | --------------------------------------------------------------------------------------- |
| `root` | `Path` | required | The version directory written by `TimeFWriter`,`<writer_root>/<dataset_id>/<version>/`. |

**Raises**

| Exception           | Condition                                                            |
| ------------------- | -------------------------------------------------------------------- |
| `FileNotFoundError` | `root` does not exist, or `root/manifest.json` is missing.           |
| `ValueError`        | `manifest.json["format_version"]` is not supported.                  |
| `ValueError`        | A `tasks/task=<task_id>/` partition references an unknown `task_id`. |
| `ValueError`        | An `annotations.parquet` row's `key` is not in `schema.annotations`. |
| `FileNotFoundError` | A file listed in `manifest.json["files"]` does not exist on disk.    |

---

### `read()`

```python
def read(self) -> TimeFDataset: ...
```

Reads `samples.parquet`, builds every `Sample` with lazy `TimeSeries.reader` and `timestamps` closures, attaches `Annotation` instances by resolving each row's `annotation_ids` against the lookup built by `__init__`, populates `task_ids` from the parquet column, and returns a `TimeFDataset` constructed with the reconstructed `metadata` and containing those samples and the tasks loaded by `__init__`. The reader [populates `dataset.schema`](timef-dataset.md#schema) from `manifest.json["schema"]` once at read time (the same value exposed as [`reader.schema`](#schema)).

```python
reader = TimeFReader(root)
dataset = reader.read()
```

---

### `iter_samples()`

```python
def iter_samples(self) -> Iterator[Sample]: ...
```

Yields `Sample` objects one at a time without materializing the full sample list. Useful when the sample count is large and the caller processes one sample at a time.

`iter_samples()` does not return a `TimeFDataset`. Tasks remain accessible through `reader.tasks`; per-sample annotations are attached to each yielded `Sample`.

```python
reader = TimeFReader(root)
for sample in reader.iter_samples():
    values = sample.time_series[0].reader()
```

---

### `metadata`

```python
@property
def metadata(self) -> DatasetMetadata: ...
```

The dataset's descriptive identity, reconstructed from `manifest.json["metadata"]`.

---

### `schema`

```python
@property
def schema(self) -> DatasetSchema: ...
```

The dataset's type declaration, reconstructed from `manifest.json["schema"]`. Carries the synthesized `TimeSeriesSpec`, `Device`, and `Annotation` subclasses and the resolved `Task` types (see [Subclass synthesis](#subclass-synthesis)).

---

### `tasks`

```python
@property
def tasks(self) -> tuple[Task, ...]: ...
```

All tasks in the dataset, with `from_tasks` resolved to the rebuilt `Task` instances. Same tuple is returned by `read().tasks`.

---

## Subclass synthesis

`TimeSeriesSpec`, `Device`, and `Annotation` subclasses declared by the connector are not stored on disk; only their serialized fields. The reader rebuilds them from `manifest.json`:

- **`TimeSeriesSpec`**: for each entry in `manifest.json["schema"]["time_series_specs"]`, an anonymous subclass is created with `spec_id`, `name`, `unit_sampling_rate`, `unit_timestamp`, `unit_value`, and `device` populated as ClassVars. `device` resolves against the synthesized `Device` subclasses by `device_id`. The synthesized class is the type carried by every `TimeSeries.spec` for that `spec_id`.
- **`Device`**: one anonymous subclass per `manifest.json["schema"]["devices"]` entry; ClassVars (`device_id`, `name`, `manufacturer`, `model`) set from the manifest.
- **`Annotation`**: one anonymous subclass per `manifest.json["schema"]["annotations"]` entry. The entry's `annotation_type` selects the base shape — `StaticAnnotation`, `PointAnnotation`, or `IntervalAnnotation` — and the synthesized subclass derives from it. ClassVars (`key`, `unit`, `description`) are set from the manifest. When `value_type` is non-`null`, the subclass's `value` field type is narrowed from it (`"int" → int`, `"float" → float`, `"str" → str`, `"bool" → bool`, `"list" → list`) and per-row `value` cells are decoded from JSON against that type; when `value_type` is `null` the annotation carries no value. Temporal subtypes additionally read `start_time_s` / `end_time_s` / `time_series_ids` from each row.
- **`Task`**: looked up in the built-in [`TASKS` registry](types.md#task-registry) by `task_id`. Task payload shapes are fixed in code. An unknown `task_id` raises `ValueError` during `__init__`.

Synthesized subclasses are not `is`-equal to the originals the connector defined. They compare equal by their discriminator (`spec_id`, `device_id`, `key`) and carry the same field values.

---

## Lifecycle

| Step | Method                | Action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `__init__`            | Reads `manifest.json`. Synthesizes `TimeSeriesSpec`, `Device`, and `Annotation` subclasses (the latter as `StaticAnnotation` / `PointAnnotation` / `IntervalAnnotation` per `annotation_type`). Resolves `Task` subclasses against the [`TASKS` registry](types.md#task-registry). Reads every `tasks/task=*/part-*.parquet`, constructs `Task` objects, resolves `from_tasks`. Reads `annotations.parquet` and constructs the `id → Annotation` lookup. Loads `time_series_index.parquet` into the lookup dict keyed by `(sample_id, series_id)`. Populates `metadata`, `schema`, and `tasks`. Single I/O step. |
| 2a   | `read()`              | Reads `samples.parquet`. Constructs each `Sample` with lazy `TimeSeries.reader` / `timestamps` closures, attaches `Annotation` instances (static and temporal) by resolving `annotation_ids` against the annotations lookup, populates `task_ids` from the parquet column. Returns a `TimeFDataset` bundling those samples with the tasks loaded in step 1.                                                                                                                                                                                                                                                      |
| 2b   | `iter_samples()`      | Same as `read()` but yields each `Sample` lazily without building a `TimeFDataset`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| —    | `TimeSeries.reader()` | Opens the shard parquet file via `pyarrow.parquet.read_table` with a row-group filter, reads each chunk's `values`, concatenates in `chunk_idx` order, returns a 1-D `float32` array. The file handle is opened and closed within the call. `TimeSeries.timestamps()` does the same against the chunk's `timestamps` column when present.                                                                                                                                                                                                                                                                        |

---

## Round-trip semantics

For a `TimeFDataset` `D` that passes `TimeFWriter` validation:

**Preserved**

- Per `Sample`: `sample_id`, `view`, `subject_ids`, `task_ids`, `annotations`, the channel set and order of `time_series`.
- Per `TimeSeries`: `spec.spec_id`, `spec.channel`, `source_id`, `sampling_rate`, `series_id`, `t_start_s`, `t_end_s`, the byte content of `reader()`, and the byte content of `timestamps()` when set.
- Per `Annotation`: `id`, synthesized subclass identity (carries `key`, `unit`, `description` as `ClassVar`s, plus the `annotation_type` shape), `value` when present, and `start_time_s` / `end_time_s` / `time_series_ids` for temporal subtypes. Back-refs to samples live in `annotations.parquet`, not on the in-memory `Annotation`.
- Per `Task`: `id`, subclass and payload fields, `sample_ids`, `from_tasks` (resolved to the rebuilt `Task` objects).

**Not preserved**

- Python object identity of `TimeSeries` instances. `series_id` is the durable handle: two samples that referenced the same series before write reference the same `series_id` after read, but the Python objects are distinct (`id(sample_a.time_series[0]) != id(sample_b.time_series[0])`).
- Python object identity of synthesized `TimeSeriesSpec`, `Device`, and `Annotation` subclasses. The synthesized subclasses match the originals field-for-field but are not the originals.

---

## Validation

Checked during `__init__`, in this order:

| Check                                                                                         | Raises              |
| --------------------------------------------------------------------------------------------- | ------------------- |
| `root` exists                                                                                 | `FileNotFoundError` |
| `root/manifest.json` exists                                                                   | `FileNotFoundError` |
| `manifest.json["format_version"]` matches a supported version (`1`)                           | `ValueError`        |
| Every file listed in `manifest.json["files"]` exists (incl. `annotations.parquet`)            | `FileNotFoundError` |
| Every `tasks/task=<task_id>/` partition has a known `task_id`                                 | `ValueError`        |
| Every `annotations.parquet` row's `key` is in `schema.annotations`                            | `ValueError`        |
| Every `annotations.parquet` row's `annotation_type` is one of `static` / `point` / `interval` | `ValueError`        |

A chunk row referenced by `time_series_index.parquet` that points at a missing shard or row group is detected when the corresponding `TimeSeries.reader()` is called, not eagerly. The call raises `ValueError`.

---

## Examples

Read a full dataset and iterate every sample's first channel:

```python
from pathlib import Path

from timenet.timef.reader import TimeFReader

reader = TimeFReader(Path("~/.timenet/processed/ecg_dataset/1.0.0").expanduser())
dataset = reader.read()
for sample in dataset.samples:
    values = sample.time_series[0].reader()
    # train, score, or aggregate values here
```

Stream samples one at a time:

```python
reader = TimeFReader(root)
for sample in reader.iter_samples():
    process(sample)
```

Another example:

```python
def load_ecg_samples(path: Path) -> tuple[Sample, ...]:
    return TimeFReader(path).read().samples

samples = load_ecg_samples(root / "ecg_dataset" / "1.0.0")
arr = samples[0].time_series[0].reader()
```

Look up tasks by sample. Tasks live at the dataset level on the reader; annotations (static and temporal) live on each sample:

```python
from timenet.timef.types import PointAnnotation, IntervalAnnotation

reader = TimeFReader(root)

# Tasks: dataset-level entities, exposed at the reader.
tasks_by_sample: dict[str, list[Task]] = {}
for t in reader.tasks:
    for sid in t.sample_ids:
        tasks_by_sample.setdefault(sid, []).append(t)

# Annotations: per-sample, reach via samples. Filter by subtype as needed.
dataset = reader.read()
for sample in dataset.samples:
    for ann in sample.annotations:
        if isinstance(ann, (PointAnnotation, IntervalAnnotation)):
            ...   # temporal marker
        else:
            ...   # static context
```
