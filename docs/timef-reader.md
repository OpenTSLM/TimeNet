# TimeFReader

Deserializes a TimeF version directory into an in-memory `TimeFDataset`. Inverse of `TimeFWriter`.

One public type lives in `timenet/timef/reader.py`:

```
timenet/timef/reader.py
    TimeFReader
```

The reader holds the manifest, the annotations, and the time-series index in memory. Per-series values are read lazily through closures attached to each `TimeSeries`.

---

## At a glance

```python
from collections.abc import Iterator
from pathlib import Path

from timenet.timef.dataset import Annotation, Sample, TimeFDataset
from timenet.timef.metadata import DatasetMetadata


class TimeFReader:

    def __init__(self, root: Path) -> None: ...

    def read(self) -> TimeFDataset: ...
    def iter_samples(self) -> Iterator[Sample]: ...

    @property
    def metadata(self) -> DatasetMetadata: ...
    @property
    def annotations(self) -> tuple[Annotation, ...]: ...
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

Opens the dataset. Reads `manifest.json`, synthesizes the spec subclasses described in the manifest, builds the annotation set with `from_annotations` resolved, and loads `time_series_index.parquet` into a lookup dict. After the constructor returns, `metadata` and `annotations` are populated and `read()` / `iter_samples()` can be called.

**Parameters**

| Name   | Type   | Default  | Description                                                                             |
| ------ | ------ | -------- | --------------------------------------------------------------------------------------- |
| `root` | `Path` | required | The version directory written by `TimeFWriter`,`<writer_root>/<dataset_id>/<version>/`. |

**Raises**

| Exception           | Condition                                                                   |
| ------------------- | --------------------------------------------------------------------------- |
| `FileNotFoundError` | `root` does not exist, or `root/manifest.json` is missing.                  |
| `ValueError`        | `manifest.json["format_version"]` is not supported.                         |
| `ValueError`        | An `annotations/task=<task_id>/` partition references an unknown `task_id`. |
| `FileNotFoundError` | A file listed in `manifest.json["files"]` does not exist on disk.           |

---

### `read()`

```python
def read(self) -> TimeFDataset: ...
```

Reads `samples.parquet`, builds every `Sample` with lazy `TimeSeries.reader` and `timestamps` closures, and returns a `TimeFDataset` containing those samples and the annotations loaded by `__init__`.

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

`iter_samples()` does not return a `TimeFDataset`. Annotations remain accessible through `reader.annotations`.

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

The dataset metadata reconstructed from `manifest.json["metadata"]`.

---

### `annotations`

```python
@property
def annotations(self) -> tuple[Annotation, ...]: ...
```

All annotations in the dataset, with `from_annotations` resolved. Same tuple is returned by `read().annotations`.

---

## Spec class synthesis

`TimeSeriesSpec`, `DeviceSpec`, and `Task` subclasses are not stored on disk; only their serialized fields. The reader rebuilds them from `manifest.json`:

- **`TimeSeriesSpec`**: for each entry in `manifest.json["metadata"]["time_series_specs"]`, an anonymous subclass is created with `spec_id`, `name`, `unit_sampling_rate`, `unit_timestamp`, `unit_value`, and `device` populated from the manifest. `device` resolves against the synthesized `DeviceSpec` subclasses by `device_id`. The synthesized class is the type carried by every `TimeSeries.spec` for that `spec_id`.
- **`DeviceSpec`**: same pattern, one anonymous subclass per `manifest.json["metadata"]["device_specs"]` entry.
- **`Task`**: looked up in the built-in [`TASKS` registry](types.md#task-registry). An unknown `task_id` raises `ValueError` during `__init__`.

Synthesized subclasses are not `is`-equal to the original subclasses the connector defined. They compare equal by `spec_id` / `device_id` and carry the same field values.

---

## Lifecycle

| Step | Method                | Action                                                                                                                                                                                                                                                                                                                                                                      |
| ---- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `__init__`            | Reads `manifest.json`. Synthesizes `TimeSeriesSpec`, `DeviceSpec`, and resolves `Task` subclasses. Reads every `annotations/task=*/part-*.parquet`, constructs `Annotation` objects, resolves `from_annotations`. Reads `events.parquet` and constructs the `event_id → Event` lookup. Loads `time_series_index.parquet` into the lookup dict keyed by `(sample_id, series_id)`. Populates `metadata` and `annotations`. Single I/O step. |
| 2a   | `read()`              | Reads `samples.parquet`. Constructs each `Sample` with lazy `TimeSeries.reader` / `timestamps` closures, and attaches `Event` instances via the row's `event_ids`. Returns a `TimeFDataset` bundling those samples with the annotations loaded in step 1.                                                                                                                                                                                |
| 2b   | `iter_samples()`      | Same as `read()` but yields each `Sample` lazily without building a `TimeFDataset`.                                                                                                                                                                                                                                                                                                                                                      |
| —    | `TimeSeries.reader()` | Opens the shard parquet file via `pyarrow.parquet.read_table` with a row-group filter, reads each chunk's `values`, concatenates in `chunk_idx` order, returns a 1-D `float32` array. The file handle is opened and closed within the call. `TimeSeries.timestamps()` does the same against the chunk's `timestamps` column when present.                                   |

---

## Round-trip semantics

For a `TimeFDataset` `D` that passes `TimeFWriter` validation:

**Preserved**

- Per `Sample`: `sample_id`, `view`, `subject_ids`, `annotation_ids`, `events`, the channel set and order of `time_series`.
- Per `TimeSeries`: `spec.spec_id`, `spec.channel`, `source_id`, `sampling_rate`, `series_id`, `t_start_s`, `t_end_s`, the byte content of `reader()`, and the byte content of `timestamps()` when set.
- Per `Event`: `event_id`, `name`, `kind`, `start_time_s`, `end_time_s`, `time_series_ids`.
- Per `Annotation`: `annotation_id`, `task` (subclass and fields), `sample_ids`, `spec_id`, `from_annotations` (resolved to the rebuilt `Annotation` objects).

**Not preserved**

- Python object identity of `TimeSeries` instances. `series_id` is the durable handle: two samples that referenced the same series before write reference the same `series_id` after read, but the Python objects are distinct (`id(sample_a.time_series[0]) != id(sample_b.time_series[0])`).
- Python object identity of `TimeSeriesSpec` and `DeviceSpec` subclasses. The synthesized subclasses match the originals field-for-field but are not the originals.

---

## Validation

Checked during `__init__`, in this order:

| Check                                                                       | Raises              |
| --------------------------------------------------------------------------- | ------------------- |
| `root` exists                                                               | `FileNotFoundError` |
| `root/manifest.json` exists                                                 | `FileNotFoundError` |
| `manifest.json["format_version"]` matches a supported version (`1`)         | `ValueError`        |
| Every file listed in `manifest.json["files"]` exists (incl. `events.parquet`) | `FileNotFoundError` |
| Every `annotations/task=<task_id>/` partition has a known `task_id`         | `ValueError`        |
| Every `events.parquet` row has a `kind` value in [`EventKind`](types.md#eventkind) | `ValueError`        |

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

Look up annotations by sample:

```python
reader = TimeFReader(root)
by_sample: dict[str, list[Annotation]] = {}
for ann in reader.annotations:
    for sid in ann.sample_ids:
        by_sample.setdefault(sid, []).append(ann)
```
