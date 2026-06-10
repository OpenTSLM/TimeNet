# TimeFWriter

Serializes a `TimeFDataset` and its associated signal arrays to disk in the TimeF format.

Two public types live in `timenet/timef/writer.py`:

```
timenet/timef/writer.py
    WriteProgressEvent     (frozen dataclass)
    TimeFWriter            (context manager)
```

The writer pulls bytes from `Signal.reader` callables attached to the dataset's samples.

---

## At a glance

```python
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from timenet.timef.dataset import TimeFDataset


@dataclass(frozen=True)
class WriteProgressEvent:
    stage: Literal["signal", "shard_finalized", "samples", "annotations", "index", "manifest", "commit"]
    completed: int
    total: int | None
    message: str | None = None


class TimeFWriter:

    def __init__(
        self,
        root: Path,
        dataset: TimeFDataset,
        *,
        shard_target_bytes: int = 512 * 2**20,
        chunk_max_bytes: int = 8 * 2**20,
        compression: Literal["zstd", "snappy", "none"] = "zstd",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> None: ...

    def __enter__(self) -> "TimeFWriter": ...
    def __exit__(self, exc_type, exc_val, tb) -> None: ...

    def write(self) -> None: ...
    def close(self) -> None: ...
    def abort(self) -> None: ...
```

---

## How sharing and splitting work

**Identity-based sharing.** Before serializing, the writer walks `dataset.samples` and groups `Signal` instances by Python identity (`id(signal)`). One unique `Signal` produces one or more chunks on disk, every chunk's `sample_ids` column lists every sample that referenced that Signal.

Two `Signal`s with equal fields but different Python identity produce **two** chunks. Reusing the same `Signal` object across samples is how connectors declare shared bytes.

**Chunk splitting.** A single `Signal.reader()` call may return a large array. If its payload exceeds `chunk_max_bytes` (default 8 MB), the writer splits it into sequential sub-chunks with increasing `t_start_s`. Each sub-chunk gets its own `chunk_idx` row in `signal_index.parquet`. The Signal's `(spec_id, channel, source_id, t_start_s, t_end_s)` metadata in `samples.parquet` is unaffected — splitting is purely a storage-layer concern.

---

## `WriteProgressEvent`

Emitted by the writer at each stage of the write pipeline. Passed to `progress_cb` if one was supplied.

```python
@dataclass(frozen=True)
class WriteProgressEvent:
    stage: Literal["signal", "shard_finalized", "samples", "annotations", "index", "manifest", "commit"]
    completed: int
    total: int | None
    message: str | None = None
```

**Fields**

| Field       | Type           | Description                                                                  |
| ----------- | -------------- | ---------------------------------------------------------------------------- |
| `stage`     | `Literal[...]` | Which phase is reporting.                                                    |
| `completed` | `int`          | Units completed so far in this stage.                                        |
| `total`     | `int \| None`  | Total units in this stage, or `None` when the total is not known in advance. |
| `message`   | `str \| None`  | Optional human-readable detail.                                              |

---

## `TimeFWriter`

Context manager that serializes the dataset's signals into the TimeF format on disk.

### `__init__()`

```python
def __init__(
    self,
    root: Path,
    dataset: TimeFDataset,
    *,
    shard_target_bytes: int = 512 * 2**20,
    chunk_max_bytes: int = 8 * 2**20,
    compression: Literal["zstd", "snappy", "none"] = "zstd",
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> None: ...
```

**Parameters**

| Name                 | Type                                           | Default       | Description                                                                                         |
| -------------------- | ---------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------- |
| `root`               | `Path`                                         | required      | Parent directory. The writer creates `<root>/<dataset_id>/<version>/` on `__enter__`.               |
| `dataset`            | `TimeFDataset`                                 | required      | Populated dataset. The writer reads `dataset.samples` and `dataset.metadata` during finalize.       |
| `shard_target_bytes` | `int`                                          | `512 * 2**20` | Target for shard file size. A new shard is opened when the current one exceeds this threshold.      |
| `chunk_max_bytes`    | `int`                                          | `8 * 2**20`   | A `Signal.reader()` payload exceeding this size is split into sequential sub-chunks before storing. |
| `compression`        | `Literal["zstd", "snappy", "none"]`            | `"zstd"`      | Parquet compression codec applied to all output files.                                              |
| `progress_cb`        | `Callable[[WriteProgressEvent], None] \| None` | `None`        | Called from the writer thread after each progress event.                                            |

**Raises**

| Exception    | Condition                                           |
| ------------ | --------------------------------------------------- |
| `ValueError` | `dataset.dataset_id` or `dataset.version` is empty. |

---

### `__enter__()` / `__exit__()`

```python
def __enter__(self) -> "TimeFWriter": ...
def __exit__(self, exc_type, exc_val, tb) -> None: ...
```

`__enter__` creates `<root>/<dataset_id>/<version>/` if it does not exist. Raises `FileExistsError` if the directory already exists and contains a `manifest.json` (i.e. a committed version is already there).

`__exit__` calls `close()` on success or `abort()` on exception.

**Raises** (from `__enter__`)

| Exception         | Condition                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `FileExistsError` | `<root>/<dataset_id>/<version>/manifest.json` already exists (committed version present). |

---

### `write()`

```python
def write(self) -> None: ...
```

Walks `dataset.samples`, dedupes signals by Python identity, calls each unique `Signal.reader()` once, splits the payload into chunks if needed, and finalizes every output file. After `write()` returns successfully, `close()` (or the context manager's `__exit__`) only has to flush the manifest.

Callers just open the writer as a context manager and call `write()`

```python
with TimeFWriter(root, dataset) as writer:
    writer.write()
```

Raises on the first validation failure and leaves the writer unusable; `__exit__` will call `abort()`. See [Validation](#validation) for the full check list.

---

### `close()`

```python
def close(self) -> None: ...
```

Commits `manifest.json` if `write()` has run successfully. Called automatically by `__exit__` on success. If `write()` has not run, `close()` raises `RuntimeError`. See [Lifecycle](#lifecycle) for the full sequence.

---

### `abort()`

```python
def abort(self) -> None: ...
```

Deletes `<root>/<dataset_id>/<version>/`. Called automatically by `__exit__` on exception. Safe to call multiple times.

---

## Disk layout

```
<root>/<dataset_id>/<version>/
  manifest.json
  samples.parquet
  annotations/
    task=classification/part-0.parquet
    task=labeling/part-0.parquet
    task=captioning/part-0.parquet
    task=question_and_answer/part-0.parquet
    task=forecasting/part-0.parquet
    task=reasoning/part-0.parquet
  signals/
    shard-00000.parquet
    shard-00001.parquet
    ...
  signal_index.parquet
```

`manifest.json` is the commit marker. A version directory without it is in-progress and must be ignored by readers.

Annotations are partitioned by `task_id` because task subclasses have disjoint payload fields; one wide table would have many nulls.

---

## File schemas

### `samples.parquet`

One row per sample, sorted by `sample_id`.

| Column          | Arrow type                      | Description                                                                                                     |
| --------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `sample_id`     | `string`                        |                                                                                                                 |
| `view`          | `string`                        | String value of the [`View`](types.md#view) enum member on `Sample.view`.                                       |
| `subject_ids`   | `list<string>`                  | Empty list for subject-less domains (finance, seismology, synthetic).                                           |
| `source_ids`    | `list<string>`                  | Distinct `Signal.source_id` values across `signals`, in first-seen order. Derived from `signals` at write time. |
| `signals`       | `list<struct<...>>` — see below | One entry per `Signal` on the sample.                                                                           |
| `n_annotations` | `int32`                         |                                                                                                                 |

`signals` struct fields:

| Field              | Arrow type           | Notes                                         |
| ------------------ | -------------------- | --------------------------------------------- |
| `spec_id`          | `string`             | Dictionary-encoded.                           |
| `channel`          | `string`             | Dictionary-encoded.                           |
| `source_id`        | `string`             | Dictionary-encoded.                           |
| `sampling_rate_hz` | `float64`            |                                               |
| `t_start_s`        | `float64`            | Window start in the source's timeline.        |
| `t_end_s`          | `float64` (nullable) | `null` when the signal runs to end of source. |

---

### `annotations/task=<task_id>/part-0.parquet`

One file per task type. Only created if the dataset has at least one annotation of that type.

Common columns across all partitions:

| Column                | Arrow type          |
| --------------------- | ------------------- |
| `annotation_id`       | `string`            |
| `sample_ids`          | `list<string>`      |
| `spec_id`             | `string` (nullable) |
| `from_annotation_ids` | `list<string>`      |

Task-specific columns:

| Task                  | Extra columns                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `classification`      | `label: string`                                                                                   |
| `labeling`            | `label: string`, `channels: list<string>` (nullable), `windows_s: list<list<float64>>` (nullable) |
| `captioning`          | `answer: string`                                                                                  |
| `question_and_answer` | `question: string`, `answer: string`                                                              |
| `forecasting`         | `context_sample_ids: list<string>`, `target_sample_id: string`                                    |
| `reasoning`           | `question: string`, `answer: string`                                                              |

---

### `signals/shard-N.parquet`

One row per chunk. Sorted by `(spec_id, channel, chunk_idx)`. Row groups flushed every 10,000 rows [TODO: this has to be discussed].

| Column             | Arrow type                                  |
| ------------------ | ------------------------------------------- |
| `sample_ids`       | `list<string>` (dictionary-encoded entries) |
| `spec_id`          | `string` (dictionary-encoded)               |
| `channel`          | `string` (dictionary-encoded)               |
| `chunk_idx`        | `int32`                                     |
| `t_start_s`        | `float64`                                   |
| `n_samples`        | `int32`                                     |
| `sampling_rate_hz` | `float64`                                   |
| `values`           | `list<float32>`                             |
| `timestamps`       | `list<float64>` (nullable)                  |

A new shard is opened when the current one exceeds `shard_target_bytes`. Per-sample lookup goes through `signal_index.parquet`.

---

### `signal_index.parquet`

The reader's primary lookup table. One row per `(sample_id, chunk)` pair. A chunk shared by N samples contributes N rows, all pointing at the same shard row.

Sorted by `(sample_id, spec_id, channel, chunk_idx)`.

| Column       | Arrow type |
| ------------ | ---------- |
| `sample_id`  | `string`   |
| `spec_id`    | `string`   |
| `channel`    | `string`   |
| `chunk_idx`  | `int32`    |
| `shard_path` | `string`   |
| `row_group`  | `int32`    |
| `row_offset` | `int32`    |
| `t_start_s`  | `float64`  |
| `t_end_s`    | `float64`  |
| `n_samples`  | `int32`    |

---

### `manifest.json`

Written last by `close()`. Its presence is the commit signal.

```json
{
  "writer_version": "0.1.0",
  "format_version": 1,
  "created_utc": "2026-05-11T09:00:00Z",
  "dataset_id": "ecg_dataset",
  "version": "1.0.0",
  "metadata": {
    "dataset_id": "ecg_dataset",
    "version": "1.0.0",
    "description": "100 patients, one 12-lead ECG recording each.",
    "license": "CC BY 4.0",
    "domains": ["cardiology"],
    "source_url": null,
    "tags": [],
    "signal_specs": [
      {
        "spec_id": "ecg_12lead",
        "name": "ECG",
        "channels": [
          "I",
          "II",
          "III",
          "aVR",
          "aVL",
          "aVF",
          "V1",
          "V2",
          "V3",
          "V4",
          "V5",
          "V6"
        ],
        "unit_sampling_rate": "Hz",
        "unit_timestamp": "s",
        "unit_value": "mV",
        "sensor_id": null
      }
    ],
    "annotation_specs": [
      { "spec_id": "rhythm_cls", "task": "classification", "schema": null }
    ]
  },
  "counts": {
    "samples": 1000,
    "annotations_by_task": { "classification": 1000, "labeling": 4000 },
    "signal_chunks": 12000,
    "signal_index_rows": 14500,
    "signal_specs": { "ecg_12lead": { "samples": 1000, "channels": 12 } }
  },
  "files": {
    "samples": "samples.parquet",
    "annotations": ["annotations/task=classification/part-0.parquet"],
    "signals": ["signals/shard-00000.parquet"],
    "signal_index": "signal_index.parquet"
  },
  "checksums": {
    "samples.parquet": "sha256:...",
    "signal_index.parquet": "sha256:..."
  }
}
```

`signal_chunks` counts unique rows across all shards. `signal_index_rows` counts exploded index rows, equal to `signal_chunks` when no chunks are shared, greater when sharing occurs.

---

## Lifecycle

Full sequence from construction to commit:

| Step | Method      | Action                                                                                                                                                                                                                             |
| ---- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `__init__`  | Validates `dataset` (non-empty `dataset_id` and `version`). No disk I/O.                                                                                                                                                           |
| 2    | `__enter__` | Creates `<root>/<dataset_id>/<version>/`. Raises `FileExistsError` if `manifest.json` already exists there.                                                                                                                        |
| 3a   | `write`     | Validates every `Sample.view` and every `Signal` against the dataset's metadata (see [Validation](#validation)). Raises `ValueError` on the first failure before any disk I/O.                                                     |
| 3b   | `write`     | Dedupes signals by Python identity. Builds the unique-signal set and a `signal_id → list[sample_id]` index.                                                                                                                        |
| 3c   | `write`     | For each unique signal, calls `signal.reader()` (and `signal.timestamps()` when set), validates the returned array, splits it if it exceeds `chunk_max_bytes`, and appends chunk rows to the open shard. Rotates shards as needed. |
| 3d   | `write`     | Validates that every annotation's `sample_ids` resolves. Writes `samples.parquet`, `annotations/`, `signal_index.parquet`.                                                                                                         |
| 4    | `close`     | Writes `manifest.json`, this is the commit.                                                                                                                                                                                        |
| —    | `abort`     | Deletes `<root>/<dataset_id>/<version>/` without committing. Called by `__exit__` on exception.                                                                                                                                    |

**Commit contract:** `manifest.json` is written last. A version directory that exists but has no `manifest.json` is in-progress or failed and must be ignored by readers.

**Failure in `write`:** raises immediately and leaves the writer unusable. `__exit__` will call `abort()`, deleting the partial directory.

**Failure in `close`:** `abort()` is called, deleting the partial directory.

---

## Validation

### Dataset-level (enforced at the start of `write()`)

| Check                                                                                       | Raises       |
| ------------------------------------------------------------------------------------------- | ------------ |
| Every `Sample.view` is a member of the `View` enum                                          | `ValueError` |
| `Sample.signals` is non-empty for every sample                                              | `ValueError` |
| Every `Signal.spec_id` is the `spec_id` of a `SignalSpec` in `DatasetMetadata.signal_specs` | `ValueError` |
| Every `Signal.channel` is one of the referenced `SignalSpec.channels`                       | `ValueError` |
| `Signal.sampling_rate_hz > 0`                                                               | `ValueError` |
| `Signal.t_start_s >= 0`, and `t_end_s is None or t_end_s > t_start_s`                       | `ValueError` |
| Every annotation's `sample_ids` only contains IDs present in `dataset.samples`              | `ValueError` |

### Per-signal (enforced as each `reader()` is called)

| Check                                                                                                                                | Raises       |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| `signal.reader()` returns a 1-D `float32`, non-empty, finite array                                                                   | `ValueError` |
| When `t_end_s is not None`: `len(values) == round((t_end_s - t_start_s) * sampling_rate_hz)` (off-by-one tolerated within ±1 sample) | `ValueError` |
| When `signal.timestamps` is set: same length as `reader()` output and monotonic non-decreasing                                       | `ValueError` |

Failure leaves the writer unusable. `__exit__` will call `abort()`.

---

## Progress reporting

`progress_cb` is called once per event. Events are emitted in the order listed below.

| `stage`           | `completed`              | `total`                                 | When                                                          |
| ----------------- | ------------------------ | --------------------------------------- | ------------------------------------------------------------- |
| `signal`          | signals serialized       | total unique signals across the dataset | after each unique `Signal` is read, validated, and serialized |
| `shard_finalized` | shards finalized so far  | `None`                                  | each time a shard is flushed and closed                       |
| `samples`         | `len(dataset.samples)`   | `len(dataset.samples)`                  | once, after `samples.parquet` is written                      |
| `annotations`     | annotations written      | total annotation count                  | once per task partition written                               |
| `index`           | total index rows written | total index rows                        | once, after `signal_index.parquet` is written                 |
| `manifest`        | `1`                      | `1`                                     | once, after `manifest.json` is written                        |
| `commit`          | `1`                      | `1`                                     | once, immediately after `manifest`                            |
