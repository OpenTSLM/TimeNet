# TimeFWriter

Serializes a `TimeFDataset` and its associated signal arrays to disk in the TimeF format.

Three public types live in `timenet/timef/writer.py`:

```
timenet/timef/writer.py
    SignalChunk            (frozen dataclass)
    WriteProgressEvent     (frozen dataclass)
    TimeFWriter            (context manager)
```

---

## At a glance

```python
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator
from typing import Literal

from timenet.timef.dataset import TimeFDataset


@dataclass(frozen=True)
class SignalChunk:
    sample_ids: tuple[str, ...]
    spec_id: str
    channel: str
    values: np.ndarray
    sampling_rate_hz: float
    t_start_s: float = 0.0
    timestamps: np.ndarray | None = None


@dataclass(frozen=True)
class WriteProgressEvent:
    stage: Literal["chunk", "shard_finalized", "samples", "annotations", "index", "manifest", "commit"]
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

    def add_chunk(self, chunk: SignalChunk) -> None: ...
    def add_chunks(self, chunks: Iterable[SignalChunk]) -> None: ...

    def close(self) -> None: ...
    def abort(self) -> None: ...
```

---

## `SignalChunk`

The unit the connector hands to the writer. One chunk = one channel of one or more samples.

```python
@dataclass(frozen=True)
class SignalChunk:
    sample_ids: tuple[str, ...]
    spec_id: str
    channel: str
    values: np.ndarray
    sampling_rate_hz: float
    t_start_s: float = 0.0
    timestamps: np.ndarray | None = None
```

**Fields**

| Field              | Type                 | Required | Description                                                                                                                                         |
| ------------------ | -------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_ids`       | `tuple[str, ...]`    | yes      | One or more sample IDs that own this data.                                                                                                          |
| `spec_id`          | `str`                | yes      | Must match a `SignalSpec.spec_id` declared in the dataset's metadata.                                                                               |
| `channel`          | `str`                | yes      | Channel name. Must be declared on the referenced `SignalSpec`.                                                                                      |
| `values`           | `np.ndarray`         | yes      | 1-D array, `dtype=float32`.                                                                                                                         |
| `sampling_rate_hz` | `float`              | yes      | Sampling rate in Hz.                                                                                                                                |
| `t_start_s`        | `float`              | no       | Time offset in seconds of `values[0]` within the recording. Default `0.0`.                                                                          |
| `timestamps`       | `np.ndarray \| None` | no       | Explicit per-sample timestamps (same length as `values`) `None` for uniform sampling, timestamps are derived as `t_start_s + i / sampling_rate_hz`. |

**Automatic splitting**

A connector may emit chunks of any length. If a chunk's `values` array exceeds the writer's `chunk_max_bytes` threshold (default 8 MB), the writer splits it into sub-chunks with increasing `t_start_s` before storing. The index records each sub-chunk as a separate row addressable by `chunk_idx`.

---

## `WriteProgressEvent`

Emitted by the writer at each stage of the write pipeline. Passed to `progress_cb` if one was supplied.

```python
@dataclass(frozen=True)
class WriteProgressEvent:
    stage: Literal["chunk", "shard_finalized", "samples", "annotations", "index", "manifest", "commit"]
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

Context manager that streams chunks into the TimeF format on disk.

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

| Name                 | Type                                           | Default       | Description                                                                                        |
| -------------------- | ---------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------- |
| `root`               | `Path`                                         | required      | Parent directory. The writer creates `<root>/<dataset_id>/<version>/` on `__enter__`.              |
| `dataset`            | `TimeFDataset`                                 | required      | Populated dataset. The writer reads `dataset.samples` and `dataset.metadata` during finalize.      |
| `shard_target_bytes` | `int`                                          | `512 * 2**20` | Target for shard file size. A new shard is opened when the current one exceeds this threshold.     |
| `chunk_max_bytes`    | `int`                                          | `8 * 2**20`   | Chunks whose `values` array exceeds this size are split into sequential sub-chunks before storing. |
| `compression`        | `Literal["zstd", "snappy", "none"]`            | `"zstd"`      | Parquet compression codec applied to all output files.                                             |
| `progress_cb`        | `Callable[[WriteProgressEvent], None] \| None` | `None`        | Called from the writer thread after each progress event.                                           |

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

### `add_chunk()`

```python
def add_chunk(self, chunk: SignalChunk) -> None: ...
```

Validates the chunk, optionally splits it, appends it to the open shard for its `spec_id`, and emits a `chunk` progress event.

Raises immediately on validation failure and leaves the writer in an unusable state; `__exit__` will call `abort()`.

**Validation**

| Check                                                                                                      | Raises       |
| ---------------------------------------------------------------------------------------------------------- | ------------ |
| `chunk.sample_ids` is non-empty                                                                            | `ValueError` |
| Each `sid` in `chunk.sample_ids` exists in `dataset.samples`                                               | `ValueError` |
| Each `sid` declares a `SignalRef` with `spec_id == chunk.spec_id`                                          | `ValueError` |
| `chunk.channel` is in the ref's `channels`, or in `SignalSpec.channels` when the ref's `channels` is empty | `ValueError` |
| `chunk.values` is 1-D, non-empty, and finite                                                               | `ValueError` |
| `chunk.sampling_rate_hz > 0`                                                                               | `ValueError` |
| If `chunk.timestamps is not None`: same length as `values`, monotonic non-decreasing                       | `ValueError` |
| `(sid, spec_id, channel, chunk_idx)` not seen before for any `sid`                                         | `ValueError` |

---

### `add_chunks()`

```python
def add_chunks(self, chunks: Iterable[SignalChunk]) -> None: ...
```

Calls `add_chunk()` for each element in `chunks`.

---

### `close()`

```python
def close(self) -> None: ...
```

Validates, finalizes all open shards, writes all output files, and commits `manifest.json`. Called automatically by `__exit__` on success. See [Lifecycle](#lifecycle) for the full sequence.

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

| Column          | Arrow type                                              |
| --------------- | ------------------------------------------------------- |
| `sample_id`     | `string`                                                |
| `subject_ids`   | `list<string>`                                          |
| `source_ids`    | `list<string>`                                          |
| `view`          | `string`                                                |
| `signals`       | `list<struct<spec_id: string, channels: list<string>>>` |
| `n_annotations` | `int32`                                                 |

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

A new shard is opened when the current one exceeds `shard_target_bytes`. Per-sample lookup goes through `signal_index.parquet`. Bulk modality scans filter on the dictionary-encoded `spec_id` column and benefit from row group statistics on `(spec_id, channel)` for skipping.

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

Scalar equality on `sample_id` is fast because the index is exploded.

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
    ],
    "view_specs": [
      {
        "name": "full",
        "description": "Whole recording with all available leads."
      }
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

`signal_chunks` counts unique rows across all shards. `signal_index_rows` counts exploded index rows — equal to `signal_chunks` when no chunks are shared, greater when sharing occurs.

The `metadata` object is the canonical snapshot of every `SignalSpec`, `AnnotationSpec`, and `ViewSpec` for this version. All other files reference them by `spec_id` / `task_id` string only.

---

## Lifecycle

Full sequence from construction to commit:

| Step | Method          | Action                                                                                                                                                                       |
| ---- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `__init__`      | Validates `dataset` (non-empty `dataset_id` and `version`). No disk I/O.                                                                                                     |
| 2    | `__enter__`     | Creates `<root>/<dataset_id>/<version>/`. Raises `FileExistsError` if `manifest.json` already exists there.                                                                  |
| 3    | `add_chunk` × N | Validates chunk, splits if `values` exceeds `chunk_max_bytes`, appends to the open shard. Rotates to a new shard when `shard_target_bytes` is exceeded.                      |
| 4a   | `close`         | Validates completeness: every `(sample_id, SignalRef)` in the dataset must have at least one chunk for at least one of its channels. Raises `ValueError` on missing signals. |
| 4b   | `close`         | Flushes all open shards.                                                                                                                                                     |
| 4c   | `close`         | Writes `samples.parquet`, `annotations/`, `signal_index.parquet`.                                                                                                            |
| 4d   | `close`         | Writes `manifest.json`, this is the commit                                                                                                                                   |
| —    | `abort`         | Deletes `<root>/<dataset_id>/<version>/` without committing. Called by `__exit__` on exception.                                                                              |

**Commit contract:** `manifest.json` is written last. A version directory that exists but has no `manifest.json` is in-progress or failed and must be ignored by readers.

**Failure in `add_chunk`:** raises immediately and leaves the writer unusable. `__exit__` will call `abort()`, deleting the partial directory.

**Failure in `close` (step 4a–4c):** `abort()` is called, deleting the partial directory.

---

## Validation

### Per-chunk (enforced in `add_chunk`)

| Check                                                                                                      | Raises       |
| ---------------------------------------------------------------------------------------------------------- | ------------ |
| `chunk.sample_ids` is non-empty                                                                            | `ValueError` |
| Each `sid` in `chunk.sample_ids` exists in `dataset.samples`                                               | `ValueError` |
| Each `sid` declares a `SignalRef` with `spec_id == chunk.spec_id`                                          | `ValueError` |
| `chunk.channel` is in the ref's `channels`, or in `SignalSpec.channels` when the ref's `channels` is empty | `ValueError` |
| `chunk.values` is 1-D, non-empty, and all values are finite                                                | `ValueError` |
| `chunk.sampling_rate_hz > 0`                                                                               | `ValueError` |
| If `chunk.timestamps is not None`: same length as `values` and monotonic non-decreasing                    | `ValueError` |
| `(sid, chunk.spec_id, chunk.channel, chunk_idx)` not previously seen for any `sid`                         | `ValueError` |

The last check ensures the exploded `signal_index.parquet` will have unique keys. `chunk_idx` is assigned by the writer per `(sid, spec_id, channel)` tuple and increments with each chunk for that combination.

Failure leaves the writer unusable. `__exit__` will call `abort()`.

### At-close-time (enforced in `close`)

| Check                                                                                                              | Raises       |
| ------------------------------------------------------------------------------------------------------------------ | ------------ |
| For every `(sample_id, SignalRef)` in the dataset, at least one chunk arrived for at least one channel of that ref | `ValueError` |
| Every annotation's `sample_ids` contains only sample IDs present in `dataset.samples`                              | `ValueError` |

---

## Progress reporting

`progress_cb` is called once per event. Events are emitted in the order listed below.

| `stage`           | `completed`              | `total`                | When                                          |
| ----------------- | ------------------------ | ---------------------- | --------------------------------------------- |
| `chunk`           | chunks processed so far  | `None`                 | after each `add_chunk` call                   |
| `shard_finalized` | shards finalized so far  | `None`                 | each time a shard is flushed and closed       |
| `samples`         | `len(dataset.samples)`   | `len(dataset.samples)` | once, after `samples.parquet` is written      |
| `annotations`     | annotations written      | total annotation count | once per task partition written               |
| `index`           | total index rows written | total index rows       | once, after `signal_index.parquet` is written |
| `manifest`        | `1`                      | `1`                    | once, after `manifest.json` is written        |
| `commit`          | `1`                      | `1`                    | once, immediately after `manifest`            |

`total` is `None` for `chunk` and `shard_finalized` because the writer does not know how many chunks or shards will arrive before `close()` is called.
