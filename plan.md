# TimeFWriter — Spec Plan

Design plan for the `TimeFWriter` component and the connector additions that drive it. This document is the working blueprint; the formal spec at `docs/timef-writer.md` is being derived from it incrementally.

---

## Goals

1. Serialize a populated `TimeFDataset` plus its associated signal arrays to disk in a format optimized for AI-lab access patterns: random sample lookup, channel-subset reads, large-scale streaming.
2. Stay scalable: never require holding all signals in memory; data volume bounded by per-chunk size, not per-dataset size.
3. Be atomic: a reader either sees a valid dataset version or doesn't see it at all.
4. **Support shared signal data across samples.** A connector may emit multiple samples derived from the same source recording (history/future, multi-view, channel-subset variants). The format stores each unique chunk once and lets multiple samples reference it.
5. Round-trip cleanly with the reader (`Dataset` in `sdk.md`) — every byte the writer produces must be addressable from `(sample_id, spec_id, channel)`.

---

## Module structure

```
timenet/timef/writer.py
    SignalChunk            (frozen dataclass)
    WriteProgressEvent     (frozen dataclass)
    TimeFWriter            (the writer class)
```

`BaseConnector` in `timenet/connectors/base.py` gains two methods (`iter_signals`, `store`). No other module changes for this spec.

---

## End-to-end flow

```
connector.download(cache_dir)            # I/O, returns raw_refs
connector.convert(raw_refs)              # CPU, returns TimeFDataset (metadata-only)
connector.store(raw_refs, dataset, root)
        ↓
   TimeFWriter(root, dataset)            # context manager
        ↓
   for chunk in connector.iter_signals(raw_refs, dataset):
       writer.add_chunk(chunk)
        ↓
   on __exit__:
     finalize shards → write samples.parquet → write annotations/* →
     write signal_index.parquet → write manifest.json
```

The `TimeFDataset` returned by `convert()` is metadata-only and small; signals are streamed independently via `iter_signals()`, so the in-memory footprint is bounded by one chunk regardless of dataset size.

---

## Shared-chunk model

Connectors may now emit multiple samples derived from the same source recording. Some of those samples may share signal data — for example:

- `rec_001::full` (all 12 leads) and `rec_001::lead_II_only` (just lead II) share lead-II arrays.
- `rec_001::window_0_to_60s` and `rec_001::window_30_to_90s` share the 30–60s overlap.

A `SignalChunk` therefore carries `sample_ids: tuple[str, ...]` — every sample that owns this chunk's data. The writer stores the chunk once in the shard, and the index "explodes" the ownership list so per-sample lookups stay scalar-equality fast.

### Connector responsibility

The connector pre-aggregates owners before yielding. If two samples share a chunk, `iter_signals()` yields **one** `SignalChunk` whose `sample_ids` lists both. The connector must not yield the same data twice with different `sample_ids`.

Because `sample_id` values are derived in `convert()` and may not be trivially re-derivable from `TRaw` alone (e.g. `"rec_001::window_0"`), `iter_signals()` receives the already-built `dataset` as a second argument. The connector builds a source → sample_id(s) index from `dataset.samples` and uses it when constructing chunks:

```python
def iter_signals(self, raw_refs, dataset):
    by_source: dict[str, list[str]] = {}
    for s in dataset.samples:
        for source_id in s.source_ids:
            by_source.setdefault(source_id, []).append(s.sample_id)

    for rec in raw_refs:
        sids = tuple(by_source[rec.recording_id])  # length 1 or N
        for channel, values in read_signals(rec.path):
            yield SignalChunk(sample_ids=sids, ...)
```

The writer does **not** content-hash chunks to dedup. It trusts the connector's grouping. This keeps the hot path cheap.

### Common case is unaffected

A connector that emits non-overlapping samples just sets `sample_ids=("rec_001",)` — a length-1 tuple. Dictionary encoding makes the storage cost of the list-typed column indistinguishable from a scalar. The exploded index has one row per chunk (no explosion). All schemas and lookup paths behave identically to a single-owner-per-chunk design when sharing isn't used.

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

### Why these splits

- **Flat signals directory**: all modalities share the same shards. The shard schema is already generic (`channel`, `values`, `sampling_rate_hz`) and the index routes every read to the exact shard + row. Spec_id partitioning would multiply file count at no routing benefit — every access already goes through `signal_index.parquet`. Bulk modality scans filter on the dictionary-encoded `spec_id` column instead.
- **Partition annotations by `task_id`**: task subclasses have disjoint payload fields. One wide table forces ~80% nulls; partitioned tables stay tight and let `query_samples(task=...)` open just one file.
- **`signal_index.parquet`**: the primary lookup path. One row per `(sample_id, chunk)` pair (exploded view of the shard's `sample_ids` lists). Lets the reader resolve a request by simple scalar equality on `sample_id` without scanning shards.

---

## Public API

### `SignalChunk` (new)

The unit the connector hands to the writer. One chunk = one row in a signal shard, owned by one or more samples.

```python
@dataclass(frozen=True)
class SignalChunk:
    sample_ids: tuple[str, ...]     # samples this chunk's data belongs to (>=1)
    spec_id: str
    channel: str
    values: np.ndarray              # 1-D, dtype float32 by default
    sampling_rate_hz: float
    t_start_s: float = 0.0
    timestamps: np.ndarray | None = None  # only for non-uniform sampling
```

Notes:

- One chunk = one channel of one or more samples. Most chunks have a length-1 `sample_ids`. Multi-owner chunks are reserved for samples that genuinely share data.
- A short channel produces one chunk per owner-set; a long channel produces many chunks with increasing `t_start_s`.
- The connector emits chunks lazily and may emit them in any order — the writer handles bin-packing and sorting within shards.

### `BaseConnector` additions

`iter_signals` is abstract (each connector implements it). `store` is concrete (default orchestration; connectors rarely override).

```python
@abstractmethod
def iter_signals(self, raw_refs: list[TRaw], dataset: TimeFDataset) -> Iterator[SignalChunk]: ...

def store(
    self,
    raw_refs: list[TRaw],
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> None: ...
```

Default `store()` body:

```python
with TimeFWriter(root, dataset, progress_cb=progress_cb) as writer:
    for chunk in self.iter_signals(raw_refs, dataset):
        writer.add_chunk(chunk)
```

### `TimeFWriter`

```python
class TimeFWriter:
    def __init__(
        self,
        root: Path,
        dataset: TimeFDataset,
        *,
        shard_target_bytes: int = 512 * 2**20,
        chunk_max_bytes: int = 8 * 2**20,          # split chunks exceeding this size
        compression: Literal["zstd", "snappy", "none"] = "zstd",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> None: ...

    # Context-manager — recommended usage
    def __enter__(self) -> "TimeFWriter": ...
    def __exit__(self, exc_type, exc_val, tb) -> None: ...

    # Hot path
    def add_chunk(self, chunk: SignalChunk) -> None: ...
    def add_chunks(self, chunks: Iterable[SignalChunk]) -> None: ...

    # Manual lifecycle
    def close(self) -> None: ...
    def abort(self) -> None: ...
```

### `WriteProgressEvent`

```python
@dataclass(frozen=True)
class WriteProgressEvent:
    stage: Literal["chunk", "shard_finalized", "samples", "annotations", "index", "manifest", "commit"]
    completed: int
    total: int | None
    message: str | None = None
```

### What the writer does NOT expose

- No `add_sample` / `add_annotation` — those come from the populated `TimeFDataset`. The writer reads `dataset.samples` and serializes; the connector does not push samples one-by-one.
- No `read_*` methods — reads are `Dataset`'s job (in `sdk.md`).
- No partial-update / append API — `store()` is one-shot per `(dataset_id, version)`. Re-running with `force=True` rewrites the whole version directory.
- No content-hash deduplication — the connector controls grouping.

---

## File schemas

### `samples.parquet`

One row per sample. Sorted by `sample_id` for predicate pushdown. No arrays.

| Column          | Arrow type                                              |
| --------------- | ------------------------------------------------------- |
| `sample_id`     | `string`                                                |
| `subject_ids`   | `list<string>`                                          |
| `source_ids`    | `list<string>`                                          |
| `view`          | `string`                                                |
| `signals`       | `list<struct<spec_id: string, channels: list<string>>>` |
| `n_annotations` | `int32`                                                 |

### `annotations/task=<task_id>/part-N.parquet`

Partitioned by `task_id`. Each partition has a tight, task-specific schema (no nulls).

Common columns across all partitions:

| Column                | Type                |
| --------------------- | ------------------- |
| `annotation_id`       | `string`            |
| `sample_ids`          | `list<string>`      |
| `spec_id`             | `string` (nullable) |
| `from_annotation_ids` | `list<string>`      |

Plus task-specific columns:

| Task                  | Extra columns                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `classification`      | `label: string`                                                                                   |
| `labeling`            | `label: string`, `channels: list<string>` (nullable), `windows_s: list<list<float64>>` (nullable) |
| `captioning`          | `answer: string`                                                                                  |
| `question_and_answer` | `question: string`, `answer: string`                                                              |
| `forecasting`         | `context_sample_ids: list<string>`, `target_sample_id: string`                                    |
| `reasoning`           | `question: string`, `answer: string`                                                              |

A `task=X/` directory is only created if the dataset has at least one annotation of that task type.

### `signals/shard-N.parquet`

Sorted by `(spec_id, channel, chunk_idx)`. Row groups flushed every 10,000 rows (internal default, not exposed as a parameter).

| Column             | Type                                        |
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

Per-sample lookup goes through `signal_index.parquet`. Bulk modality scans filter on the dictionary-encoded `spec_id` column and benefit from row group statistics on `(spec_id, channel)` for skipping.

Why one channel per chunk:

- Variable channel coverage per sample (a sample may carry only a subset of its spec's channels).
- Channel-subset reads are a core AI-lab access pattern; predicate pushdown on `channel` skips bytes for unwanted channels.
- Independent chunking across channels (a sensor dropout on one channel doesn't constrain others).
- 1-D `list<float32>` is a first-class arrow type; jagged 2-D would compress and decode worse.

### `signal_index.parquet`

The reader's primary lookup table. **Exploded view**: one row per `(sample_id, chunk)` pair. A chunk shared by N samples contributes N rows here, all pointing at the same shard row.

| Column       | Type      |
| ------------ | --------- |
| `sample_id`  | `string`  |
| `spec_id`    | `string`  |
| `channel`    | `string`  |
| `chunk_idx`  | `int32`   |
| `shard_path` | `string`  |
| `row_group`  | `int32`   |
| `row_offset` | `int32`   |
| `t_start_s`  | `float64` |
| `t_end_s`    | `float64` |
| `n_samples`  | `int32`   |

Sorted by `(sample_id, spec_id, channel, chunk_idx)`. Lets the reader resolve `(sample_id, spec_id, channel, t_start, t_end)` to the exact shard row via scalar equality filters. `row_group` identifies which row group to read; `row_offset` is the row's position within that row group.

The exploded shape preserves predicate pushdown on `sample_id` (parquet handles dictionary-encoded scalars far better than list-contains predicates). Index size grows with `sum_over_chunks(num_owners)` — for non-shared chunks (the common case) this equals `num_chunks`, so there's no overhead. For shared chunks the multiplier equals the average ownership count.

### `manifest.json`

```json
{
  "writer_version": "0.1.0",
  "format_version": 1,
  "created_utc": "2026-05-10T14:32:01Z",
  "dataset_id": "ecg_dataset",
  "version": "1.0.0",
  "metadata": { "...full DatasetMetadata snapshot..." },
  "counts": {
    "samples": 1000,
    "annotations_by_task": { "classification": 1000, "labeling": 4000 },
    "signal_chunks": 12000,
    "signal_index_rows": 14500,
    "signal_specs": { "ecg_12lead": { "samples": 1000, "channels": 12 } }
  },
  "files": {
    "samples": "samples.parquet",
    "annotations": ["annotations/task=classification/part-0.parquet", "..."],
    "signals": ["signals/shard-00000.parquet", "..."],
    "signal_index": "signal_index.parquet"
  },
  "checksums": {
    "samples.parquet": "sha256:...",
    "annotations/task=classification/part-0.parquet": "sha256:...",
    "...": "..."
  }
}
```

`signal_chunks` counts unique chunk rows in shards. `signal_index_rows` counts exploded index rows (`>= signal_chunks`; equal when no chunks are shared).

`manifest.json` is the **commit marker**. A version directory without it must be treated as in-progress / invalid by readers.

The full `DatasetMetadata` snapshot inside `metadata` is the canonical home for every `SignalSpec`, `AnnotationSpec`, and `ViewSpec` declared by the connector. Everything else on disk references them by `spec_id` / `task_id` strings only — no duplication.

---

## Lifecycle & atomicity

Atomicity is provided by `manifest.json` as the commit marker. A version directory without it is treated as in-progress / invalid by readers. No staging dir or atomic rename.

1. `__init__` validates `dataset` (dataset_id / version non-empty). Does not touch disk.
2. `__enter__` creates `<root>/<dataset_id>/<version>/`. Raises `FileExistsError` if `manifest.json` already exists there (committed version present).
3. `add_chunk` validates the chunk, splits if it exceeds `chunk_max_bytes`, routes it to the open shard for its `spec_id`, rotates shards when `shard_target_bytes` is exceeded, and emits a `chunk` progress event.
4. `__exit__` (or `close()`):
   1. Finalize all open shards.
   2. Write `samples.parquet`, `annotations/`, `signal_index.parquet` (with explosion).
   3. Write `manifest.json` last — this is the commit.
5. On exception during `__exit__`, `abort()` is called.
6. `abort()` deletes `<root>/<dataset_id>/<version>/` entirely.

---

## Validation

### Per-chunk (in `add_chunk`)

- `chunk.sample_ids` is non-empty.
- For each `sid in chunk.sample_ids`:
  - `sid` exists in `dataset.samples`.
  - That sample declares a `SignalRef` with `spec_id == chunk.spec_id`.
  - `chunk.channel` is in the ref's `channels` (or in the SignalSpec's `channels` if the ref's tuple is empty).
- `chunk.values` is 1-D, finite, non-empty.
- `chunk.sampling_rate_hz > 0`.
- If `chunk.timestamps is not None`: same length as `values`, monotonic non-decreasing.
- For each `sid in chunk.sample_ids`, the `(sid, spec_id, channel, chunk_idx)` tuple has not been seen before. (Ensures the exploded index will have unique keys.)

### At close time

- For each `(sample_id, SignalRef)` in the dataset, at least one chunk arrived for at least one channel of that ref. Missing signals raise; orphan chunks raise.
- Every annotation's `sample_ids` reference samples in the dataset.

Failure during `add_chunk` raises immediately and leaves the writer unusable; `__exit__` will call `abort()`.

---

## Progress reporting

Events are emitted from the writer thread. `progress_cb` must be thread-safe if the engine wraps multiple writers.

| Stage             | `total` semantics      | When emitted                     |
| ----------------- | ---------------------- | -------------------------------- |
| `chunk`           | unknown (`None`)       | per `add_chunk`                  |
| `shard_finalized` | unknown                | per shard rotation + final flush |
| `samples`         | `len(dataset.samples)` | once during finalize             |
| `annotations`     | total annotation count | once per task partition          |
| `index`           | total index rows       | once during finalize             |
| `manifest`        | `1`                    | once just before commit          |
| `commit`          | `1`                    | once after manifest is written   |

Engine maps these to its coarser `DownloadProgressEvent(stage="write", ...)`.

---

## Concurrency

Single-threaded per writer instance. Multiple datasets can be written in parallel by instantiating multiple writers. The engine's `cpu_workers` controls dataset-level parallelism, not within-dataset parallelism.

---

## Read-side contract

The read API lives in `sdk.md` on `Dataset`, but this writer must produce files that satisfy it. Two methods, replacing the placeholder `signal(signal_id: int)` currently in `sdk.md`:

```python
class Dataset:
    def signal(
        self,
        sample_id: str,
        spec_id: str,
        channel: str,
        *,
        t_start_s: float | None = None,
        t_end_s: float | None = None,
    ) -> pd.DataFrame: ...

    def signals(
        self,
        sample_id: str,
        spec_id: str,
        *,
        channels: tuple[str, ...] | None = None,   # None = all channels declared on the sample
        t_start_s: float | None = None,
        t_end_s: float | None = None,
    ) -> pd.DataFrame: ...
```

Lookup path the writer optimizes for:

1. Open `signal_index.parquet`, filter on `(sample_id, spec_id, channel)` and time-range overlap. Scalar equality on `sample_id` works the same whether or not the chunk is shared — the index already exploded the ownership.
2. For each matching index row, open `shard_path` and read **only that row group**.
3. Concatenate `values` in `chunk_idx` order, derive timestamps from `t_start_s + i / sampling_rate_hz` (or use the explicit `timestamps` column).
4. Trim to `[t_start_s, t_end_s]`.

Two writer-side requirements that make this fast:

- `signal_index.parquet` schema as defined above, exploded by `sample_id`.
- Within-shard sort order is `(channel, chunk_idx)` so sequential channel scans are cache-friendly. Per-sample lookup goes through the index, not row-group statistics on the shard's `sample_ids` column.

`Sample` itself stays metadata-only. No `Sample.load_signal()` — the in-memory dataclass must not couple to disk. The split is: `Sample` for metadata, `Dataset` for bytes.

The full read spec belongs in a separate document (reader spec). This section exists only to lock in the addressing scheme the writer must support.

---

## Out of scope (explicitly)

- Reading. Defined separately on `Dataset` in `sdk.md`.
- Append / partial-update. `store()` is one-shot per `(dataset_id, version)`. Re-running rewrites.
- Compaction. No tooling to merge or rewrite shards after the fact.
- Quantization / non-float32 value dtypes. Could be added later; v1 is float32.
- Content-hash deduplication. Sharing is connector-declared, not auto-detected.
- Multi-version coexistence — assumed but not specified here (different `<version>/` directories side-by-side just work).

---

## Decisions log

- **Atomicity**: direct writes to final directory; `manifest.json` written last is the commit marker. No staging dir or atomic rename.
- **Chunk splitting**: byte-based adaptive at 8 MB default (`chunk_max_bytes`). Writer splits transparently; connector and reader are unaware.
- **`iter_signals` signature**: takes `dataset: TimeFDataset` as a second argument so connectors can look up derived `sample_id` values without re-deriving them from `TRaw`.
