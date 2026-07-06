---
icon: lucide/pencil
description: "TimeFWriter: stream a TimeFDataset to the TimeF on-disk format."
tags:
  - reference
  - writer
---

# TimeFWriter

Serializes a [`TimeFDataset`](timef-dataset.md) to the TimeF on-disk format. A context manager that
streams shards with bounded memory and commits atomically. Lives in `timenet.writer`.

```python
from timenet.writer import TimeFWriter

dataset.derive_schema()
with TimeFWriter(root, dataset) as writer:
    writer.write()
# committed at <root>/<dataset_id>/<version>/
```

Most connectors don't use `TimeFWriter` directly — `BaseConnector.store()` and the
[engine](engine.md) wrap it.

---

## Disk layout

```
<root>/<dataset_id>/<version>/
  manifest.json            # written last; its presence marks a committed version
  samples.parquet
  annotations.parquet
  time_series_index.parquet
  tasks/task=<task_type>/part-0.parquet
  time_series/shard-00000.parquet ...
```

## Constructor options

| Option | Default | Meaning |
| --- | --- | --- |
| `shard_target_bytes` | 128 MiB | Rotate to a new shard once a shard's buffered values exceed this. |
| `row_group_target_bytes` | 4 MiB | Flush a row group once buffered values exceed this. |
| `chunk_max_bytes` | 1 MiB | Split a series into chunks no larger than this. |
| `compression` | `"zstd"` | Parquet codec. |
| `compression_level` | 3 | Pinned level (zstd) for reproducible output. |
| `progress_cb` | `None` | Called with each `WriteProgressEvent`. |

Targets are measured in **uncompressed value bytes**; on disk (zstd) files are smaller.

## Streaming and chunking

`write()` dedupes series by `time_series_id` (each unique series' loader is called exactly once), sorts
them by `(spec_type, channel, time_series_id)`, then streams: each series is split into chunks of at most
`chunk_max_bytes`, chunks are buffered until `row_group_target_bytes` and flushed as one row group, and
shards rotate at `shard_target_bytes`. A row group never spans shards, so the index's
`(shard_path, row_group, row_offset)` pointers are exact. A hard invariant caps a row group at 2³¹
values (`list<float32>` uses 32-bit offsets); the byte-based flush keeps it well under.

## Encodings

Pinned by data role, not left to pyarrow heuristics, so re-curated versions stay stable:

- `values.list.element` → **BYTE_STREAM_SPLIT** + zstd (verified applied via a read-back self-check).
- monotonic ints (`chunk_idx`, `row_group`, `row_offset`) → DELTA_BINARY_PACKED.
- bounded categoricals (`spec_type`, `channel`, `view`, `key`, `annotation_type`, `label`, `shard_path`)
  → dictionary + RLE.
- id columns → plain, but stored as **`binary(16)`** when every value in the id's space is a canonical
  UUID (see below), otherwise as a UTF-8 string.

Every file is written with `write_statistics`, `write_page_index`, `write_page_checksum`, and
**`use_content_defined_chunking`** on. Content-defined chunking aligns data pages to content so a
re-curated or [edited](#copy-on-write-edits) version re-stores only the chunks that changed on a
deduplicating backend (e.g. Xet); the reader treats the files as ordinary Parquet.

### Id storage

Entity ids default to a **UUIDv7** string (`timenet.types.new_id`), time-ordered so sorting by id — which
the writer already does — clusters values by creation time and compresses their shared prefix. For each
of the six logical ids (`sample_id`, `time_series_id`, `annotation_id`, `task_id`, `source_id`,
`subject_id`) the writer checks whether every value is a canonical UUID; if so it stores that id's columns
as 16 raw bytes (`binary(16)`) instead of a 36-char string and records `"<id>": "uuid16"` in the
manifest's `id_encoding`. Connector-supplied non-UUID ids (e.g. `ecgqa-test-0`) stay strings. The reader
decodes `binary(16)` back to the canonical string, so callers always see string ids.

## Validation

Intrinsic per-sample/annotation/task checks happen at insertion (see [TimeFDataset](timef-dataset.md)).
The writer adds two checks, raising `TimeFValidationError`:

- **Cross-sample** (before any I/O): annotations sharing an `id` across samples must be field-equal.
- **Per-series** (as each loader runs): values are a non-empty, finite `float32` array; when `t_end_s`
  is set, `len(values) == round((t_end_s - t_start_s) * sampling_rate_hz)`.

## Commit protocol

Everything is staged in `<version>.tmp-<uuid>/`; `close()` writes `manifest.json` last, then publishes
with a single atomic `os.replace` to `<version>/`. `__enter__` raises `FileExistsError` if a committed
`manifest.json` already exists. On any failure the context manager calls `abort()`, which removes only
the staging directory, so a partial dataset is never visible.

## Manifest

The writer assembles the [manifest](manifest.md) from `dataset.schema`, write-time counts, the file
list, a per-file sha256 checksum for every parquet artifact, and the `id_encoding` map. A copy-on-write
edit also records a `derived_from` lineage block.

## Copy-on-write edits

A committed version is immutable, so removing a row means writing a **new** version with the row gone.
`timenet.dataset.edit.edit_version(base_dir, out_root, *, dataset_version, remove_sample_ids=(),
cascade=False)` reads the base version into memory (values stay lazy, pulled from the base shards),
applies the removals, repairs every cross-reference, and writes a fresh version through the normal
atomic-commit writer. Because ids are stable and never reused, surviving references stay valid without
renumbering, and with content-defined chunking the rewrite re-stores only the chunks that changed.

Referential integrity is enforced *before* the write (never filtered on read), so a committed version is
always consistent. Removing a sample strips its id from every task's `sample_ids` and drops task ids the
surviving samples can no longer resolve. A task that would lose a **required** reference (a forecasting
`target_sample_id` / `context_sample_ids`, its last remaining sample, or a `from_task` edge to a removed
task) makes the edit fail with `TimeFEditError` unless `cascade=True`, which removes the invalidated
dependents transitively. The new manifest's `derived_from` records the base version and the operation.

---

See the [API reference for `timenet.writer`](api/writer.md) for the full symbol listing.
