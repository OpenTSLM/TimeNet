---
icon: lucide/pencil
description: "TimeFWriter: stream a TimeFDataset to the TimeF on-disk format."
tags:
  - reference
  - writer
---

# TimeFWriter

`TimeFWriter` serializes a [`TimeFDataset`](timef-dataset.md) to the TimeF on-disk format. It is a
context manager. It streams shards with bounded memory and commits the result atomically. The class
lives in `timenet.writer`.

```python
from timenet.writer import TimeFWriter

dataset.derive_schema()
with TimeFWriter(root, dataset) as writer:
    writer.write()
# committed at <root>/<dataset_id>/<version>/
```

Most connectors do not use `TimeFWriter` directly. `BaseConnector.store()` and the
[engine](build.md) wrap it.

---

## Disk layout

Every control table shards into numbered parts under its own directory. A small table writes a
single `part-00000000.parquet` file. A large table splits into a new part when a part reaches
`control_shard_target_bytes`. The manifest lists the parts. As a result, the reader discovers the
parts and does not assume fixed names.

```
<root>/<dataset_id>/<version>/
  manifest.json   # written last; its presence marks a committed version
  samples/part-00000000.parquet ...
  annotations/part-00000000.parquet ...
  time_series_index/part-00000000.parquet ...
  tasks/task=<task_type>/part-00000000.parquet ...
  time_series/part-00000000.parquet ...   # values_backend="parquet" (default)
  time_series.zarr/<spec_type>/...      # values_backend="zarr" (alternative)
  time_series.zarr/_irregular/<spec_type>/...   # values of series storing time offsets
  time_series.zarr/_time_offsets/<spec_type>/...    # their int64 time offsets, one per value
```

## Constructor options

| Option | Default | Meaning |
| --- | --- | --- |
| `shard_target_bytes` | 128 MiB | Rotate to a new shard (Parquet), or a new Zarr shard, when the buffered values exceed this size. |
| `control_shard_target_bytes` | 128 MiB | Split a control table (samples, annotations, index, tasks) into a new part when its in-memory size exceeds this. |
| `row_group_target_bytes` | 4 MiB | Flush a row group when the buffered values exceed this size (Parquet only). |
| `chunk_max_bytes` | 1 MiB | Split a series into chunks no larger than this. |
| `compression` | `"zstd"` | Codec for the values (Parquet codec, or Zarr Blosc inner codec). |
| `compression_level` | 3 | Fixed zstd level, for reproducible output. |
| `values_backend` | `"parquet"` | Storage backend for the values plane: `"parquet"` or `"zarr"`. |
| `progress_cb` | `None` | The writer calls this with each `WriteProgressEvent`. |

The writer measures the targets in uncompressed value bytes. On disk, files are smaller because of
zstd compression.

## Values backends

Only the **values plane** (the temporal values of each series) is backend-specific. Samples,
annotations, tasks, and the time-series index are always Parquet. The manifest records the choice in
`values_backend`. The index locates every chunk with a backend-agnostic
`(chunk_file, chunk_major_idx, chunk_minor_idx)` locator:

| Backend | Layout | Chunk locator |
| --- | --- | --- |
| `parquet` (default) | Rotating `time_series/part-*.parquet` files of `list<{dtype}>` rows. Each file also has a `list<int64>` `time_offsets_us` column, null unless the series stores per-value time offsets. One shard carries one modality, so its `values` element type is the spec's dtype. `str` stores text and `enum` stores its category codebook, both with the dictionary encoding. The backend supports scalar values only. | `(shard path, row group, row offset)` |
| `zarr` | One array per `(spec_type, stores_time_offsets)` under `time_series.zarr/`. Each array has the shape `(total_steps, *value_shape)` and the spec dtype. Irregular values sit under `_irregular/`. Their int64 time offsets sit in a parallel array under `_time_offsets/`. A series is **one index row** that spans its time axis. The backend stores every scalar dtype except `str`. | `(array path, step start, –)` |

Each backend chunks the data in its own way. Parquet needs the logical `chunk_max_bytes` split to
pack series into row groups. Zarr chunks the storage itself. As a result, its index carries one
placement per series. The index splits a series only past 2³⁰ values, to keep `n_values` in int32.
The Zarr writer buffers appends per `(spec_type, stores_time_offsets)` partition. It flushes at
shard-aligned boundaries. As a result, the writer writes each shard object exactly once. It does not
read, change, and write the object again for each series.

The Zarr backend needs the `zarr` extra (`pip install 'timenet[zarr]'`). The core package never
imports the extra. A Zarr series can hold embeddings, pose tensors, spectrogram frames, or image
sequences. Recordings can have different durations. Every series that shares a `spec_type` must have
the same dtype and trailing shape. Parquet deliberately rejects N-D specs, which stay on Zarr. The
Parquet backend stores scalar values of every spec dtype: `float32`/`float64`, the integer types,
`bool`, `str`, and `enum`. The Zarr backend stores every scalar dtype **except `str` and `enum`**: it
has no dictionary layer, so string-like channels inflate on disk and read slowly, and the writer
rejects them. A dataset's
values plane uses **one** backend for the whole dataset (the manifest's single `values_backend`
field), so a dataset with a `str` or `enum` channel must be entirely Parquet, and a dataset needing
N-D tensors must be entirely Zarr. They cannot be mixed per channel.
A [copy-on-write edit](#copy-on-write-edits) keeps the backend of
the base version, unless overridden.

## Streaming and chunking

`write()` removes duplicate series by `time_series_id`. The writer calls the loader of each unique
series exactly once. `write()` sorts the series by `(spec_type, channel, time_series_id)`. Then it
streams the series through the values backend.

With Parquet, the writer splits each series into chunks of at most `chunk_max_bytes`. It buffers the
chunks until they reach `row_group_target_bytes`, then flushes them as one row group. Shards rotate at
`shard_target_bytes` and at every `spec_type` boundary, so each shard holds one modality. A row group
never spans shards. As a result, the `(chunk_file, chunk_major_idx, chunk_minor_idx)` pointers in the
index are exact.

The 2³¹ element cap on a row group comes from the 32-bit offsets of the `values` list column, so the
cap does not depend on the element type. The byte-based flush keeps the row group well under it.

## Encodings

The writer pins each encoding by data role. It does not leave the choice to pyarrow heuristics. As a
result, re-built versions stay stable:

- `time_offsets_us.list.element` -> **DELTA_BINARY_PACKED** + zstd, for the irregular series that
  carry one time offset per value. A monotonic stream stores as small deltas instead of full int64
  values. The same read-back self-check verifies this encoding.
- monotonic ints (`chunk_idx`, `chunk_major_idx`, `chunk_minor_idx`) -> DELTA_BINARY_PACKED.
- bounded categoricals (`spec_type`, `channel`, `key`, `target`, `chunk_file`)
  -> dictionary + RLE.
- id columns -> plain. The writer stores an id column as **`binary(16)`** when every value in the
  space for that id is a canonical UUID (see below). Otherwise, it stores the column as a UTF-8
  string.
- `values.list.element` -> measured, not pinned (see below).

The writer turns on `write_statistics`, `write_page_index`, `write_page_checksum`, and
**`use_content_defined_chunking`** for every file. Content-defined chunking aligns data pages to
content. As a result, a re-built or [edited](#copy-on-write-edits) version re-stores only the
changed chunks, on a deduplicating backend such as Xet. The reader treats the files as ordinary
Parquet.

### Values encoding

No single encoding is right for every waveform. As a result, the writer measures the data instead of
pinning one encoding. The table below shows real measurements from real sources, at zstd level 3, for
the values column only:

| Encoding | PTB-XL ECG (quantized, 11k distinct in 69M) | TSQA (continuous, 10.5M distinct in 11.5M) |
| --- | --- | --- |
| `dictionary` | **71.1 MB** | 42.9 MB |
| `plain` | 85.1 MB | 42.3 MB |
| `byte_stream_split` | 127.8 MB | **37.6 MB** |

BYTE_STREAM_SPLIT transposes each float into four byte planes and compresses each plane apart. It
wins on smooth, high-cardinality signals, where the sign and high-mantissa planes stay nearly
constant. It loses badly on quantized data. There, the low mantissa byte is noise, and the split
isolates this noise into an incompressible plane. Dictionary wins instead on quantized data. A
physical conversion onto a fixed grid leaves only a few thousand distinct values behind tens of
millions of samples. Examples include the 0.001 mV step of wfdb and an integer ADC scale.

The rule uses cardinality, counted on a sample. If the sample has at most 65,536 distinct values, the
writer selects `dictionary`. If it has more, the writer selects `byte_stream_split`. The writer never
selects `plain` automatically. The sample is the values already buffered for the first row group of a
modality. As a result, the decision costs only a distinct-value count, with no extra reads. The writer
makes one decision per `spec_type`, before it opens the first shard for that type. The decision is
deterministic in the data. As a result, re-building an unchanged source reaches the same encoding.

The manifest records the choice as `value_encoding`, a `spec_type` -> encoding map. This record is
provenance, not a contract. Parquet records the applied encoding in the footer of every file. As a
result, a reader resolves the encoding without the manifest.

The `TimeFWriter(value_encoding=...)` argument controls this choice. It defaults to `auto`, the
measured selection described above. You can set a concrete value, `dictionary`,
`byte_stream_split`, or `plain`, to force one encoding for every modality. High-cardinality
*quantized* data, such as a 24-bit integer-scaled signal, needs a forced encoding. This kind of data
suits neither branch. It has too many distinct values for a dictionary and too much low-bit noise for
a byte split. Only the Parquet backend applies an encoding. The writer rejects a forced encoding when
you write another backend.

```python
with TimeFWriter(root, dataset, value_encoding="plain") as writer:
    writer.write()
```

The writer logs the decision for each modality at `INFO`, under the
`timenet.values_backends.parquet.writer` logger. The log includes the chosen encoding. For `auto`, it
also includes the sampled cardinality behind the choice. This is standard-library logging with no
handler attached. You must configure a handler to see the output. The same decisions also appear in
the manifest.

A read-back self-check verifies that the selection reached every shard. This check exists because
pyarrow silently drops a column encoding when the column path does not match. Parquet has its own
dictionary-to-plain fallback. This fallback happens when a dictionary outgrows its page limit. It is
lossless, and the writer permits it.

### Id storage

Entity ids default to a **UUIDv7** string (`timenet.types.new_id`). UUIDv7 is time-ordered. The
writer already sorts series by id, so this ordering clusters values by creation time and compresses
their shared prefix.

For each of the six logical ids (`sample_id`, `time_series_id`, `annotation_id`, `task_id`,
`source_id`, `subject_id`), the writer verifies whether every value is a canonical UUID. If every
value is a canonical UUID, the writer stores the columns for that id as 16 raw bytes (`binary(16)`)
instead of a 36-char string. It also records `"<id>": "uuid16"` in the `id_encoding` map of the
manifest. Connector-supplied non-UUID ids, for example `ecgqa-test-0`, stay strings. The reader
decodes `binary(16)` back to the canonical string, so callers always see string ids.

## Validation

Intrinsic checks for each sample, annotation, and task happen at insertion (see
[TimeFDataset](timef-dataset.md)). The writer adds two more checks. Each check raises
`TimeFValidationError`:

- Cross-sample check, before any I/O: annotations that share an `id` across samples must have equal
  fields.
- Per-series check, while each loader runs. The values of a series must meet these conditions:
    - The values are not empty.
    - The values match the `dtype` and `value_shape` of the spec.
    - The values are finite, when their dtype supports non-finite values.
    - `len(values) == n_values`. Every series must declare `n_values`.

## Commit protocol

The writer stages everything in `<version>.tmp-<uuid>/`. `close()` writes `manifest.json` last. It
then publishes the version with a single atomic `os.replace` call to `<version>/`. `__enter__` raises
`FileExistsError` if a committed `manifest.json` already exists. If any failure occurs, the context
manager calls `abort()`. `abort()` removes only the staging directory, so a partial dataset is never
visible.

## Manifest

The writer assembles the [manifest](manifest.md) from these parts:

- `dataset.schema`
- write-time counts
- the complete file list
- a per-file SHA-256 checksum for every staged artifact, including Zarr metadata and chunks
- the values-backend tag
- the `id_encoding` map

A copy-on-write edit also records a `derived_from` lineage block.

## Copy-on-write edits

A committed version is immutable. As a result, removing a row means writing a **new** version
without that row.

`timenet.dataset.edit.edit_version(base_dir, out_root, *, dataset_version, remove_sample_ids=(),
cascade=False)` performs the edit. It reads the base version into memory. Values stay lazy and pull
from the base shards. It applies the removals, repairs every cross-reference, and writes a fresh
version through the normal atomic-commit writer.

Ids are stable and never reused. As a result, surviving references stay valid without renumbering.
With content-defined chunking, the rewrite re-stores only the changed chunks.

The writer enforces referential integrity *before* the write. It never filters on read. As a result,
a committed version is always consistent.

Removing a sample strips the id of that sample from the `sample_ids` list of every task. It also
drops task ids that the surviving samples can no longer resolve. A task can lose a **required**
reference: a forecasting `target_sample_id` or `context_sample_ids`, its last remaining sample, or a
`from_task` edge to a removed task. When this happens, the edit fails with `TimeFEditError`, unless
`cascade=True`. With `cascade=True`, the edit removes the invalidated dependents transitively.

The `derived_from` field in the new manifest records the base version and the operation.

---

See the [API reference for `timenet.writer`](api/writer.md) for the full symbol listing.
