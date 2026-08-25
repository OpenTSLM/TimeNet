---
icon: lucide/binary
description: "The TimeF on-disk format: the files in a dataset version, the columns in each, how ids tie them together, and how the values are encoded."
tags:
  - reference
  - format
  - writer
  - reader
---

# TimeF format

TimeF stores one dataset version as one directory. The directory holds a JSON manifest, a few
Parquet tables, and the time-series values. This page describes each file, the columns inside it,
and how the parts link together. It also describes how the writer encodes the values.

The manifest is the contract. A reader loads `manifest.json` first and never runs connector code.
Everything else is addressed from the manifest, not from a directory scan.

## On-disk layout

A version directory looks like this. A [registry](registry.md) addresses it by `org/name` and
version. Its `open_version()` method returns a [`DatasetVersion`](registry.md) handle for the
version: the parsed manifest plus a filesystem-rooted view of its files.

```text
<org>/<name>/<version>/
├── manifest.json       # the contract: metadata, schema, and the file list
├── samples/            # one row per sample (the control plane)
│   └── part-00000000.parquet
├── annotations/        # one row per annotation
│   └── part-00000000.parquet
├── time_series_index/  # locates every values chunk
│   └── part-00000000.parquet
├── tasks/              # one partition directory per task type
│   ├── task=classification/part-00000000.parquet
│   └── task=answer/part-00000000.parquet
└── time_series/        # the values plane (default Parquet backend)
    ├── part-00000000.parquet
    └── part-00000001.parquet
```

The reader does not glob these paths. The manifest's `files` block lists every part, so a table can
shard into more parts later without a format change.

## Two planes

TimeF splits a dataset into a control plane and a values plane.

The **control plane** is the sample, annotation, task, and index tables. These are always Parquet,
and they do not depend on how the values are stored.

The **values plane** is the typed waveform of every series. This is the one part whose storage is
swappable. The manifest's `values_backend` field names the backend: `parquet` (the default,
rotating shards) or `zarr` (a chunked array store). The control plane stays the same either way.

## The manifest

`manifest.json` is a single JSON object. A reader parses it, checks `timef_format_version` against
the versions it supports, and rebuilds the dataset types from the flat descriptors it holds.

| Key | Meaning |
| --- | --- |
| `timef_format_version` | Format version. A reader rejects a version it does not support. |
| `dataset_id` | The `org/name` id. A copy of `metadata.dataset_id`, readable without parsing metadata. |
| `metadata` | Descriptive identity: name, version, license, domains, tags, source URL. |
| `schema` | Structural schema: the time-series specs, the annotation descriptors, and the task types. |
| `counts` | Row and entity counts, for quick inspection. |
| `files` | Each artifact's path, `sha256:` checksum, and size, grouped by kind. |
| `id_encoding` | Which id columns are stored as raw bytes (see [Id storage](#id-storage)). |
| `values_backend` | The values plane backend: `parquet` or `zarr`. |
| `value_encoding` | The Parquet values encoding per `spec_type`, for provenance (see [below](#choosing-the-values-encoding)). |
| `derived_from` | Copy-on-write lineage, or `null` for a freshly built version. |

## The control-plane tables

Every table is Parquet. The column layout is fixed. Only the id columns change type, and
`id_encoding` records that choice.

### samples/part-00000000.parquet

One row per [sample](data-model/samples.md). A sample groups the series that were recorded
together, plus the tasks and annotations that point at them.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | id | The sample's id. |
| `start_time_us` | int64 | Wall-clock start, in microseconds since the Unix epoch. |
| `subject_ids` | list of id | The subjects the sample belongs to. |
| `time_series` | list of struct | The series in this sample, with their metadata and time axis. |
| `task_ids` | list of id | The tasks that reference this sample. |
| `annotation_ids` | list of id | The annotations that reference this sample. |

The `time_series` struct carries each series' identity and its time axis. It holds `spec_type`,
`channel`, `source_id`, `time_series_id`, an `axis_type`, and `n_values`. A regular axis fills
`period_numerator_us`, `period_denominator`, and `start_index`. An irregular axis fills
`first_time_offset_us` and `last_time_offset_us` instead. The struct does not hold the values
themselves.

### annotations/part-00000000.parquet

One row per [annotation](data-model/annotations.md).

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | id | The annotation's id. |
| `key` | string | The annotation key. |
| `value` | string | The value, JSON-encoded. Null for a pure marker. |
| `span` | struct | The time span the annotation covers. Null for a static annotation. |
| `sample_ids` | list of id | The samples the annotation applies to. |

A span struct holds `start_us`, `end_us`, and the `time_series_ids` it is scoped to. A null `end_us`
means the span is a point at `start_us`.

### time_series_index/part-00000000.parquet

One row per values chunk. This table is the join between a series and its bytes on disk. A series is
split into chunks, and each chunk gets one row here.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | id | The sample the chunk belongs to. |
| `time_series_id` | id | The series the chunk belongs to. |
| `spec_type`, `channel` | string | The series' modality and channel. |
| `chunk_idx` | int32 | The chunk's position within the series. |
| `chunk_file` | string | The values file that holds the chunk. |
| `chunk_major_idx` | int64 | Coarse locator inside `chunk_file`. |
| `chunk_minor_idx` | int64 | Fine locator inside `chunk_file`. |
| `n_values` | int32 | How many values the chunk holds. |

The two locator columns are backend-neutral. For the Parquet backend, `chunk_major_idx` is the row
group in the shard and `chunk_minor_idx` is the row within that row group. For the Zarr backend,
`chunk_major_idx` is the element-start index in the per-modality array and `chunk_minor_idx` is null.

### tasks/task=&lt;type&gt;/part-00000000.parquet

Tasks are partitioned by type, one directory per [task type](data-model/tasks.md). Every partition
shares a common set of columns: `id`, `sample_ids`, `from_task_ids`, `prompt`, `scope`,
`input_annotation_ids`, `target_annotation_ids`, and `rationale`. Each type then adds its own payload
columns. A classification task adds `target` and `target_schema`. A scalar prediction adds `target`,
`unit`, and `target_name`. A temporal localization adds a list of span structs.

## The values plane

The waveform values live outside the sample table, in the values plane. The default backend writes
rotating Parquet shards.

### part-00000000.parquet

| Column | Type | Meaning |
| --- | --- | --- |
| `time_series_id` | id | The series the chunk belongs to. |
| `spec_type`, `channel` | string | The series' modality and channel. |
| `chunk_idx` | int32 | The chunk's position within the series. |
| `n_values` | int32 | How many values the chunk holds. |
| `values` | list of the spec dtype | The chunk's values. A `"str"` chunk stores text. |
| `time_offsets_us` | list of int64 | Per-value time offsets, for an irregular axis. Null for a regular one. |

The writer builds the shards in a fixed order. It dedupes series by `time_series_id`, sorts them by
`(spec_type, channel, time_series_id)`, and streams them through the backend. It splits each series
into chunks of at most `chunk_max_bytes`, buffers chunks until they reach `row_group_target_bytes`,
and flushes them as one row group. A shard rotates once it reaches `shard_target_bytes`. A row group
never spans two shards, so the index locators are exact.

The defaults are a 128 MiB shard target, a 4 MiB row-group target, a 1 MiB chunk limit, and zstd
compression at level 3.

## How the parts link together

Ids tie the whole dataset together. Six logical ids cross-reference the entities: `sample_id`,
`time_series_id`, `annotation_id`, `task_id`, `source_id`, and `subject_id`. A sample lists the ids
of its series, tasks, annotations, and subjects. A task and an annotation each list the sample ids
they point at.

To read one series, the reader joins the sample to its bytes through the index:

1. Read the sample row and take a `time_series_id` from its `time_series` list.
2. Find that series' chunks in the `time_series_index` table.
3. For each chunk, open `chunk_file` and go to `chunk_major_idx`, then `chunk_minor_idx`.
4. Read the `values` list and rebuild the series on its time axis.

The reader loads the manifest, tasks, annotations, and index eagerly. It keeps the values and the
sample objects lazy, so a large dataset opens without reading every shard.

## Encodings

The writer pins each column's Parquet encoding by its data role, rather than leaving the choice to
pyarrow. Pinned encodings keep a re-built version byte-stable.

| Column role | Encoding |
| --- | --- |
| `values` (waveform floats) | Chosen from the data, per modality (see [below](#choosing-the-values-encoding)). |
| `time_offsets_us`, `chunk_idx`, and the index locator ints | DELTA_BINARY_PACKED. |
| Bounded categoricals (`spec_type`, `channel`, `axis_type`, `key`, task labels) | Dictionary plus RLE. |
| Id columns | Plain, stored as raw bytes or a string (see [Id storage](#id-storage)). |

Every file also carries column statistics, a page index, and per-page checksums. Every file uses
content-defined chunking, which aligns data pages to content. A re-built or edited version then
re-stores only the pages that changed on a deduplicating backend such as Xet.

### Choosing the values encoding

No single encoding is best for every waveform, so the writer measures the data instead of pinning
one. It samples the values it has already buffered for a modality, counts the distinct values (bit
patterns for floats), and picks:

- **dictionary** at or below **65,536** distinct values,
- **BYTE_STREAM_SPLIT** above that.

`plain` is never chosen automatically. It stays reachable as a manual override. The writer takes one
decision per `spec_type`, before that modality's first shard opens. The decision reads only buffered
data, so re-building an unchanged source reaches the same encoding and writes the same bytes.

The rule follows the measurements. On real data, at zstd level 3, the values column measures:

| Encoding | PTB-XL ECG (quantized, ~11k distinct in 69M) | TSQA (continuous, 10.5M distinct in 11.5M) |
| --- | --- | --- |
| `dictionary` | **71.1 MB** | 42.9 MB |
| `plain` | 85.1 MB | 42.3 MB |
| `byte_stream_split` | 127.8 MB | **37.6 MB** |

BYTE_STREAM_SPLIT transposes each float into four byte planes and compresses each plane apart. It
wins on smooth, high-cardinality signals, where the sign and high-mantissa planes are near constant.
It loses on quantized data, where the low mantissa byte is noise the split isolates into an
incompressible plane. Dictionary wins there, because a physical conversion onto a fixed grid (a
0.001 mV step, an integer ADC scale) leaves only a few thousand distinct values behind tens of
millions of samples.

The writer records its choice in the manifest under `value_encoding`, a `spec_type` to encoding
map. That is provenance, not contract. Parquet already records the applied encoding in every file's
footer, so a reader decodes a shard without the manifest.

One case needs a manual override: high-cardinality quantized data (say a 24-bit integer-scaled
signal) suits neither branch. To force an encoding, set `value_encoding` on the
[dataset card](timef-dataset.md), or pass it to the [writer](timef-writer.md):

```python
with TimeFWriter(root, dataset, value_encoding="dictionary") as writer:
    writer.write()
```

### Id storage

An entity id is a UUIDv7 string by default. When every value in an id's space is a canonical UUID,
the writer stores that column as 16 raw bytes (`binary(16)`) instead of a 36-character string. It
records the choice in the manifest's `id_encoding` map. A reader decodes the bytes back to the
canonical string, so a caller always sees a string id.

## Integrity

The manifest lists a `sha256:` checksum for every file. The writer hashes each file a block at a
time as it commits the version. A reader can hash the files again and compare, so a truncated or
corrupt download fails loudly instead of returning wrong data.

## Related pages

- [TimeFWriter](timef-writer.md) covers the write path and its options.
- [TimeFReader](timef-reader.md) covers the read path.
- [Manifest](manifest.md) documents every manifest field in detail.
