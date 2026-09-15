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

TimeF stores one dataset version as one directory. The directory holds a JSON manifest, one embedded
DuckDB database, and the time-series values. This page describes each file, the tables inside the
database, and how the parts link together. It also describes how the writer encodes the values.

The manifest is the contract. A reader loads `manifest.json` first and never runs connector code.
Everything else is addressed from the manifest, not from a directory scan.

## On-disk layout

A version directory looks like this. A [registry](registry.md) addresses it by `org/name` and
version. Its `open_version()` method returns a [`DatasetVersion`](registry.md) handle for the
version: the parsed manifest plus a filesystem-rooted view of its files.

```text
<org>/<name>/<version>/
├── manifest.json   # the contract: metadata, schema, and the file list
├── control.duckdb  # the control plane: records, series, annotations, tasks, chunk locators
└── time_series/    # the values plane (default Parquet backend)
    ├── part-00000000.parquet
    └── part-00000001.parquet
```

The reader does not glob these paths. The manifest's `files` block names the control database and
lists every values part, so the values plane can shard further without a format change.

## Two planes

TimeF splits a dataset into a control plane and a values plane.

The **control plane** is one embedded DuckDB database per version. It holds the records, their
series, the annotations, the tasks and the locators that point into the values plane. It does not
depend on how the values are stored.

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
| `values_backend` | The values plane backend: `parquet` or `zarr`. |
| `value_encoding` | The Parquet values encoding per `spec_type`, for provenance (see [below](#choosing-the-values-encoding)). |
| `build_env` | The interpreter and package set that produced the version, for provenance. |

## The control-plane tables

`control.duckdb` holds every table below. Each entity carries a dense `UINTEGER` surrogate id that
the joins run on, and the id the caller gave it survives beside it as `external_id`. Surrogate ids
are assigned per build, so nothing outside the file quotes one; `external_id` is what a reader hands
back and what stays stable across a rebuild.

The database ships no primary or foreign keys. The writer checks every invariant they would have
enforced once, as a bulk anti-join, inside the load transaction. A version is written once and is
immutable afterwards, so a shipped constraint would re-check something that cannot change. On
ECG-QA, 1.35M control rows, plain tables measure 19.7 MB against 189.0 MB with keys and indexes, and
the write takes 3.9 s against 11.9 s.

### `meta`, `specs`, `annotation_descriptors`

`meta` is a `(key, value)` table holding the control-plane schema version and the dataset id.

`specs` is one row per time-series spec: `spec_type`, `name`, `unit_value`, `dtype`, `categories`,
`value_shape`, `dimension_names`, `nullable`, and the spec's optional data source as three columns.
`annotation_descriptors` is one row per annotation key: `annotation_type`, `value_type`, `unit` and
`description`. Together they are the dataset's type declaration, stored once and referenced by id, so
a million series never repeat a unit and a client that attaches the database over HTTP can ask what a
signal carries without the manifest beside it. The manifest's `schema` block holds the same content
as a projection, so a registry can filter datasets without downloading the database.

### `records`, `time_series`, `record_time_series`, `record_tasks`

`records` is one row per [record](data-model/records.md): `external_id`, `start_time_us`, the
optional session span as `time_span_start_us` and `time_span_end_us`, and the ordered `subject_ids`
list.

`time_series` is one row per distinct series: `external_id`, `signal`, `source_id`, `n_values`, and a
reference to the `specs` and `axes` rows it shares with every other series of the same shape. A
regular axis fills `period_numerator_us`, `period_denominator`, and `start_index`. An irregular axis
fills `first_us` and `last_us` instead.

`record_time_series` links them, with a `position` that keeps a record's series in the order it
declared them. It is a link table rather than a column on `time_series` because one series can be
shared by several records and is then stored once.

`record_tasks` is the reverse of the records a task names: two dense ids per link. An array of task
ids on the record row could not be joined or filtered; this can, from either side.

### `annotations`, `record_annotations`, `dataset_annotations`, `task_annotations`

`annotations` holds each [annotation](data-model/annotations.md) payload once: `external_id`, `key`,
the JSON-encoded `value`, the optional `source`, and its span as `span_start_us`, `span_end_us` and
`span_time_series_ids`. A null `span_start_us` means the annotation has no place in time; a null
`span_end_us` with a start means it is a point.

One attachment table per target kind says what carries it: `record_annotations` for a record,
`dataset_annotations` for an annotation registered for tasks to reference that no record carries, and
`task_annotations` for one a task holds as input or as its answer. A polymorphic
`(object_type, object_id)` table measured 55.7 MB against 50.2 MB for one table per kind on a
3.06M-attachment corpus, and the gather query 25.50 ms against 21.24 ms.

### `values_artifacts`, `time_series_chunks`

`time_series_chunks` is one row per values chunk: the series it belongs to, its `chunk_idx`, the artifact
it landed in, `chunk_major_idx`, `chunk_minor_idx`, and `n_values`. The two locator columns are
backend-neutral. For the Parquet backend, `chunk_major_idx` is the row group in the shard and
`chunk_minor_idx` is the row within that row group. For the Zarr backend, `chunk_major_idx` is the
element-start index in the per-modality array and `chunk_minor_idx` is null.

`values_artifacts` names each file the values plane wrote and which backend wrote it, so a chunk row
carries an id rather than repeating a path and a reader knows what the two indexes mean.

### `tasks`, `task_items`, `task_from_tasks`, `task_fields`, `task_refs`, `task_spans`

`tasks` holds the frame every [task](data-model/tasks.md) shares: `external_id`, `task_type`,
`prompt` and `rationale`. Its `scope` sits in `task_spans` under the field name `scope`.

`task_items` is the task's ordered list of inputs and targets, whatever they are made of:
`(role, position, item_type, text_value, record_id)`, with `role` either `input` or `target` and
`item_type` either `record` or `text`. The records a task is about are its input items; a free-text
answer is its target item. `task_from_tasks` keeps the ids of the tasks it derives from. That is the
one reference stored as the caller's string, because a streamed task can name a parent the version
never stores and the reader has to be able to say which one.

The three payload tables hold what makes each type a different kind of thing. Every payload field
that is set gets one `task_fields` row, whatever its kind, and a plain scalar keeps its value there
(`text_value` or `double_value`). A field holding ids puts its elements in `task_refs`, and one
holding spans puts them in `task_spans`. The `task_fields` row alone says the field is set, which is
what keeps an empty list (a localization that searched and found nothing) distinguishable from an
absent one (an answer stored by reference).

A payload spread over three shared tables cannot say in its column types what each task type holds,
so the writer checks it: every task carries the fields its class requires, in the column its kind
belongs in, and none the class does not declare. A malformed payload fails the write rather than the
read.

## The values plane

The waveform values live outside the record table, in the values plane. The default backend writes
rotating Parquet shards.

### part-00000000.parquet

| Column | Type | Meaning |
| --- | --- | --- |
| `time_series_id` | id | The series the chunk belongs to. |
| `spec_type`, `signal` | string | The series' modality and signal. |
| `chunk_idx` | int32 | The chunk's position within the series. |
| `n_values` | int32 | How many values the chunk holds. |
| `values` | list of the spec dtype | The chunk's values. A `"str"` or `"enum"` chunk stores text. |
| `time_offsets_us` | list of int64 | Per-value time offsets, for an irregular axis. Null for a regular one. |

The writer builds the shards in a fixed order. It dedupes series by `time_series_id`, sorts them by
`(spec_type, signal, time_series_id)`, and streams them through the backend. It splits each series
into chunks of at most `chunk_max_bytes`, buffers chunks until they reach `row_group_target_bytes`,
and flushes them as one row group. A shard rotates once it reaches `shard_target_bytes`. A row group
never spans two shards, so the index locators are exact.

The defaults are a 128 MiB shard target, a 4 MiB row-group target, a 1 MiB chunk limit, and zstd
compression at level 19 (Parquet) or 9 (Zarr).

## How the parts link together

To read one series, the reader joins the record to its bytes through the control plane:

1. Read the record row and walk `record_time_series` to its series.
2. Find that series' chunks in `time_series_chunks`, joined to `values_artifacts` for the file name.
3. For each chunk, open `chunk_file` and go to `chunk_major_idx`, then `chunk_minor_idx`.
4. Read the `values` list and rebuild the series on its time axis.

The reader reads only the manifest when it opens a version. Records come back a batch at a time, and
each batch costs one query per table for the whole batch rather than one query per record. Step 2 is
batched the same way: the first series in a batch that reads its values locates the chunks of every
record in that batch. Values stay lazy, so a large dataset opens without reading a shard.

## Encodings

The writer pins each column's Parquet encoding by its data role, rather than leaving the choice to
pyarrow. Pinned encodings keep a re-built version byte-stable.

| Column role | Encoding |
| --- | --- |
| `values` (waveform floats) | Chosen from the data, per modality (see [below](#choosing-the-values-encoding)). |
| `time_offsets_us` and `chunk_idx` | DELTA_BINARY_PACKED. |
| Bounded categoricals (`spec_type`, `signal`) | Dictionary plus RLE. |
| `time_series_id` | Plain, stored as raw bytes or a string (see [Id storage](#id-storage)). |

Every file also carries column statistics, a page index, and per-page checksums. Every file uses
content-defined chunking, which aligns data pages to content. A re-built or edited version then
re-stores only the pages that changed on a deduplicating backend such as Xet.

### Choosing the values encoding

No single encoding is best for every waveform, so the writer measures the data instead of pinning
one. It samples the values it has already buffered for a modality, counts the distinct values (bit
patterns for floats), and picks:

- **dictionary** at or below **65,536** distinct values,
- **BYTE_STREAM_SPLIT** above that for floats,
- **plain** above that for strings and integers (byte-plane splitting has no
  meaning for these types).

Bool signals always use plain and enum signals always use dictionary, both without measuring
anything. The writer takes one decision per `spec_type`, before that modality's first shard opens.
The decision reads only buffered data, so re-building an unchanged source reaches the same encoding
and writes the same bytes.

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

An entity id is a UUIDv7 string by default. The control plane stores every id as the caller wrote it
and joins on its own dense integer surrogate instead, so no id storage choice arises there. A values
shard carries one id column, `time_series_id`: when every series id is a canonical UUID, the writer
stores it as 16 raw bytes (`binary(16)`) instead of a 36-character string, and the shard's own
Parquet schema records that choice.

## Integrity

The manifest lists a `sha256:` checksum for every file. The writer hashes each file a block at a
time as it commits the version. A reader can hash the files again and compare, so a truncated or
corrupt download fails loudly instead of returning wrong data.

## Related pages

- [TimeFWriter](timef-writer.md) covers the write path and its options.
- [TimeFReader](timef-reader.md) covers the read path.
- [Manifest](manifest.md) documents every manifest field in detail.
