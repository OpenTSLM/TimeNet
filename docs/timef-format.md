---
icon: lucide/binary
description: "The TimeF on-disk format: the files in a dataset version, the control-plane tables, the chunk locator, and how values are encoded."
tags:
  - reference
  - format
  - writer
  - reader
---

# TimeF format

TimeF stores one dataset version as one directory. The directory holds a JSON manifest, one embedded
DuckDB database, and the time-series values. This page describes each file, the tables inside the
database, and how the writer encodes the values.

The manifest is the contract. A reader loads `manifest.json` first, and every file is addressed from
it rather than from a directory scan.

## On-disk layout

A version directory looks like this. A [registry](registry.md) addresses it by `org/name` and
version, and its `open_version()` returns a [`DatasetVersion`](registry.md) handle: the parsed
manifest plus a filesystem-rooted view of the version's files.

```text
<org>/<name>/<version>/
├── manifest.json       # the contract: identity, counts, and the file list
├── control.duckdb      # the control plane: one database per version
└── time_series/        # the values plane (default Parquet backend)
    ├── part-00000000.parquet
    └── part-00000001.parquet
```

A version written with the Zarr backend replaces `time_series/` with a `time_series.zarr/` group.
The control plane is identical either way.

## Two planes

The **control plane** is the structure of the dataset: which records exist, what sources they hold,
which signals those sources produce, what the tasks ask, and what every annotation says. It is
small, deeply cross-referenced, and read by point lookups, joins, and tree walks. That is a
relational workload, so it lives in an embedded database.

The **values plane** is the samples. It is bulk numeric data read by random access at a known
offset, which is not relational, so it stays in files. The `signal_chunks` table bridges the two.

## The manifest

`manifest.json` is a single JSON object. A reader parses it and checks `timef_format_version`
against the versions it supports.

| Key | Meaning |
| --- | --- |
| `timef_format_version` | Format version. A reader rejects a version it does not support. |
| `dataset_id` | The `org/name` id. A copy of `metadata.dataset_id`, readable without parsing metadata. |
| `metadata` | Descriptive identity: name, version, license, domains, tags, source URL. |
| `counts` | Row and entity counts, for quick inspection. |
| `files` | Each artifact's path, `sha256:` checksum, and size. `control_db` is the database; `time_series` is the values plane. |
| `values_backend` | The values plane backend: `parquet` or `zarr`. |
| `value_encoding` | The values-column encoding chosen per `spec_type`, for provenance. |
| `build_env` | The interpreter and package set that produced the version. `timenet.provenance.build_env` collects it; the writer leaves it empty unless a builder fills it in. |

## The control-plane tables

The database holds one dataset. Its id and schema version live in `meta`. Every join runs on a dense
integer surrogate the writer hands out during the walk; the caller's own id survives as
`external_id` on the entity that owns it.

| Table | One row per | Key columns |
| --- | --- | --- |
| `records` | recording session | `record_id`, `external_id`, `start_time_us` |
| `sources` | device or assembly in a record | `source_id`, `record_id`, `parent_source_id`, `path`, `depth`, `position` |
| `signals` | sequence of values | `signal_id`, `external_id`, `name`, `axis_id`, `spec_id`, `n_values` |
| `source_signals` | (source, signal) link | `source_id`, `signal_id`, `position` |
| `axes` | distinct time axis | `axis_id`, `axis_type`, `period_numerator_us`, `period_denominator`, `start_index`, `first_us`, `last_us` |
| `specs` | distinct modality | `spec_id`, `spec_type`, `name`, `unit`, `dtype`, `nullable` |
| `tasks` | task | `task_id`, `external_id`, `prompt` |
| `task_items` | input or target item | `task_id`, `role`, `position`, `item_type`, `text_value`, `record_id` |
| `annotations` | distinct `(name, value, unit)` payload | `annotation_id`, `name`, `value`, `unit` |
| `*_annotations` | one attachment of a payload | `attachment_id`, `annotation_id`, the target id, `span_type`, `start_us`, `end_us` |
| `values_artifacts` | file the values plane wrote | `artifact_id`, `chunk_file`, `backend` |
| `signal_chunks` | chunk of a signal's values | `signal_id`, `chunk_idx`, `artifact_id`, `chunk_major_idx`, `chunk_minor_idx`, `n_values` |

Three shapes in that list are worth stating outright.

**A signal belongs to many sources.** `source_signals` is a link table, not a column on `signals`,
because real corpora reuse one series across many records: 7,013 of ARFBench's 9,187 series are
referenced by more than one. Shared values are then written once and linked many times.

**An annotation's payload is stored once.** `annotations` holds the `(name, value, unit)` triple;
the attachment tables hold where it applies and when. `patient_sex=male` on 100,000 records costs
one payload row and 100,000 short attachment rows.

**An attachment lives in one table per target kind.** There is a table for each of dataset, task,
record, source, and signal, so every target column is `NOT NULL`, there is no discriminator column
to branch on, and a signal lookup never touches the attachment rows of other kinds. Measured on a
3.06M-attachment corpus, that is 50.2 MB against 55.7 MB for a polymorphic `(object_type,
object_id)` pair, and the query that gathers everything applying to one signal in context runs in
21.24 ms against 25.50 ms.

`source_annotations` also carries `scope_record_id`, so "every annotation anywhere in this record" is
an equality rather than a walk of the source tree. `signal_annotations` deliberately has no such
column: a series shared by several records belongs to no single one.

### No keys, no indexes

The shipped database declares no primary keys, no foreign keys, and no indexes. Every invariant they
enforced is checked once against the finished database, as a bulk anti-join, before the writer
publishes. A version is written once and is immutable afterwards, so an invariant that holds at
publish time holds for the rest of its life. `CHECK` constraints stay, because they cost no storage
and catch a bad `role`, `item_type`, `span_type`, or `backend` at insert time.

### Source trees

`parent_source_id` records the edge, and `path` (`0000.0001`) records the ancestry as dot-separated
zero-padded positions. A subtree is then a prefix match and depth-first display order is
`ORDER BY path`, neither of which needs recursion. Materialized paths are a liability under
re-parenting; these trees are written once.

## The chunk locator

A signal's values are split into chunks, and `signal_chunks` says where each one landed:
`artifact_id` names a file in `values_artifacts`, and `chunk_major_idx` plus `chunk_minor_idx`
address a run of values inside it. The two indexes mean whatever the backend that wrote the artifact
says they mean:

- **Parquet**: `chunk_major_idx` is the row group in the shard, `chunk_minor_idx` is the row inside
  that row group.
- **Zarr**: `chunk_major_idx` is the element offset along the per-modality array, and
  `chunk_minor_idx` is null.

Because the artifact declares its own backend, a reader picks the right one without a per-chunk tag,
and a chunk row is checked against a declared artifact at write time.

## The values plane

The default backend writes rotating Parquet shards under `time_series/`.

| Column | Type | Meaning |
| --- | --- | --- |
| `signal_id` | uint32 | The signal the chunk belongs to, as the control plane numbers it. |
| `chunk_idx` | int32 | The chunk's position within the signal. |
| `spec_type`, `signal` | string | The signal's modality and name. |
| `values` | list of the spec dtype | The chunk's values. |
| `time_offsets_us` | list of int64 | Per-value time offsets, for an irregular axis. Null for a regular one. |

A shard carries one modality's chunks, so its `values` column has a single dtype. How many values a
chunk holds is not repeated here; `signal_chunks.n_values` in the control plane already says it.

The writer splits each signal into chunks of at most `chunk_max_bytes`, buffers chunks until they
reach `row_group_target_bytes`, and flushes them as one row group. A shard rotates once it reaches
`shard_target_bytes`. A row group never spans two shards, so a locator is exact. The defaults are a
128 MiB shard target, a 4 MiB row-group target, a 1 MiB chunk limit, and zstd compression.

The Zarr backend appends every series of one modality along time into one chunked typed array, and a
locator is `(array path, element offset)`. Arrays are partitioned by `(spec_type, whether the series
stores time offsets)`, because a series with per-value offsets writes to a values array and a
parallel time-offsets array, and one element offset can only address both if the two always advance
together.

## Encodings

The writer pins each column's Parquet encoding by its data role rather than leaving the choice to
pyarrow. Pinned encodings keep a rebuilt version byte-stable.

| Column role | Encoding |
| --- | --- |
| `values` | Chosen from the data, per modality (see below). |
| `time_offsets_us`, `chunk_idx`, locator ints | DELTA_BINARY_PACKED. |
| Bounded categoricals (`spec_type`, `name`) | Dictionary plus RLE. |

### Choosing the values encoding

No single encoding is best for every waveform, so the writer measures the data instead of pinning
one. It samples the arrays it has already buffered for a modality, counts the distinct values (bit
patterns for floats), and picks:

- **dictionary** at or below **65,536** distinct values,
- **BYTE_STREAM_SPLIT** above that for floats,
- **plain** above that for strings and integers, where byte-plane splitting means nothing.

The decision is taken once per `spec_type`, before that modality's first shard opens, and it reads
only buffered data, so rebuilding an unchanged source reaches the same encoding and writes the same
bytes.

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

The writer records its choice in the manifest under `value_encoding`. That is provenance, not
contract: Parquet records the applied encoding in every file's footer, so a reader decodes a shard
without the manifest.

## Integrity

The manifest lists a `sha256:` checksum for every file, hashed a block at a time as the version is
committed. A reader can hash the files again and compare, so a truncated or corrupt download fails
loudly instead of returning wrong data. The reader opens the database read-only, which writes no
write-ahead log beside it, so a downloaded copy stays byte-identical to what was checksummed.

## Related pages

- [TimeFWriter](timef-writer.md) covers the write path and its options.
- [TimeFReader](timef-reader.md) covers the read path.
- [Manifest](manifest.md) documents every manifest field in detail.
