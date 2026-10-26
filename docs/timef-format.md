---
icon: lucide/database
description: "The TimeF DuckDB control plane and lazy values plane."
tags:
  - reference
  - format
---

# TimeF format

TimeF separates relational structure from large signal arrays. One immutable DuckDB database
stores the hierarchy, relationships, annotations, axes, and chunk locations. Parquet or Zarr stores
the signal values. Large text fields and irregular time offsets use packed Parquet payloads.

## Directory layout

A Parquet-backed version looks like this:

```
org/name/version/
├── manifest.json
├── control.duckdb
├── time_series/
│   ├── part-00000000.parquet
│   └── ...
└── payloads/
    ├── part-00000000.parquet
    └── ...
```

A Zarr-backed version uses `time_series.zarr/` instead of Parquet value shards. The manifest lists
every artifact with its byte size and SHA-256 checksum. Readers do not discover files with a glob.

## Object hierarchy

The in-memory hierarchy is:

```
Task ──N:M── Record ──1:N── Source ──1:N── Signal ──N:1── TimeAxis
                                  └──1:N── Source
```

`Source` is recursive. A `Record` can therefore represent a simple sensor or a machine containing
subsystems and sensor packages. A `Signal` is a leaf and has exactly one owning `Source`. Several
Signals can reference the same immutable `TimeAxis`.

## DuckDB control tables

`control.duckdb` contains these authoritative tables:

| Table | Purpose |
| --- | --- |
| `records` | Recording sessions and session-level timing metadata. |
| `clocks` | Record start times, shared by records that use the same `TimeOrigin`. |
| `sources` | Recursive sources, linked to their record and parent source by ID. |
| `signals` | Signal identity plus inline `TimeSeriesSpec` fields. |
| `axes` | Regular, irregular, and ordinal axis definitions. |
| `external_values` | File and row locations for external text and irregular time offsets. |
| `tasks` | Concrete task type, typed scope columns, and subclass configuration columns. |
| `task_targets` | Ordered inline targets, one typed row each. |
| `task_record_refs` | Ordered task-to-record object relationships. |
| `task_annotation_refs` | Ordered task-to-annotation occurrence relationships. |
| `task_dependencies` | Ordered task derivation relationships. |
| `annotation_contents` | Reusable annotation content with one typed value column per value shape. |
| `annotation_occurrences` | One attachment of content to an object, with its span. |
| `signal_chunks` | Locations of Signal values in Parquet or Zarr. |

Object relationships are normalized rather than embedded in JSON. DuckDB can follow the same row
in either direction, so TimeF does not store duplicate reverse-index tables.

Every table except `clocks` uses IDs as keys. Every relationship stores the ID that it refers to.
The only internal key is `clocks.clock_id`, and `records.clock_id` refers to it. The writer numbers
clocks from 1 on each write. As a result, the value has no meaning outside one file, and the Python
API does not expose it.

Public IDs (`record_id`, `signal_id`, and `task_id`) and the writer's storage IDs (`source_id`,
`axis_id`, `content_id`, and `occurrence_id`) are positive int64 values stored as `BIGINT`. The
writer numbers storage IDs from 1 on every write, like clocks. Each ID is unique within its own
kind, so a Record and a Signal can have the same ID. The `dataset_id` is the exception: it is the
`org/name` string, stored as `VARCHAR`. Fixed-shape fields are typed columns, so SQL can filter
spans, task configuration, target values, and annotation values. Only the free-form `metadata`
dictionaries are JSON.

The `control_metadata` table stores the last public ID used for each kind, under
`last_id.Record`, `last_id.Signal`, and `last_id.Task`. A version edited from this one continues
from these values. As a result, it never reuses the ID of a removed object.

## Signals and axes

Each `signals` row identifies its owning Source and axis. The fixed `TimeSeriesSpec` attributes are
stored inline:

```text
signal_id, source_id, name, axis_id, spec_type, spec_name, unit, dtype,
categories, value_shape, dimension_names, nullable, n_values, metadata,
modality
```

There is no separate specification table. The fields are small, typed, directly queryable, and
compress well in DuckDB. `categories` and `dimension_names` are `VARCHAR[]` and `value_shape` is
`BIGINT[]`. The reader reconstructs one `TimeSeriesSpec` object per distinct combination and shares
it between the Signals that use it.

Regular axes store a rational microsecond period and origin. Ordinal axes need only their type.
Irregular axes store their endpoints in `axes` and their ordered microsecond time offsets in
packed Parquet files. The `external_values` table locates each array. Signals that share an axis
reference the same `axis_id`, so the time offsets are stored once for that timeline.

## External values

The writer keeps text and string lists inline when their encoded size is at most 4,096 bytes.
Larger values and the time offsets of every irregular axis go into Zstandard-compressed Parquet
files under `payloads/`. This storage is independent of the Signal values backend. Each file
targets about 4 MiB of encoded input, but one large value can exceed that target. Each row of a
payload file fills exactly one of its three columns, `text` (string), `texts` (list of non-null
strings), or `time_offsets` (list of non-null int64), and readers reject a file with any other
schema.

The `external_values` table identifies the owner, field, ordered position, file, and row for each
value, and records its value kind and length. Like an annotation occurrence, it names its owner by
kind and ID: `object_kind` is `Task` with the public `task_id`, `Content` with the `content_id` of
the annotation content row that holds the value, or `Axis` with the `axis_id` of an irregular axis.
Content that several occurrences share is packed once, and so are the time offsets of an axis that
several Signals share. An irregular axis's time offsets use the field and value kind
`time_offsets` and sit in the payload column of that name. For them, `value_digest` also records
the SHA-256 digest of the time offsets as contiguous little-endian int64 values, stored as
`sha256:<hex>`, so a writer can compare a stored axis with other time offsets without reading it.
`source_key` is `NULL` when this release stores the payload file, and otherwise names the payload
source that does; see [Shared payloads](#shared-payloads). The inline column keeps a placeholder: an
empty string for text and an empty list for a string list. An empty inline value alone therefore
does not mean empty text; readers look up `external_values` first. Direct SQL queries over inline
columns do not resolve external values. The public Arrow table APIs return the complete text.

## Annotations

Subject IDs use a list-valued record annotation with the key `subject_ids`. DuckDB can filter
records by subject through the annotation tables:

```sql
SELECT DISTINCT r.record_id
FROM records r
JOIN annotation_occurrences o
  ON o.object_type = 'Record' AND o.object_id = r.record_id
JOIN annotation_contents c USING (content_id)
WHERE c.name = 'subject_ids'
  AND list_contains(c.text_list_value, 'patient-1');
```

An annotation has reusable content and one or more occurrences. `annotation_contents` stores the
content once: its key (`name`), unit, description, metadata, and value. The dedicated `description`
column is separate from metadata, which can contain its own `description` key. The value sits in
one typed column selected by `value_kind`: `text_value`, `integer_value`, `float_value`,
`boolean_value`, or `text_list_value`
for a list of strings such as a target vocabulary. A marker annotation, which only places a span,
has `NULL` in all of them. `annotation_occurrences` says where the content applies. It points at
the content through `content_id`. It also carries the span as `span_type`, `start_us`, and
`end_us`, plus provenance, confidence, and occurrence metadata.

An occurrence can annotate a Dataset, Task, Record, Source, or Signal. It names its owner with
`object_type` and `object_id`. For a Record, Signal, or Task, `object_id` is the public ID of the
owner. For a Source, it is the storage ID of the Source. `object_id` is `NULL` exactly when the
owner is the Dataset. IDs are unique only within one kind, so the type is part of the reference.

A read of every Record or every Task rejects an occurrence with an unknown owner type or a missing
owner. It does not drop that occurrence.

One content row can apply to many objects without copying a long value. Each attachment still has
its own `occurrence_id` and temporal placement.

## Tasks

`Task` is abstract. Each concrete task gets one row in `tasks`. The `task_type` column selects the
Python subclass. Four typed columns store the scope: `scope_type`, `scope_start`, `scope_end`, and
`scope_signal_ids`. Subclass configuration uses one nullable column per field (`target_schema`,
`unit`, `target_name`, `mode`). With these columns, SQL can filter tasks without parsing JSON.

Each inline target is one row in `task_targets`. Its `target_kind` names the typed value column
that is set. A target stores public IDs directly: `record_id` for a Record, `signal_id` for a
Signal, and `span_signal_ids` for the series of a span target. A step target names one series. The
relationship tables hold the references to Records, annotation occurrences, and parent Tasks.

The `field` column distinguishes relationships such as `inputs` and `candidate_records`.
`position` preserves tuple order. On read, TimeNet hydrates each object once and restores ordinary
Python object references. It can also give back the same rows as Arrow tables of public IDs. See
[TimeFReader](timef-reader.md).

## Values plane

Signal arrays remain outside DuckDB. A `signal_chunks` row contains:

```text
signal_id, chunk_index, value_path, source_key, chunk_major_index,
chunk_minor_index, n_values
```

`source_key` is `NULL` when this release stores the chunk. Otherwise it is a key of the manifest's
`payload_sources`, which names the release that does.

For Parquet, the locator identifies a shard row group and row. For Zarr, it identifies an array and
element range. The reader resolves only the requested Signal's rows and keeps its values lazy. Range
reads load only intersecting chunks. The values plane does not duplicate irregular time offsets.
Within one read, the Signals on one axis share one handle to its packed time offsets, which checks
them against the axis on its first read and keeps them. All chunks of one Signal have the same
`source_key`.

## Shared payloads

A release can reuse the Signal chunks, packed text, and time offsets that other releases store,
instead of copying them. A release that stores a reused file is a payload source of the release
that reads it. The rows that locate such a payload in `signal_chunks` and `external_values` carry
a source key, which names that payload source as `org/name@version`. A source key always names the
release that stores the file, even when the value came through a release that reused it in turn,
so a reader resolves every payload in one step, without opening the source's catalog.

The manifest's `payload_sources` maps each source key to that exact release and pins the files the
release reads from it: copies of the source's own `FilePart` records, with path, checksum, and
size. For example:

```json
"payload_sources": {
  "physionet/ptb-xl@1.0.0": {
    "release": {"dataset_id": "physionet/ptb-xl", "version": "1.0.0"},
    "files": [
      {
        "path": "time_series/part-00000000.parquet",
        "checksum": "sha256:<hex digest>",
        "size": 1048576
      }
    ]
  }
}
```

The pins let a reader check, from the two manifests alone, that a source is still the release this
one was written against. A source replaced under the same version fails that check. The writer
lists every source a release reads from, flattened, so a source's own sources appear only when the
release reads their files too.

## Integrity

The writer builds the complete version in a temporary directory. DuckDB relationships are written
inside a transaction and validated as they stream, and the database carries no persisted
constraints. The writer then checkpoints and closes the database, records final checksums in
`manifest.json`, and atomically publishes the directory. `audit_control_database()` in
`timenet.format.control_audit` re-checks every relational invariant of a finished database with
SQL scans, for tests and tooling.

`TimeFReader.verify()` hashes every listed artifact. Normal reads stay lazy and verify the cached
remote `control.duckdb` before opening it read-only.
