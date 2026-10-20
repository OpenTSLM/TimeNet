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
the signal values.

## Directory layout

A Parquet-backed version looks like this:

```
org/name/version/
├── manifest.json
├── control.duckdb
└── time_series/
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
| `axis_offsets` | Shared offsets for irregular axes. |
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

Every table except `clocks` is keyed by IDs, and every relationship stores the ID it refers to.
The one internal key is `clocks.clock_id`, which `records.clock_id` refers to. The writer numbers
clocks from 1 on each write, so the value has no meaning outside one file and the Python API does
not expose it. Public IDs (`record_id`, `signal_id`, and `task_id`) and the writer's storage IDs
(`source_id`, `axis_id`, `content_id`, and `occurrence_id`) are positive int64 values stored as
`BIGINT`. Each ID is unique within its own kind, so a Record and a Signal can share one. The
`dataset_id` is the exception: it is the `org/name` string and is stored as `VARCHAR`. Fixed-shape
fields are typed columns, so spans, task configuration, target values, and annotation values can be
filtered in SQL. Only the free-form `metadata` dictionaries are JSON.

The `control_metadata` table stores the last ID used for each kind, under keys such as
`last_id.record` and `last_id.signal`. A version edited from this one continues from these values,
so it never reuses the ID of a removed object.

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
Irregular axes store their endpoints in `axes` and their ordered microsecond offsets in
`axis_offsets`. Signals that share an axis reference the same `axis_id`, so the offsets are stored
once regardless of how many Signals use that timeline.

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
the content through `content_id` and carries the span as `span_type`, `start_us`, and `end_us`,
plus provenance, confidence, and occurrence metadata.

An occurrence can annotate a Dataset, Task, Record, Source, or Signal. It names its owner with
`object_type` and `object_id`. For a Record, Signal, or Task, `object_id` is the owner's public
ID, and for a Source it is the Source's storage ID. `object_id` is `NULL` exactly when the owner
is the Dataset. IDs are unique only within one kind, so the type is part of the reference. A read
of every Record or every Task rejects an occurrence with an unknown owner type or a missing owner,
instead of dropping it. One content row can apply to many objects without copying a long value.
Each attachment still has its own `occurrence_id` and temporal placement.

## Tasks

`Task` is abstract. Each concrete task gets one row in `tasks`; `task_type` selects the Python
subclass. The scope is stored as four typed columns (`scope_type`, `scope_start`, `scope_end`,
`scope_signal_ids`), and subclass configuration as one nullable column per field
(`target_schema`, `unit`, `target_name`, `mode`), so tasks can be filtered in SQL without parsing
JSON. Inline targets are one row each in `task_targets`, with `target_kind` naming which typed value
column is set. A target stores public IDs directly: `record_id` for a Record, `signal_id` for a
Signal or the series of a step target, and `span_signal_ids` for a time point or time interval.
References to Records, annotation occurrences, and parent Tasks live in the relationship tables.

The `field` column distinguishes relationships such as `inputs` and `candidate_records`.
`position` preserves tuple order. On read, TimeNet hydrates each object once and restores ordinary
Python object references, or hands the same rows back as Arrow tables of public IDs. See
[TimeFReader](timef-reader.md).

## Values plane

Signal arrays remain outside DuckDB. A `signal_chunks` row contains:

```text
signal_id, chunk_index, value_path, chunk_major_index,
chunk_minor_index, n_values
```

For Parquet, the locator identifies a shard row group and row. For Zarr, it identifies an array and
element range. The reader resolves only the requested Signal's rows and keeps its values lazy. Range
reads load only intersecting chunks. The values plane does not duplicate irregular offsets; the
reader obtains them from the shared axis in `control.duckdb` and caches each loaded axis once.

## Integrity

The writer builds the complete version in a temporary directory. DuckDB relationships are written
inside a transaction and validated as they stream, and the database carries no persisted
constraints. The writer then checkpoints and closes the database, records final checksums in
`manifest.json`, and atomically publishes the directory. `audit_control_database()` in
`timenet.format.control_audit` re-checks every relational invariant of a finished database with
SQL scans, for tests and tooling.

`TimeFReader.verify()` hashes every listed artifact. Normal reads stay lazy and verify the cached
remote `control.duckdb` before opening it read-only.
