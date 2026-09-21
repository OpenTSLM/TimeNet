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
| `sources` | Recursive sources, linked by `record_id` and `parent_source_id`. |
| `signals` | Signal identity plus inline `TimeSeriesSpec` fields. |
| `axes` | Regular, irregular, and ordinal axis definitions. |
| `axis_offsets` | Shared offsets for irregular axes. |
| `tasks` | Concrete task type and scalar payload. |
| `task_record_refs` | Ordered task-to-record object relationships. |
| `task_signal_refs` | Ordered task-to-signal object relationships. |
| `task_annotation_refs` | Ordered task-to-annotation occurrence relationships. |
| `task_dependencies` | Ordered task derivation relationships. |
| `annotation_contents` | Reusable annotation payloads. |
| `annotation_occurrences` | One attachment of content to an object. |
| `signal_chunks` | Locations of Signal values in Parquet or Zarr. |

Object relationships are normalized rather than embedded in JSON. DuckDB can follow the same row
in either direction, so TimeF does not store duplicate reverse-index tables.

## Signals and axes

Each `signals` row identifies its owning Source and axis. The fixed `TimeSeriesSpec` attributes are
stored inline:

```text
signal_id, source_id, name, axis_id, spec_type, spec_name, unit,
dtype, categories, value_shape, dimension_names, nullable, n_values, metadata
```

There is no separate specification table. The fields are small, typed, directly queryable, and
compress well in DuckDB. The reader reconstructs a `TimeSeriesSpec` object from them.

Regular axes store a rational microsecond period and origin. Ordinal axes need only their type.
Irregular axes store their endpoints in `axes` and their ordered microsecond offsets in
`axis_offsets`. Signals that share an axis reference the same `axis_id`.

## Annotations

An annotation has reusable content and one or more occurrences. `annotation_contents` stores the
payload once. `annotation_occurrences` says where it applies and carries its optional span,
provenance, confidence, and occurrence metadata.

An occurrence can annotate a Dataset, Task, Record, Source, or Signal. One content row can therefore
apply to many objects without copying a long value. Each attachment still has its own
`occurrence_id` and temporal placement.

## Tasks

`Task` is abstract. Each concrete task gets one row in `tasks`; `task_type` selects the Python
subclass. `payload` contains only subclass-specific scalar and span fields. References to Records,
Signals, annotation occurrences, and parent Tasks live in the relationship tables.

The `field` column distinguishes relationships such as `inputs`, `context_records`,
`target_record`, and `candidate_records`. `position` preserves tuple order. On read, TimeNet hydrates
each object once and restores ordinary Python object references.

## Values plane

Signal arrays remain outside DuckDB. A `signal_chunks` row contains:

```text
signal_id, chunk_index, value_path, chunk_major_index,
chunk_minor_index, n_values
```

For Parquet, the locator identifies a shard row group and row. For Zarr, it identifies an array and
element range. The reader resolves only the requested Signal's rows and keeps its values lazy. Range
reads load only intersecting chunks.

## Integrity

The writer builds the complete version in a temporary directory. DuckDB relationships are written
inside a transaction and validated before publication. The writer then checkpoints and closes the
database, records final checksums in `manifest.json`, and atomically publishes the directory.

`TimeFReader.verify()` hashes every listed artifact. Normal reads stay lazy and verify the cached
remote `control.duckdb` before opening it read-only.
