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
Signals can reference the same immutable `TimeAxis`. See
[Sources](data-model/sources.md#storage-and-read-back) for how TimeF stores and reads back Sources.

## DuckDB control tables

`control.duckdb` contains these authoritative tables:

| Table | Purpose |
| --- | --- |
| `control_metadata` | Schema version and the last entity ID used for each kind. |
| `datasets` | The one dataset that the file describes, keyed by its `org/name` `dataset_id`. |
| `records` | Recording sessions, session-level timing metadata, and the order in which readers list the records. |
| `clocks` | Record start times, shared by records that use the same `TimeOrigin`. |
| `sources` | Recursive [sources](data-model/sources.md#storage-and-read-back), linked to their record and parent source by ID. |
| `signals` | Signal identity plus inline `TimeSeriesSpec` fields. |
| `axes` | Regular, irregular, and ordinal axis definitions. |
| `external_values` | Locations of large text, string lists, and irregular time offsets. |
| `label_sets` | The label sets of classification tasks, one row per set. |
| `tasks` | Concrete task type, typed scope columns, and subclass configuration columns. |
| `task_targets` | Ordered inline targets, one typed row each. |
| `task_record_refs` | Ordered task-to-record object relationships. |
| `task_annotation_refs` | Ordered task-to-annotation occurrence relationships. |
| `task_dependencies` | Ordered task derivation relationships. |
| `annotation_contents` | Reusable annotation content with one typed value column per value shape. |
| `annotation_occurrences` | One attachment of content to an object, with its span. |
| `signal_chunks` | Payload source and locations of Signal values in Parquet or Zarr. |
| `imports` | Each parent alias of the release, with its exact parent release. |
| `imported_objects` | For each copy: the parent alias, the object kind, the parent ID, and the ID of the copy. |

Object relationships are normalized rather than embedded in JSON. DuckDB can follow the same row
in either direction, so TimeF does not store duplicate reverse-index tables.

Every relationship stores the ID that it refers to. Only the entity IDs (`record_id`, `signal_id`,
and `task_id`) stay the same across edits. The dataset assigns them, and users do not set them.
The storage IDs (`clock_id`, `source_id`, `axis_id`, `content_id`, and `occurrence_id`) are
internal row numbers. The writer numbers them from 1 on every write. As a result, a storage ID has
no meaning outside one file, and the Python API does not expose it.

Entity IDs and storage IDs are positive int64 values stored as `BIGINT`. Each ID is unique within
its own kind, so a Record and a Signal can have the same ID. The `dataset_id` is the exception. It
is the `org/name` string, stored as `VARCHAR`. Fixed-shape fields are typed columns, so SQL can
filter spans, task configuration, target values, and annotation values. Only the free-form
`metadata` dictionaries are JSON.

The `control_metadata` table stores the last entity ID used for each kind, under
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

The writer stores small text values in the catalog and larger values in separate payload files.
The size limit is 4,096 bytes of encoded text, for both strings and lists of strings.
`LazyText` and `LazyTextList` always use payload files, so readers can load their text on request.
Connectors use these handles to supply text through a function, such as a function that reads a
source file. Every irregular axis also stores its time offsets in a payload file.

Payload files use Zstandard-compressed Parquet under `payloads/`, independent of the Signal values
backend. Each file targets about 4 MiB of encoded input. One large value can exceed that target.
Each row fills exactly one column: `text` (string), `texts` (list of non-null strings), or
`time_offsets` (list of non-null int64). Readers reject files with any other schema.

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
column is separate from metadata, which can contain its own `description` key.

The value sits in one typed column selected by `value_kind`: `text_value`, `integer_value`, `float_value`,
`boolean_value`, or `text_list_value`. The `text_list_value` column holds a list of strings, such as
the names of activities in a recording. A marker annotation defines only a span. All its value columns are
`NULL`.

The `annotation_occurrences` table records where the content applies. It points at
the content through `content_id`. It also carries the span as `span_type`, `start_us`, and
`end_us`, plus provenance, confidence, and occurrence metadata.

An occurrence can annotate a Dataset, Task, Record, Source, or Signal. It names its owner with
`object_type` and `object_id`. For a Record, Signal, or Task, `object_id` is the entity ID of the
owner. For a Source, it is the storage ID of the Source. `object_id` is `NULL` exactly when the
owner is the Dataset. IDs are unique only within one kind, so the type is part of the reference.

A read of every Record or every Task rejects an occurrence with an unknown owner type or a missing
owner. It does not drop that occurrence.

One content row can apply to many objects without copying a long value. Each attachment still has
its own `occurrence_id` and temporal placement.

## Tasks

`Task` is abstract. Each concrete task gets one row in `tasks`. The `task_type` column selects the
Python subclass. Four typed columns store the scope: `scope_type`, `scope_start`, `scope_end`, and
`scope_signal_ids`.

Configuration for each task type uses the columns `label_set`, `unit`, `target_name`, and `mode`.
A column that a task type does not use is `NULL`. SQL can filter these columns without parsing JSON.

Every classification task has a label set, and `tasks.label_set` stores its name. Each row in
`label_sets` stores one set with the fields `name`, `labels`, and `multi_label`. The name is unique
within a release.

The `labels` field holds one `{name, description}` entry per label, in label order. If a label has
no description, its `description` is `NULL`.

Each inline target is one row in `task_targets`. Its `target_kind` names the typed value column
that is set. A target stores entity IDs directly: `record_id` for a Record, `signal_id` for a
Signal, and `span_signal_ids` for the series of a span target. A step target names one series. The
relationship tables hold the references to Records, annotation occurrences, and parent Tasks.

The `field` column distinguishes relationships such as `inputs` and `candidate_records`.
`position` preserves tuple order. On read, TimeNet builds each object once and restores ordinary
Python object references. It can also give back the same rows as Arrow tables of entity IDs. See
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

`TimeFReader.verify()` hashes every file that the release reads. A download makes sure that each
file has its recorded size and checksum before it adds the complete copy to the cache. Opening a
version makes sure that its local files exist with their recorded sizes, and hashes nothing. Reads
then use the local files, or byte ranges of the remote files. Normal reads stay lazy. Before a
reader opens a cached remote `control.duckdb` read-only, it makes sure that the file matches the
manifest.

## Catalogs of child datasets

A child dataset holds every catalog row that it needs, also for the objects that it imports. A
read of the child never opens the catalog of a parent. The `source_key` of a row in `signal_chunks`
or `external_values` names the release that stores the file, as
[Shared payloads](#shared-payloads) describes. The writer reuses the stored values of unchanged
copies and decodes none of them.

The `imports` and `imported_objects` tables map each parent ID to the ID of its copy. Only
`get(..., parent=)` and `read()` use them. Other reads use the IDs of the child and the file
locations in the catalog.
