---
icon: lucide/book-open
description: "TimeFReader: query a committed TimeF version's control plane and read its values."
tags:
  - reference
  - reader
---

# TimeFReader

`TimeFReader` reads a committed TimeF version: the control plane with SQL, the values through
whichever backend wrote them. It is the inverse of [`TimeFWriter`](timef-writer.md) and lives in
`timenet.control_plane`.

```python
from timenet.client import TimeNet
from timenet.control_plane import TimeFReader

# through a registry, which hands over a rooted view of the version's files:
with TimeNet("./local_registry").open("demo/bedside-monitor") as reader:
    record = reader.record("record-123")

# ...or straight from a version directory already on disk:
with TimeFReader(version_dir) as reader:
    print(reader.counts())
```

Use it as a context manager, or call `close()`. The database opens read-only, which writes no
write-ahead log beside the file, so reading a downloaded version leaves it byte-identical to what
the manifest checksummed.

## Views

A query hands back plain dataclasses, not rows.

| View | What it carries |
| --- | --- |
| `RecordView` | `record_id`, `external_id`, `start_time_us`, root `sources`, `annotations`. `walk_sources()` returns every source parents-first, `signals()` every signal in walk order. |
| `SourceView` | `source_id`, `external_id`, `name`, `depth`, child `sources`, `signals`, `annotations`. |
| `SignalView` | `signal_id`, `external_id`, `name`, `spec_type`, `unit`, `dtype`, `axis_type`, `n_values`, the resolved `time_axis`, the spec's `nullable` flag, and `annotations`. |
| `TaskView` | `task_id`, `external_id`, `prompt`, and `inputs` / `target` as ordered lists of text and rebuilt records. |
| `ResolvedAnnotation` | `name`, `value`, `unit`, `span_type`, `start_us`, `end_us`, `provenance`, `confidence`. `to_text()` renders one line for a prompt. |

`signal_id`, `record_id`, and `task_id` are the surrogates the writer assigned. They are what the
values calls take. `external_id` is the id the builder gave, and it is what survives a rebuild.

## Reading records and tasks

| Member | Description |
| --- | --- |
| `record(external_id)` / `records(external_ids)` | Rebuild one record, or a batch, by the id it was built under. |
| `records_by_id(record_ids)` | The same, by surrogate id, skipping the lookup. |
| `iter_records(*, batch_size, worker_index, num_workers)` | Walk every record in this worker's slice, hydrating a batch per round of queries. |
| `task(external_id)` / `tasks(external_ids)` / `tasks_by_id(task_ids)` | The same three shapes for tasks. Each task's referenced records are rebuilt in full. |
| `iter_tasks(...)` | Walk every task, hydrating each batch's records together. |
| `record_ids()` / `task_ids()` | Every external id, sorted. |
| `counts()` | Row count per table. |

A batch costs a fixed number of queries however large it is, so batching is the whole read contract.
Without indexes each lookup scans its table, and reading one record at a time degrades linearly with
corpus size: 29 records a second against 15,540 on a 618,508-record corpus. Prefer `iter_records`
and the batch forms.

`worker_index` and `num_workers` split the corpus into disjoint slices, which is how a `DataLoader`'s
workers together see every record exactly once. The reader drops its connection when it is pickled
into a worker and opens a new one there.

## Querying annotations

| Member | Description |
| --- | --- |
| `annotations_for(object_type, object_id=0)` | Every annotation on one object, where `object_type` is `dataset`, `task`, `record`, `source`, or `signal`. |
| `objects_with(name, value=None)` | Every `(object_type, object_id)` carrying an annotation, starting from the annotation. |
| `records_with(name, value=None)` | Every record carrying an annotation, directly or on one of its sources or signals. |
| `tasks_for_record(external_id)` | Every task that refers to a record. |
| `subtree(record_id, source_id)` | One source and everything beneath it, depth-first, without walking edge by edge. |

`connection` exposes the open DuckDB connection for a caller that wants its own SQL. Every table in
[the format page](timef-format.md#the-control-plane-tables) is queryable directly.

## Reading values

| Member | Description |
| --- | --- |
| `values(signal_id)` | One signal's values, as a NumPy array in the stored dtype. |
| `values_for(signal_ids)` | Many signals in one pass, keyed by signal id. |
| `values_window(signal_id, start, stop)` | The half-open step range `[start, stop)` of one signal. |
| `values_windows(windows)` | Many `(signal_id, start, stop)` windows in one pass. |
| `chunk_locators(signal_id)` | Where a signal's chunks live, in chunk order. |
| `values_backends()` | Which backend wrote each values artifact. |

`values_for` and `values_windows` ask for every locator in one query and group the reads by artifact
and row group, so each row group is decoded once. Calling `values` per signal instead re-opens the
shard and re-decodes the row group for every signal, and a series shared by several records is
decoded once per use.

Windows matter when the canonical item is smaller than a recording. One scored 30 second epoch of an
overnight polysomnogram is 12 KB of a 34 MB recording, and `values()` would decode all 34 MB to hand
back the 12 KB. A `stop` past the last step is clamped; an empty window reads back an empty array of
the signal's dtype without touching the values plane.

## Rendering a task

`render_task(reader, external_id, *, include_values=False)` turns one task into the text a training
example starts from: the prompt, its inputs and targets in order, each record's source tree, each
signal's spec and axis, and the annotations at every level. Pass `include_values=True` to summarize
each signal's values as well.

## Errors

Opening a directory with no `control.duckdb`, or a database with no `schema_version` in its `meta`
table, raises `TimeFFormatError`. So does opening a version whose manifest names no control
database, and asking `annotations_for` for an object type that is not one of the five annotatable
kinds. A `subtree` call naming a source the record does not hold raises the same error.

---

See the [API reference for `timenet.control_plane`](api/control_plane.md) for the full symbol
listing.
