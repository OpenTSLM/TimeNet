---
icon: lucide/book-open
description: "TimeFReader: read a TimeF version directory back into a TimeFDataset."
tags:
  - reference
  - reader
---

# TimeFReader

`TimeFReader` converts a committed TimeF version into an in-memory [`TimeFDataset`](timef-dataset.md).
It is the inverse of [`TimeFWriter`](timef-writer.md). It uses only `manifest.json` and never runs
connector code. It reads through a [`DatasetVersion`](registry.md) handle, which holds the manifest
and a view of the version's files on the file system. As a result, a read never opens the registry
again or parses the manifest again. `TimeFReader` is in the `timenet.reader` module.

```python
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion, open_registry

# From a registry (reads in place, no download):
version = open_registry("~/timenet/registry").open_version("timenet/hello-world")
# ...or from a version directory already on disk:
version = DatasetVersion.open_local(version_dir)

with TimeFReader(version) as reader:
    dataset = reader.read()
    values = dataset.records[0].time_series[0].to_arrow()
```

You can use `TimeFReader` as a context manager. Its `close()` method, called by `__exit__`, releases
the open handles and decoded-chunk caches of the selected values backend.

## What is eager vs lazy

`__init__` reads nothing. The handle already has the parsed manifest. Every table resolves on first
use.

`TimeFReader` opens `control.duckdb` read-only on first use and answers every lookup with a keyed
query. Records hydrate with their Sources, Signals, axes, and annotations in one pass over the rows
related to the requested IDs. Per-series values stay lazy: `read()` and `iter_records()` build
Signals with loader closures, and `to_arrow()`, `to_numpy()`, or `read_steps()` pull chunks from
storage when called.

`iter_records(record_ids=...)` streams records one at a time and does not build a `TimeFDataset`.

Tasks have three read paths. `.tasks` hydrates the whole table in one pass and caches it.
`iter_tasks(records)` yields the tasks whose inputs include the given Records in bounded batches of
keyed queries, so a per-record consumer never holds the whole table. `task_table()`,
`target_table()`, and `annotation_table()` return Arrow tables straight from DuckDB with public IDs
in place of internal keys and build no Python objects at all.

Tasks point at the same `Record` and `Annotation` objects the reader handed out, so identity
comparisons work across `iter_records()` and `iter_tasks()` from one reader.

## Type reconstruction

`TimeFReader` reads specs, data sources, and annotation metadata directly from the manifest's flat
descriptors. It does not create any classes at runtime. `TimeSeries.spec` is the `TimeSeriesSpec`
descriptor for its `spec_type`. `TimeFReader` rebuilds annotations as real `Annotation` instances,
reading each value from its typed column so an integer comes back as an integer and a boolean as a
boolean, and rebuilds spans as `TimePoint`, `TimeInterval`, `StepPoint`, or `StepInterval`. It
resolves tasks against the built-in `TASKS` registry and links `from_tasks`. Values read back from a
control database skip re-validation, since they passed the same checks when they were written.

Everything pickles and compares equal to the original data, field by field. This equality is why
multiprocessing `DataLoader` workers are safe.

## Value reads

Each series' loader resolves its index rows, sorted by `chunk_idx`. The loader dispatches through the
manifest's `values_backend`. For Parquet, `chunk_file`, `chunk_major_idx`, and `chunk_minor_idx`
identify the shard, row group, and row offset. For Zarr, they identify the array path and the start
offset in time.

Scalars use a primitive Arrow array. N-D values use `pa.FixedShapeTensorArray`, whose length is the
number of timesteps. `to_numpy()` is the explicit conversion to the array framework and keeps the
spec's dtype and trailing shape. Range-aware reads select only the requested chunks in time.
`TimeFReader` caches Parquet handles and decoded Zarr chunks for the life of the reader. When the code
calls `close()`, it releases them.

## API

| Member | Description |
| --- | --- |
| `read()` | Materialize the full `TimeFDataset`. |
| `iter_records()` | Yield each `Record` lazily. |
| `iter_tasks(records)` | Yield the tasks attached to the given Records in bounded batches. |
| `task_table()` / `target_table()` / `annotation_table()` | Arrow tables of tasks, their targets, and annotation occurrences with public IDs. |
| `verify()` | Hash every manifest-listed artifact and reject missing or mismatched content. |
| `metadata` / `schema` / `tasks` / `values_backend` | Reconstructed metadata, schema, tasks, and selected values backend. |

## Errors

`DatasetVersion.open_local` and a registry's `open_version` build the handle. If the version directory
has no `manifest.json`, they raise `FileNotFoundError`. If the manifest is malformed or is an
unsupported version, they raise `TimeFFormatError` (an `TimeNetInvalidManifestError`).

Opening the reader reads nothing else. As a result, the reader does not catch a missing or corrupt
file at open time. The error surfaces on the first access that needs the file. Tasks raise the error
on first access to `.tasks`. Records raise it on iteration. The index and annotations raise it on the
first read that needs them.

A corrupt control-plane table raises `TimeFFormatError` with its context. You can call `verify()` for
an integrity check at construction time. It reopens every listed file through the handle and raises
`TimeFFormatError` on a missing or mismatched file.

## Round-trip guarantee

For a dataset that passes writer validation, `TimeFReader(...).read()` restores every record's
`record_id`, `subject_ids`, `task_ids`, and annotations. It also restores each series' `spec`,
`signal`, `source_id`, `time_series_id`, window, and values, with the exact dtype and shape
preserved. It restores each task's payload and resolved `from_tasks`. `TimeSeries` object identity is
not preserved. `time_series_id` is the durable handle.

---

The [API reference for `timenet.reader`](api/reader.md) has the full symbol listing.
