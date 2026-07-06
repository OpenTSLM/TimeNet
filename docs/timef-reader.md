---
icon: lucide/book-open
description: "TimeFReader: read a TimeF version directory back into a TimeFDataset."
tags:
  - reference
  - reader
---

# TimeFReader

Deserializes a TimeF version directory into an in-memory [`TimeFDataset`](timef-dataset.md). The inverse
of [`TimeFWriter`](timef-writer.md), driven entirely by `manifest.json` — it never runs connector code.
Lives in `timenet.reader`.

```python
from timenet.reader import TimeFReader

with TimeFReader(version_dir) as reader:
    dataset = reader.read()
    values = dataset.samples[0].time_series[0].to_arrow()
```

Use it as a context manager: `close()` (called by `__exit__`) closes the cached shard file handles.

## What is eager vs lazy

`__init__` reads the manifest, tasks, annotations, and the time-series index up front. Per-series values
and `Sample` construction stay lazy: `read()` / `iter_samples()` build samples with loader closures that
pull from the shards only when `to_arrow()` / `to_numpy()` is called. `iter_samples()` streams samples
one at a time without building a `TimeFDataset`.

## Type reconstruction

Specs, data sources, and annotation metadata are read straight from the manifest's flat descriptors —
there is **no runtime class synthesis**. `TimeSeries.spec` is the `TimeSeriesSpec` descriptor for its
`spec_type`; annotations are rebuilt as real `StaticAnnotation` / `PointAnnotation` /
`IntervalAnnotation` instances (values decoded from JSON); tasks are resolved against the built-in
`TASKS` registry with `from_tasks` linked. Everything pickles and compares equal to the originals
field-for-field, which is what makes multiprocessing `DataLoader` workers safe.

## Value reads

A series' loader resolves its index rows (sorted by `chunk_idx`), reads each chunk with
`ParquetFile.read_row_group(rg, columns=["values"])[row_offset]`, and concatenates them into one
`float32` Arrow array. Shard handles are cached for the reader's lifetime and closed on `close()`.

## API

| Member | Description |
| --- | --- |
| `read()` | Materialize the full `TimeFDataset`. |
| `iter_samples()` | Yield each `Sample` lazily. |
| `metadata` / `schema` / `tasks` | The reconstructed metadata, schema, and tasks. |

## Errors

`__init__` raises `FileNotFoundError` if `root`, its `manifest.json`, or any file the manifest lists is
missing, and `InvalidManifestError` (a `TimeFFormatError`) for a malformed or unsupported-version
manifest. Every id cross-reference (annotation, `from_task`, spec type, index lookup) raises a located
`ValueError` naming the offending id.

## Round-trip guarantee

For a dataset that passes writer validation, `TimeFReader(...).read()` restores every sample's
`sample_id`, `view`, `subject_ids`, `task_ids`, and annotations; each series' `spec`, `channel`,
`source_id`, `time_series_id`, window, and exact `float32` values; and each task's payload and resolved
`from_tasks`. `TimeSeries` object identity is not preserved — `time_series_id` is the durable handle.

---

See the [API reference for `timenet.reader`](api/reader.md) for the full symbol listing.
