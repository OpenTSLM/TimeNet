---
icon: lucide/book-open
description: "TimeFReader: read a TimeF version directory back into a TimeFDataset."
tags:
  - reference
  - reader
---

# TimeFReader

Deserializes a TimeF version directory into an in-memory [`TimeFDataset`](timef-dataset.md). The inverse
of [`TimeFWriter`](timef-writer.md), driven entirely by `manifest.json`: it never runs connector code.
Lives in `timenet.reader`.

```python
from timenet.reader import TimeFReader

with TimeFReader(version_dir) as reader:
    dataset = reader.read()
    values = dataset.samples[0].time_series[0].to_arrow()
```

Use it as a context manager: `close()` (called by `__exit__`) releases the selected values backend's
open handles and decoded-chunk caches.

## What is eager vs lazy

`__init__` reads the manifest, tasks, annotations, and the time-series index up front. Per-series values
and `Sample` construction stay lazy: `read()` / `iter_samples()` build samples with loader closures that
pull from storage only when `to_arrow()` / `to_numpy()` / `read_steps()` is called. `iter_samples()`
streams samples one at a time without building a `TimeFDataset`.

The index is held as Arrow and searched per lookup, rather than expanded into one Python object per
row. That keeps opening a large dataset proportional to the index file rather than to a multiple of
it: roughly 180 bytes of memory per index row, where a row is one `(sample, series, chunk)`.

## Type reconstruction

Specs, data sources, and annotation metadata are read straight from the manifest's flat descriptors.
There is no runtime class synthesis. `TimeSeries.spec` is the `TimeSeriesSpec` descriptor for its
`spec_type`; annotations are rebuilt as real `Annotation` instances (values decoded from JSON, the
span rebuilt as a `TimePoint`, `TimeInterval`, `StepPoint`, or `StepInterval`); tasks are resolved
against the built-in `TASKS` registry with `from_tasks` linked. Everything pickles and compares equal
to the originals field-for-field, which is what makes multiprocessing `DataLoader` workers safe.

## Value reads

A series' loader resolves its index rows (sorted by `chunk_idx`) and dispatches through the manifest's
`values_backend`. For Parquet, `chunk_file`, `chunk_major_idx`, and `chunk_minor_idx` identify the shard,
row group, and row offset. For Zarr, they identify the array path and temporal start offset. Scalars
use a primitive Arrow array; N-D values use `pa.FixedShapeTensorArray`, whose length is the number of
timesteps. `to_numpy()` is the explicit framework conversion and preserves the spec's dtype and
trailing shape. Range-aware reads select only the requested temporal chunks. Parquet handles and Zarr
decoded chunks are cached for the reader's lifetime and released on `close()`.

## API

| Member | Description |
| --- | --- |
| `read()` | Materialize the full `TimeFDataset`. |
| `iter_samples()` | Yield each `Sample` lazily. |
| `verify()` | Hash every manifest-listed artifact and reject missing or mismatched content. |
| `metadata` / `schema` / `tasks` / `values_backend` | Reconstructed metadata, schema, tasks, and selected values backend. |

## Errors

`__init__` raises `FileNotFoundError` if `root`, its `manifest.json`, or any file the manifest lists is
missing. A malformed or unsupported manifest, unreadable table, or disagreement between stored data
and its manifest raises `TimeFFormatError` (with `InvalidManifestError` for manifest parsing itself).
Lazy value-read failures retain their series context. `verify()` raises `TimeFFormatError` when a
checksummed artifact is missing or its content does not match the manifest.

## Round-trip guarantee

For a dataset that passes writer validation, `TimeFReader(...).read()` restores every sample's
`sample_id`, `subject_ids`, `task_ids`, and annotations; each series' `spec`, `channel`,
`source_id`, `time_series_id`, window, and exact dtype/shape-preserving values; and each task's payload
and resolved `from_tasks`. `TimeSeries` object identity is not preserved. `time_series_id` is the
durable handle.

---

See the [API reference for `timenet.reader`](api/reader.md) for the full symbol listing.
