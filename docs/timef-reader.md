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

`__init__` reads `manifest.json` and the annotations table. Tasks decode on first `.tasks` access, and
the time-series index resolves a row group at a time on the first value read, so opening a version never
scans the index whether it holds three samples or three million. Per-series values and `Sample`
construction stay lazy: `read()` / `iter_samples()` build samples with loader objects that pull from
storage only when `to_arrow()` / `to_numpy()` / `read_steps()` is called, and `iter_samples()` streams
`samples.parquet` in batches rather than materializing it.

| Part | When it loads | What is kept |
| --- | --- | --- |
| Manifest | `__init__` | metadata, schema, file list, checksums |
| Annotations | `__init__` | the decoded annotations, keyed by id |
| `tasks` | first `.tasks` access | the decoded tasks, cached for the reader's lifetime |
| Time-series index | first value read | a row-group directory (one entry per row group, not per row) and the last few decoded row groups |
| Series values | `to_arrow()` / `to_numpy()` / `read_steps()` | the values backend's handles and chunk caches |

An index lookup is a filtered read: the index is sorted by `sample_id`, so a row group whose recorded
id range excludes the sample is skipped without being read. Nothing about the index is held per row.
A sample-ordered read walks the file in order, so the small row-group cache serves a whole scan from a
handful of decodes; a shuffled read pays one row-group decode per miss instead.

Laziness moves when corruption surfaces. A structurally corrupt but checksum-valid task partition or
index fails on the first access that needs it (`.tasks`, the first value read), not at
`TimeFReader(root)`. Call `verify()` if you want an integrity check at a point you choose.

## Type reconstruction

Specs, data sources, and annotation metadata are read straight from the manifest's flat descriptors.
There is no runtime class synthesis. `TimeSeries.spec` is the `TimeSeriesSpec` descriptor for its
`spec_type`; annotations are rebuilt as real `Annotation` instances (values decoded from JSON, the
span rebuilt as a `PointSpan` or `IntervalSpan`); tasks are resolved against the built-in
`TASKS` registry with `from_tasks` linked. Everything pickles and compares equal to the originals
field-for-field, which is what makes multiprocessing `DataLoader` workers safe.

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
| `iter_samples(sample_ids=None)` | Yield each `Sample` lazily, or only the named ones. |
| `verify()` | Hash every manifest-listed artifact and reject missing or mismatched content. |
| `metadata` / `schema` / `tasks` / `values_backend` | Reconstructed metadata, schema, tasks, and selected values backend. |

## Errors

`__init__` raises `FileNotFoundError` if `root`, its `manifest.json`, or any file the manifest lists is
missing, `TimeFFormatError` (as `InvalidManifestError`) for a malformed or unsupported manifest, and
`TimeFFormatError` if the annotations table is unreadable or disagrees with the manifest. A corrupt task
partition or index surfaces where it is read: an unreadable table raises `TimeFFormatError` from the
access that touched it, and lazy value-read failures retain their series context.
`iter_samples(sample_ids=...)` raises `TimeFValidationError` for an id the dataset does not contain.
`verify()` raises `TimeFFormatError` when a checksummed artifact is missing or its content does not
match the manifest.

## Round-trip guarantee

For a dataset that passes writer validation, `TimeFReader(...).read()` restores every sample's
`sample_id`, `subject_ids`, `task_ids`, and annotations; each series' `spec`, `channel`,
`source_id`, `time_series_id`, window, and exact dtype/shape-preserving values; and each task's payload
and resolved `from_tasks`. `TimeSeries` object identity is not preserved. `time_series_id` is the
durable handle.

---

See the [API reference for `timenet.reader`](api/reader.md) for the full symbol listing.
