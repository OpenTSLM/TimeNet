---
icon: lucide/save
description: "How TimeFWriter publishes DuckDB-backed TimeF v2 datasets."
tags:
  - reference
  - writer
---

# TimeFWriter

`TimeFWriter` publishes one immutable TimeF v2 dataset version. It writes structural data and
relationships to `control.duckdb`, sends Signal arrays to the selected values backend, and writes
`manifest.json` last.

## Basic use

The declarative entry point is `TimeFDataset.write()`:

```python
from timenet.values_backends import ValuesBackend

version_path = dataset.write(
    path="./registry",
    values_backend=ValuesBackend.PARQUET,
)
```

Use `TimeFWriter` directly when you need storage tuning or progress events:

```python
from timenet.writer import TimeFWriter

with TimeFWriter(
    root,
    dataset,
    values_backend="parquet",
    chunk_max_bytes=16 * 2**20,
    shard_target_bytes=512 * 2**20,
) as writer:
    writer.write()
```

The context manager commits only after `write()` succeeds. An exception removes the temporary
version directory.

## Write stages

The writer performs these operations:

1. Derive the dataset schema when the caller has not done so.
2. Validate hierarchy ownership and reusable annotation content.
3. Read and validate each Signal, then write its values to Parquet or Zarr.
4. Create `control.duckdb` and insert hierarchy, annotation, task, axis, and chunk rows in one
   transaction.
5. Run relational integrity checks, checkpoint DuckDB, and close it.
6. Calculate artifact sizes and SHA-256 checksums.
7. Write `manifest.json` and atomically publish the version directory.

A failed operation publishes nothing.

## Values settings

| Argument | Purpose |
| --- | --- |
| `values_backend` | `"parquet"` or `"zarr"`. |
| `chunk_max_bytes` | Maximum logical Signal chunk size. |
| `shard_target_bytes` | Approximate value-shard rotation target. |
| `row_group_target_bytes` | Parquet value row-group target. |
| `compression` | Parquet codec or Zarr Blosc inner codec. |
| `compression_level` | Optional backend compression level. |
| `data_page_size` | Optional Parquet data-page target. |
| `value_encoding` | Parquet values-column encoding, or `"auto"`. |

Multidimensional Signals use Zarr. Parquet supports scalar Signals. Both backends preserve nullable
timesteps and irregular-axis offsets.

## Validation

The writer fails before publication when it finds an invalid state, including:

- duplicate Record, Source, Signal, Task, or annotation occurrence IDs;
- a Source attached to more than one parent or a recursive Source cycle;
- a Signal attached to more than one Source;
- inconsistent definitions sharing one axis ID;
- annotation content IDs reused with different payloads;
- an annotation occurrence without an owning object;
- task relationships pointing outside the dataset;
- value arrays that disagree with their specification or declared length;
- irregular offsets that disagree with their axis; or
- chunk rows that do not cover a Signal exactly.

These failures raise TimeNet validation errors with object IDs and relationship fields where
possible.

## Task streams

A dataset can provide a re-iterable task source with `set_task_stream()`. The v2 writer consumes the
validated stream once while inserting task rows and object relationships. It retains task IDs and
deferred dependency edges, not every Task object, so a large task corpus does not need to be
materialized in Python.

## Progress

Pass `progress_cb` to receive `WriteProgressEvent` objects. Events report schema work, completed
Signals, finalized value shards, and the final commit. The callback is observational: an exception
from it aborts the write like any other failure.
