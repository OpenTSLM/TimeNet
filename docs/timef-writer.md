---
icon: lucide/save
description: "How TimeFWriter publishes DuckDB-backed TimeF datasets."
tags:
  - reference
  - writer
---

# TimeFWriter

`TimeFWriter` publishes one immutable TimeF dataset version. It writes structural data and
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
4. Create `control.duckdb`, store each shared axis once, and insert hierarchy, annotation, task, and
   chunk rows in one transaction.
5. Confirm the stored Source tree, checkpoint DuckDB, and close it.
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
timesteps. Irregular-axis offsets live in `control.duckdb`, not in either values backend.

## Validation

The writer fails before publication when it finds an invalid state, including:

- duplicate Record, Signal, or Task IDs;
- one annotation occurrence object attached to more than one object;
- a recursive Source cycle;
- a Signal attached to more than one Source;
- one irregular axis object shared by Signals with different time offsets;
- an annotation occurrence without an owning object;
- task relationships pointing outside the dataset;
- value arrays that disagree with their specification or declared length;
- irregular offsets that disagree with their axis; or
- chunk rows that do not cover a Signal exactly.

These failures raise TimeNet validation errors. Where possible, the errors name object IDs and
relationship fields. The writer checks most of them while it streams rows. It uses the ID sets and
key sequences it owns. It does not keep the ID of every task, so after the last task batch it runs
one query on `tasks` to find a repeated task ID.

Each Record and Signal that a task refers to, as an input, a target, or in a span, must be the
same object that the dataset registered under that ID. So a Record from another dataset fails,
even when its ID is in use in this dataset.

The writer inserts annotation references after the occurrences get their ids. So an annotation
that is not attached in the dataset fails at that step. The writer inserts dependencies after the
last task batch. Each parent must be the same Task object that the write stored under that ID. So
a parent from another dataset fails, even when its ID is in use in this dataset.

The same invariants exist in SQL form in `timenet.format.control_audit.audit_control_database()`.
That function scans every table of a `control.duckdb`, and the test suite runs it after each round
trip.

Sources, axes, and annotations have no ID, so the writer does not compare IDs for them. It stores
equal annotation contents once, and equal regular or ordinal axes once. It matches irregular axes
and annotation occurrences by object identity. Signals share an irregular axis row only if they use
the same axis object. Each occurrence object gets one row.

## Task streams

A dataset can provide a re-iterable task source with `set_task_stream()`. The writer consumes the
validated stream once while it inserts task rows and object relationships. It keeps the deferred
dependency edges, not every Task object. So the writer does not materialize a large task corpus in
Python.

## Progress

Pass `progress_cb` to receive `WriteProgressEvent` objects. Events report schema work, completed
Signals, finalized value shards, and the final commit. The callback is observational: an exception
from it aborts the write like any other failure.
