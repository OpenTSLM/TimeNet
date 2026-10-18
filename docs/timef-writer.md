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

## Final IDs

The writer never changes your objects. They keep the IDs they had in memory, which are usually
provisional IDs from `new_id()`. The writer translates every ID it stores, including every
reference, to a final ID below `2**52`. `write()` returns a read-only `FinalIds` mapping from each
in-memory ID to its stored ID, per kind. `writer.final_ids` returns the same mapping.

```python
with TimeFWriter(root, dataset) as writer:
    ids = writer.write()
stored_record_id = ids.records[dataset.records[0].id]
```

A first write numbers each kind `1..N` in storage order. An ID below `2**52` is kept. The dataset
remembers the IDs that its last successful write assigned. When you write it again, every object
from that write keeps its stored ID, and new objects get IDs above the largest ID of their kind.
A new annotation content or regular axis equal to one that the last write stored takes that ID. A
removed object leaves a gap, and no other object gets its ID. Writing an unchanged dataset again
gives an identical catalog. An edit of a stored version follows the same rules.

A task stream keeps its IDs only when it yields the same task objects on each pass. A stream that
builds new tasks each time gets new task IDs above the last write's.

The writer stores equal annotation contents once. Two annotations with the same key, typed value,
unit, description, and metadata share one `content_id`, even when you built them separately. The
writer also stores equal regular axes once, and all ordinal axes once. Irregular axes, Records,
Sources, Signals, occurrences, and Tasks are never merged. IDs that you copy into `metadata` are
not translated.

## Write stages

The writer performs these operations:

1. Plan the final ID of every in-memory ID, in storage order. Derive the dataset schema when the
   caller has not done so.
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

- duplicate Record, Source, Signal, Task, or annotation occurrence IDs;
- a Source attached to more than one parent or a recursive Source cycle;
- a Signal attached to more than one Source;
- inconsistent definitions sharing one axis ID;
- one annotation content ID with two different payloads;
- an annotation occurrence without an owning object;
- task relationships pointing outside the dataset;
- value arrays that disagree with their specification or declared length;
- irregular offsets that disagree with their axis; or
- chunk rows that do not cover a Signal exactly.

These failures raise TimeNet validation errors with object IDs and relationship fields where
possible. The writer checks most of them while it streams rows, from the ID sets it owns, rather
than by scanning the finished tables. Task dependencies and annotation references go in with each
task batch. One query for each kind checks them after all rows are written. The same invariants
exist in SQL form in `timenet.format.control_audit.audit_control_database()`, which scans every
table of a `control.duckdb` and is what the test suite runs after each round trip.

## Task streams

A dataset can provide a re-iterable task source with `set_task_stream()`. The writer consumes the
validated stream once while inserting task rows and object relationships. It numbers the tasks in
stream order as they pass. A stream gives every task a provisional ID or every task an explicit ID
below `2**52`. The writer rejects a stream that mixes the two. A task can derive from a task that
streams later: the writer stores that dependency when the stream ends. It retains the ID mapping,
not every Task object, so a large task corpus does not need to be materialized in Python.

## Progress

Pass `progress_cb` to receive `WriteProgressEvent` objects. Events report schema work, completed
Signals, finalized value shards, and the final commit. The callback is observational: an exception
from it aborts the write like any other failure.
