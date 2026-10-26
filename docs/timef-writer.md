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

Pass `replace=True` to replace a version that is already committed. The old release stays in place
until the new one is complete, then moves aside to `<version>.old-<uuid>` for the publishing rename
and is deleted. A failed write keeps it. If a hard kill stops the writer between the two renames,
rename `<version>.old-<uuid>` back to recover the old release.

## Write stages

The writer performs these operations:

1. Derive the dataset schema when the caller has not done so.
2. Validate hierarchy ownership and reusable annotation content.
3. Read and validate each Signal, then write its values to Parquet or Zarr. A Signal read from a
   stored release keeps that release's chunks instead, without a read.
4. Create `control.duckdb`, store each shared axis once, and insert hierarchy, annotation, task, and
   chunk rows in one transaction. Packed text and time offsets read from a stored release keep
   their payload rows there.
5. Confirm the stored Source tree, checkpoint DuckDB, and close it.
6. Calculate artifact sizes and SHA-256 checksums.
7. Write `manifest.json` and atomically publish the version directory.

A failed operation publishes nothing.

Values read from a stored release are reused, not copied. The new catalog names the chunks and
payload rows where they are, and the manifest's `payload_sources` names each release that stores
them, so the values settings below do not re-encode them. The new release therefore needs a
different dataset ID or version than those releases, and its readers must find them: write it into
the registry that holds them. `LocalRegistry.store()` refuses to publish a release whose payload
sources its registry does not hold.

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
timesteps. Irregular-axis time offsets live in packed Parquet payloads under `payloads/`, not in
either values backend.

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
- irregular time offsets that disagree with their axis or their Signal's length;
- reused values whose storing release is the output release or does not list their files, or a
  reused Signal whose spec or length has changed; or
- chunk rows that do not cover a Signal exactly.

These failures raise TimeNet validation errors with object IDs and relationship fields where
possible. The writer checks them while it streams rows, from the ID sets and key sequences it
owns, rather than by scanning the finished tables. The same invariants exist in SQL form in
`timenet.format.control_audit.audit_control_database()`, which scans every table of a
`control.duckdb` and is what the test suite runs after each round trip.

## Task streams

A dataset can provide a re-iterable task source with `set_task_stream()`. The writer consumes the
validated stream once while inserting task rows and object relationships. It retains task IDs and
deferred dependency edges, not every Task object, so a large task corpus does not need to be
materialized in Python.

## Progress

Pass `progress_cb` to receive `WriteProgressEvent` objects. Events report schema work, completed
Signals, finalized value shards, and the final commit. The callback is observational: an exception
from it aborts the write like any other failure.
