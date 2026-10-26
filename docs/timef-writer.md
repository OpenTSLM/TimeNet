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
rename `<version>.old-<uuid>` back to recover the old release. If the publishing rename fails and the
old release cannot be moved back either, the error names the `<version>.old-<uuid>` directory that
holds it.

A replace is not isolated from readers that already opened the old release, or from releases that
reuse its payloads. An open reader loses the files it has not read yet, and a release that reuses
the old release's payloads fails to open, because its pins no longer match.

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
payload rows where they are, so the values settings below do not re-encode them. The manifest's
`payload_sources` names each release that stores them, its payload sources, and pins the files the
new release reads from each: a copy of that source's manifest record for each file. The new release
therefore needs a different dataset ID or version than its sources.

Its readers find the sources in the registry the release is published to, so the writer checks each
source there at its first reuse, during `write()`. Pass that registry as `registry=`. It defaults to a
local registry over `root`, and `S3Registry.store()` and `RemoteRegistry.store()` pass themselves while
they stage the release locally. The writer fetches the source's manifest from the registry once and
compares the record of each reused file. It refuses the write, before it writes the remaining values
and with nothing published, when:

- the registry does not hold the source, which raises `TimeNetDatasetNotFoundError` naming the source
  and the registry;
- the registry's source records a reused file differently, as after a forced replace, or because the
  registry holds another release under the same dataset ID and version, which raises
  `TimeFFormatError` naming the source and the first differing file; or
- the source reads payloads of the release being written, which would make each the other's source
  and raises `TimeFValidationError`.

These are the errors that [opening](timef-reader.md#opening-a-version) raises for a missing or
differing payload source.

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

- duplicate Record, Signal, or Task IDs;
- one annotation occurrence object attached to more than one object;
- a recursive Source cycle;
- a Signal attached to more than one Source;
- one irregular axis object shared by Signals with different time offsets;
- an annotation occurrence without an owning object;
- task relationships pointing outside the dataset;
- value arrays that disagree with their specification or declared length;
- irregular time offsets that disagree with their axis or their Signal's length;
- reused values whose storing release is the output release or does not list their files, or a
  reused Signal whose spec or length has changed;
- a payload source that reads payloads of the output release; or
- chunk rows that do not cover a Signal exactly.

These failures raise TimeNet validation errors. Where possible, the errors include object IDs and
relationship fields. The writer checks most failures while it streams rows. It uses the ID sets
that it owns. It does not keep the ID of every task, so after the last task batch it runs one query
on `tasks` to find a repeated task ID.

Each Record and Signal that a task refers to, as an input, a target, or in a span, must be the
same object that the dataset registered under that ID. Thus a Record from another dataset fails,
even when its ID is in use in this dataset.

The writer inserts annotation references after the occurrences get their IDs. Thus an annotation
that is not attached in the dataset fails at that step. The writer inserts dependencies after the
last task batch. Each parent must be the same Task object that the write stored under that ID. Thus
a parent from another dataset fails, even when its ID is in use in this dataset.

`timenet.format.control_audit.audit_control_database()` holds the same invariants in SQL form. It
scans every table of a `control.duckdb`. The test suite runs it after each round trip.

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
