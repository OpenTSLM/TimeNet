---
icon: lucide/pencil
description: "TimeFWriter: compile a declarative hierarchy into a committed TimeF version."
tags:
  - reference
  - writer
---

# TimeFWriter

`TimeFWriter` compiles an in-memory hierarchy into one TimeF version: a DuckDB control plane, a
values plane, and the `manifest.json` that describes them. It lives in `timenet.control_plane`.

```python
from timenet.control_plane import TimeFWriter

# committed at <root>/<dataset_id>/<version>/
with TimeFWriter(root, dataset.metadata) as writer:
    manifest = writer.write(dataset)
```

## The declarative hierarchy

A builder assembles plain objects and hands them over. They carry no storage detail: no surrogate
ids, no occurrence rows, no reverse indexes. The writer derives all of that.

| Object | What it is |
| --- | --- |
| `DeclarativeDataset` | The metadata, the records, the tasks, and the dataset-level annotations. |
| `Record` | One recording session: root sources, an optional `start_time_us`, annotations. |
| `Source` | A device or assembly. It nests: `sources` for children, `signals` for leaves. |
| `Signal` | One value array, one `TimeAxis`, one `TimeSeriesSpec`, optional `time_offsets_us`. |
| `Task` | A `prompt`, an ordered `inputs` list, and an ordered `target` list. |
| `RecordRef` | A record named by id, for a task built after the record object was dropped. |
| `Annotation` | One `(name, value, unit)` statement, placed as `static`, `point`, or `interval`. |

```python
from fractions import Fraction

from timenet.control_plane import (
    Annotation, DeclarativeDataset, Record, Signal, Source, Task,
)
from timenet.dataset.axis import RegularAxis

lead = Signal(
    id="record-123-lead-i",
    name="I",
    values=values,
    time_axis=RegularAxis(period_us=Fraction(1_000_000, 500)),
    spec=ecg_spec,
)
lead.annotate(
    Annotation.point(name="lead_status", value="Lead fell off", at_us=6_000_000)
)

ecg = Source(id="record-123-ecg", name="ECG", signals=[lead])
monitor = Source(id="record-123-monitor", name="Bedside monitor", sources=[ecg])

record = Record(id="record-123", sources=[monitor])
record.annotate(Annotation.static(name="patient_sex", value="male"))

dataset = DeclarativeDataset(metadata=metadata)
dataset.add_record(record)
dataset.add_task(
    Task(
        id="diagnosis-record-123",
        prompt="Diagnose this patient.",
        inputs=[record],
        target=["The patient is stable."],
    )
)
```

The same `Annotation` object attached to many entities is stored once and linked many times, so
`patient_sex=male` on 100,000 records costs one payload row. `Source.select(signal_names=...)`
attaches one annotation to several signals beneath a source at once.

Ids are optional. An entity without one gets a generated id, stored as `external_id`. Supply your
own wherever a name from outside the dataset matters, because that is the id the reader's `record()`
and `task()` take and the one that survives a rebuild.

## Writing

| Method | What it does |
| --- | --- |
| `write(dataset)` | Compile a whole `DeclarativeDataset` and commit it. |
| `write_stream(records, tasks=(), *, annotations=())` | Compile from iterators, so a build never holds the whole hierarchy in memory. |
| `manifest` | The committed version's manifest. It raises before anything has been written. |

`write_stream` is what a large corpus uses: records are consumed one at a time, their control rows
go into batch inserters, and their values stream into the values plane as they arrive. A task whose
records were already dropped names them with a `RecordRef`, which the writer resolves in one join
once every record has landed.

## Constructor options

```python
TimeFWriter(root, metadata, *, values_backend="parquet", **values_options)
```

| Argument | Meaning |
| --- | --- |
| `root` | The registry root. The version lands at `<root>/<dataset_id>/<version>/`. |
| `metadata` | The dataset's `DatasetMetadata`. Its id and version choose the output directory. |
| `values_backend` | `"parquet"` (default) or `"zarr"`. |
| `values_options` | Byte budgets and codec settings, forwarded to the backend. |

The Parquet backend takes `shard_target_bytes` (default 128 MiB), `row_group_target_bytes` (4 MiB),
`chunk_max_bytes` (1 MiB), `compression` (`zstd`), and `compression_level`. The Zarr backend takes
`chunk_max_bytes` and `shard_target_bytes`; row groups mean nothing to it, so it does not accept
that budget rather than accepting one it would ignore.

Constructing a writer over an already-committed version raises `TimeFValidationError`.

## Values backends

=== "Parquet (default)"

    Byte-budgeted rotating shards with a per-modality encoding chosen from measured cardinality.
    This is the portable path: no extra to install, and any Parquet tool can open a shard.

=== "Zarr"

    One chunked typed array per modality, appended along time. Install it with
    `pip install 'timenet[zarr]'`. The writer imports zarr lazily, so a Parquet-only install never
    pulls it in, and a reader that opens a Zarr-written version without the extra gets a clear
    message.

Either way the control plane is the same, and `values_artifacts` records which backend wrote each
file, so a reader dispatches without a per-chunk tag.

## Chunking and encodings

The writer splits each signal into chunks of at most `chunk_max_bytes`, buffers chunks until they
reach `row_group_target_bytes`, and flushes them as one row group. A shard rotates once it reaches
`shard_target_bytes`, and a row group never spans two shards, so a chunk locator is exact.

The values encoding is measured, not pinned: dictionary at or below 65,536 distinct values,
BYTE_STREAM_SPLIT above that for floats, plain above that for strings and integers. The decision is
taken once per `spec_type` off buffered data, so rebuilding an unchanged source writes the same
bytes. [The format page](timef-format.md#choosing-the-values-encoding) has the measurements behind
the rule.

## Validation

The control plane ships without primary keys, foreign keys, or indexes, so the writer checks every
invariant they would have enforced against the finished database before it publishes. Each check is
one bulk anti-join, and the first failure aborts the build with nothing committed. They cover:

- surrogate ids are dense: 0, 1, 2, and so on, with no gap and no repeat;
- an `external_id`, where one is set, is unique within its table;
- every link, chunk, and item names a row that exists;
- a source sits in the same record as its parent, and its `path` extends its parent's;
- a Parquet chunk carries a row offset and a Zarr chunk does not;
- no duplicate `(source_id, signal_id)` link, `(signal_id, chunk_idx)` chunk, or values artifact.

Dense ids are not a formality: they are what lets `TimeFTorchDataset` treat a position as a record
id and a `DataLoader` worker take its slice with a modulo.

## Commit protocol

Everything is built in a `<version>.tmp-<uuid>` staging directory beside the final one. The walk
streams values into the values plane and rows into the database together, inside one transaction.
The writer then validates, commits the transaction, writes `manifest.json`, and moves the staging
directory into place with one rename. A reader either sees a complete version or sees nothing.
Leaving the context manager on an exception removes the staging directory.

## Manifest

`write` returns the committed `Manifest`, and `writer.manifest` hands it back afterwards. It names
`control_db` and every values part with a `sha256:` checksum and a size, carries the counts the
build recorded, and states the values backend and the encoding chosen per modality. See
[Manifest](manifest.md).

---

See the [API reference for `timenet.control_plane`](api/control_plane.md) for the full symbol
listing.
