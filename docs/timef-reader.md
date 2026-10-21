---
icon: lucide/book-open
description: "Read a TimeF version into a TimeFDataset without running connector code."
tags:
  - reference
  - reader
---

# TimeFReader

`TimeFReader` opens a committed TimeF version and reconstructs a
[`TimeFDataset`](timef-dataset.md). It reads through a [`DatasetVersion`](registry.md) handle. The
handle contains the parsed manifest and local or remote filesystems for the version and its physical sources.
The reader never imports or runs connector code.

```python
from timenet.reader import TimeFReader
from timenet.registry import open_local_version, open_registry

# Read through a registry.
version = open_registry("~/timenet/registry").open_version(
    "timenet/hello-world"
)

# You can also open a version directory directly.
local_version = open_local_version(version_dir)

with TimeFReader(version) as reader:
    handle = reader.read()
    dataset = handle.read()

values = dataset.records[0].signals[0].to_arrow()
```

Use the reader as a context manager when possible. `close()` releases open value files and decoded
chunk caches.

## What loads when

Preparation finishes before the reader receives its version handle.
Creating the reader does not open the control database. `reader.read()` and `reader.get_handle()`
return lazy dataset handles. The first structural read opens `control.duckdb` in read-only mode.

Calling `read()` on a dataset handle loads the complete dataset into memory. Calling `read()` on an
entity handle loads the requested object and its dependencies. Field handles read selected fields.
Full reads reuse the completed dataset until the reader closes. The returned dataset remains usable
after the reader closes.

With `iter_records()`, Records load with their Sources, Signals, axes, and annotations. Signal values
stay lazy. A call to `to_arrow()`, `to_numpy()`, or `read_steps()` reads only the required chunks from
Parquet or Zarr.

Text fields return `TextValue` handles, regardless of their size or storage location. Call
`task.prompt.read()`, `task.rationale.read()`, or a text target's `read()` method to obtain a string.
An absent prompt or rationale remains `None`. Annotation strings and string lists also use handles;
numeric and boolean annotation values remain native Python values.

Constructors still accept strings, such as `AnswerTask(prompt="Question?", targets=("Answer.",))`.
They wrap the text in memory. Loaded objects use the same interface, with packed text read only when
`read()` is called. Schema derivation and modality inference do not read packed text.

Use `read_value(value)` from `timenet.types` when a value may be a text handle or a native scalar.
The Arrow `create_task_table()`, `create_target_table()`, and `create_annotation_table()` APIs return
native values. The reader caches up to eight payload files.

Irregular timestamp loaders read and validate their packed arrays on first access.

Use `scan()` for sequential reads of many signals with parallel Parquet decoding:

```python
with TimeFReader(version) as reader:
    with reader.scan() as dataset:
        for record in dataset.records:
            for signal in record.signals:
                values = signal.to_arrow()
```

Entering the context reconstructs the catalog and starts bounded prefetch with four workers.
Signal arrays and stored text remain lazy. The cache keeps a bounded number of decoded groups;
the reader does not retain every signal array. Leaving the context releases its workers, including
after an early exit or error. A dataset handle also provides `scan()`.

Entity handles support `read()` and field handles through `get_handle(field)`. A Signal handle also
supports `to_arrow()`, `to_numpy()`, and `read_steps()`. A lookup by input alias or exact parent
release resolves the original parent ID through the local catalog.

```python
from timenet.types import ObjectKind

with TimeFReader(version) as reader:
    dataset = reader.read()
    task = dataset.get_handle(ObjectKind.TASK, "question-1")
    prompt = task.get_handle("prompt").read()
```

`iter_records(record_ids=...)` yields records one at a time. Its Signal values remain lazy.
It does not create a complete `TimeFDataset`.

The Arrow table APIs resolve external text automatically. The reader caches up to eight packed
payload files. Timestamp payload reads validate their length, endpoints, ordering, and nulls.

Tasks have three useful read paths:

- `tasks` loads and caches every task.
- `iter_tasks(records)` yields tasks for selected Records in bounded batches. Its `split`,
  `required_modalities`, and `supported_modalities` arguments filter in DuckDB before any task is
  built, as they do on `create_task_table()` and `create_target_table()`.
- `create_task_table()`, `create_target_table()`, and `create_annotation_table()` return Arrow tables
  without creating Python task objects.

Tasks reuse the Record objects supplied to `read_tasks(records)` or `iter_tasks(records)`.
Separate entity-handle reads can construct separate Python objects.

## Type reconstruction

The manifest describes the dataset schema and its files. The DuckDB control database stores the
object hierarchy and relationships. The reader combines both sources to rebuild standard TimeNet
types.

Reading an annotation text handle returns its stored string or string list. Scalar annotation
values retain their native types. Spans return as `TimePoint`, `TimeInterval`, `StepPoint`, or
`StepInterval`. The task type selects a built-in task class, and `from_tasks`
contains resolved Task objects.

## Value reads

Each Signal has ordered chunk locations in `control.duckdb`. For Parquet, a location identifies a
shard, row group, and row. For Zarr, it identifies an array path and a range. Range reads select only
the chunks that intersect the requested range.

Scalar values use Arrow arrays. Multidimensional values use `pa.FixedShapeTensorArray`.
`to_numpy()` preserves the declared dtype and trailing shape. The reader caches open Parquet files
and decoded Zarr chunks until it closes.

## API

| Member | Description |
| --- | --- |
| `read()` / `get_handle()` | Return a lazy dataset handle. |
| `DatasetHandle.read()` | Load the complete `TimeFDataset` and all its values. |
| `scan()` / `DatasetHandle.scan()` | Reconstruct a lazy dataset and prefetch Parquet signal groups with four workers. |
| `iter_records()` | Yield Records without building a dataset. |
| `iter_tasks(records, split=...)` | Yield tasks for selected Records in bounded batches, optionally one split. |
| `create_task_table()` | Return the task table with public IDs. |
| `create_target_table()` | Return ordered task targets with public IDs. |
| `create_annotation_table()` | Return annotation occurrences with public IDs. |
| `verify()` | Check every manifest-listed file against its size and checksum. |
| `metadata` | Return the dataset metadata. |
| `schema` | Return the dataset schema. |
| `tasks` | Return all reconstructed tasks. |
| `values_backend` | Return the selected values backend. |

## Errors and integrity

`open_local_version()` and `open_version()` require a valid `manifest.json`. They raise
`TimeFFormatError` for an invalid or unsupported manifest.

Local preparation requires complete releases. Remote files remain lazy: a missing `control.duckdb`
fails on the first structural read, and a missing values file fails when a Signal reads its chunk.
Errors include the relevant file or table context.

Call `verify()` to check all files before reading data. It raises `TimeFFormatError` when a file is
missing or does not match the manifest.

## Round-trip guarantee

For valid writer input, the reader restores Records, Sources, Signals, annotations, and tasks. It
also restores task relationships and Signal values with their original dtype and shape. Public IDs
are the durable references. Python object identity from the writer process is not preserved.

See the [API reference for `timenet.reader`](api/reader.md) for all members.

## Inherited storage

Each child owns a complete catalog. Its chunk and external-value locators name the ultimate physical
owner directly. Reads never attach parent databases or traverse inheritance chains.

The writer computes record scan order from physical chunk locations. This keeps full scans close to
storage order, even when imported IDs differ from parent IDs. Requested Record IDs retain the
caller's order. `verify()` checks both child files and referenced payload files.

Eager full reads and explicit `scan()` contexts use a stored list of required Parquet groups.
Four workers decode a bounded window in parallel. This list is computed during the write.
Lazy handles and range reads outside a scan context do not prefetch.
