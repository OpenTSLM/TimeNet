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
handle contains the parsed manifest and access to the version files. The reader never imports or
runs connector code.

```python
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion, open_registry

# Read through a registry.
version = open_registry("~/timenet/registry").open_version(
    "timenet/hello-world"
)

# You can also open a version directory directly.
local_version = DatasetVersion.open_local(version_dir)

with TimeFReader(version) as reader:
    dataset = reader.read()
    values = dataset.records[0].signals[0].to_arrow()
```

Use the reader as a context manager when possible. `close()` releases open value files and decoded
chunk caches.

## What loads when

Creating the reader does not open the control database. The first request for records, tasks, or
annotations opens `control.duckdb` in read-only mode.

Records load with their Sources, Signals, axes, and annotations. Signal values stay lazy. A call to
`to_arrow()`, `to_numpy()`, or `read_steps()` reads only the required chunks from Parquet or Zarr.

Task prompts, rationales, and text targets use `TextHandle`, which holds one string. Annotation
values use `TextHandle` for a string and `TextListHandle` for a list of strings; `isinstance` tells
the two apart. Call `read()` on a handle to obtain its native string or list. Optional prompts and
rationales remain `None` when absent; numeric and boolean values remain native Python scalars. Use
`read_value(value)` from `timenet.types` when a value may be a text handle or a native scalar.
`read_values(values)` does the same for each item, such as a task's targets, and returns a tuple, or
`None` for `None`.

Text read from a release uses the same handle types. Packed text stays in its payload file until
`read()` or a comparison of content needs it, and its handle keeps nothing: `is_loaded` is false,
and each read goes to the payload again. Schema derivation and modality inference do not read
packed text. The Arrow `read_task_table()`, `read_target_table()`, and `read_annotation_table()`
APIs return native values. The reader keeps up to 64 MiB of decoded payload files, the newest one
even when it is larger, and rejects a locator whose payload file the manifest does not list. A
release whose manifest lists no payload files is read without any locator lookups.

Empty text handles are false in truth tests. An empty prompt remains a handle whose `read()`
returns `""`. An absent prompt remains `None`. Truth tests use the handle's length without reading
its value.

Text handles compare by content, with handles of the same kind and with native values of their
type: a `TextHandle` with strings, a `TextListHandle` with lists. A task therefore equals its
read-back copy regardless of where its text is stored. A `TextHandle` never equals a
`TextListHandle` or a list. A handle equals itself, and handles of different lengths compare
unequal, without a read. Two handles for the same row of one payload file are also equal without
a read, even from different readers, since the manifest's checksum identifies the file. Comparing
other handles of equal length reads both values. Text handles are not hashable.

`iter_records(record_ids=...)` yields records one at a time. It takes int record IDs. It does not
create a complete `TimeFDataset`.

Tasks have three useful read paths:

- `tasks` loads and caches every task.
- `iter_tasks(records)` yields tasks for selected Records in bounded batches. Its `split`,
  `required_modalities`, and `supported_modalities` arguments filter in DuckDB before any task is
  built, as they do on `read_task_table()` and `read_target_table()`.
- `read_task_table()`, `read_target_table()`, and `read_annotation_table()` return Arrow tables
  without creating Python task objects. Their ID columns are `int64`.

Objects keep their identity within one reader. For example, a task input is the same `Record`
instance that `iter_records()` returned.

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
| `read()` | Create the complete `TimeFDataset`. |
| `iter_records()` | Yield Records without building a dataset. |
| `iter_tasks(records, split=...)` | Yield tasks for selected Records in bounded batches, optionally one split. |
| `read_task_table()` | Return the task table with public IDs. |
| `read_target_table()` | Return ordered task targets with public IDs. |
| `read_annotation_table()` | Return annotation occurrences with public IDs. |
| `verify()` | Check every manifest-listed file against its size and checksum. |
| `metadata` | Return the dataset metadata. |
| `schema` | Return the dataset schema. |
| `tasks` | Return all reconstructed tasks. |
| `values_backend` | Return the selected values backend. |

## Errors and integrity

`DatasetVersion.open_local()` and `open_version()` require a valid `manifest.json`. They raise
`TimeFFormatError` for an invalid or unsupported manifest.

Other files remain lazy. A missing `control.duckdb` therefore fails on the first control-plane
request. A missing values file fails when a Signal reads the affected chunk. Errors include the
relevant file or table context.

Call `verify()` to check all files before reading data. It raises `TimeFFormatError` when a file is
missing or does not match the manifest.

## Round-trip guarantee

For valid writer input, the reader restores Records, Sources, Signals, annotations, and tasks. It
also restores task relationships and Signal values with their original dtype and shape. Public IDs
are the durable references. Entity IDs are ints. The dataset ID is its `org/name` string. The
reader does not keep the Python object identity from the writer process.

See the [API reference for `timenet.reader`](api/reader.md) for all members.
