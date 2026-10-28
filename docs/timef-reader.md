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
from timenet.registry import open_local_version, open_registry

# Read through a registry.
version = open_registry("~/timenet/registry").open_version(
    "timenet/hello-world"
)

# You can also open a version directory directly.
local_version = open_local_version(version_dir)

with TimeFReader(version) as reader:
    dataset = reader.read()
    values = dataset.records[0].signals[0].to_arrow()
```

Use the reader as a context manager when possible. `close()` releases open value files and decoded
chunk caches. For a remote release, the reader fetches and verifies `control.duckdb` once, and after
`close()` its next read reopens the verified local copy. If that copy has been removed in the
meantime, the reader fetches and verifies the catalog again.

Catalog connections close before the process forks, because DuckDB cannot share a database across a
fork. Each one opens again on its next query, in the parent and in a forked data-loader worker
alike, so a task stream that was in progress when the process forked continues. Forking while other
threads use readers is not supported: a read that the fork's closing cuts short in another thread
raises `TimeFReaderError`. Use the `spawn` or `forkserver` start method there.

## What loads when

Creating the reader does not open the control database. The first request for records, tasks, or
annotations opens `control.duckdb` in read-only mode.

Records load with their Sources, Signals, axes, and annotations. Signal values stay lazy unless the
read's load plan asks for them, as [Loading values at read time](#loading-values-at-read-time)
describes. A call to `to_arrow()`, `to_numpy()`, or `read_steps()` on a lazy Signal reads only the
required chunks from Parquet or Zarr.

Task prompts, rationales, and text targets use `TextHandle`, which holds one string. Annotation
values use `TextHandle` for a string and `TextListHandle` for a list of strings; `isinstance` tells
the two apart. Call `read()` on a handle to obtain its native string or list. Optional prompts and
rationales remain `None` when absent; numeric and boolean values remain native Python scalars. Use
`read_value(value)` from `timenet.types` when a value may be a text handle or a native scalar.
`read_values(values)` does the same for each item, such as a task's targets, and returns a tuple, or
`None` for `None`.

Text read from a release uses the same handle types. Unless the read loads text, packed text
stays in its payload file until `read()` or a comparison of content needs it, and its handle keeps
nothing: `is_loaded` is false, and each read goes to the payload again. Schema derivation and
modality inference do not read packed text. The Arrow `read_task_table()`, `read_target_table()`,
and `read_annotation_table()` APIs return native values. The reader keeps up to 64 MiB of decoded
payload files, the newest one even when it is larger, and rejects a locator whose payload file the
manifest does not list. A release whose manifest lists no payload files is read without any locator
lookups.

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

Irregular time offsets are packed as well, one array per axis. The first `time_offsets_us()` call
on a Signal reads its axis's array and checks it against the axis and against the digest its
locator records. Within one read, the Signals on one axis share the checked array, so later calls
on any of them read nothing.

`iter_records(record_ids=...)` yields records one at a time. It takes int record IDs. It does not
create a complete `TimeFDataset`.

Tasks have three useful read paths:

- `read_tasks(records)` builds the tasks of selected Records, or every task, in one pass.
- `iter_tasks(records)` yields tasks for selected Records in bounded batches. Its `split`,
  `required_modalities`, and `supported_modalities` arguments filter in DuckDB before any task is
  built, as they do on `read_task_table()` and `read_target_table()`.
- `read_task_table()`, `read_target_table()`, and `read_annotation_table()` return Arrow tables
  without creating Python task objects. Their ID columns are `int64`.

One call returns one consistent set of objects. A Record that two tasks use is one object, and the
Signals on one irregular axis share one handle to its time offsets. `read_tasks(records)` and
`iter_tasks(records)` reuse the Record objects you pass: a task input is the same `Record` instance
that `iter_records()` returned. Separate calls build separate objects, because the reader keeps no
object caches; keep a result if you need it again.

`iter_tasks()` builds each batch on its own. The tasks of one batch share the objects the batch
builds, as one call does, and batches share nothing but the Records you pass. A Record that tasks
of two batches use arrives as two separate objects with the same ID. So does a parent task that
one batch builds for a task's `from_tasks` and a later batch yields. In return, the stream keeps
none of a batch's objects once it has yielded the batch: it holds one batch of objects and that
batch's task IDs at a time, and selects each next batch after the last task ID it yielded. The
reader itself keeps compact chunk locators for the Signals it has resolved until it closes. By
default, a batch holds up to 5,000 tasks (`TASK_BATCH_SIZE`). Pass `iter_tasks(batch_size=...)` for
larger or smaller batches. With an eager plan the stream holds every value one batch loaded, so a
smaller batch bounds the memory its arrays take.

## Loading values at read time

`read()`, `read_tasks()`, and `iter_tasks()` take a `load=` plan that says which kinds of stored
values the read loads into the objects it builds. A `LoadPlan` has one field per kind of value:
`signal_values`, `time_offsets` for irregular axes, and `text` for packed prompts, rationales, text
targets, and annotation values. Each field is `Loading.LAZY`, the default, or `Loading.EAGER`.

```python
from timenet.reader import Loading, LoadPlan, TimeFReader

with TimeFReader(version) as reader:
    # Every value loads; the dataset needs no storage once the reader closes.
    dataset = reader.read(load=LoadPlan.ALL_EAGER)
    # Prompts and targets load; signal arrays stay lazy.
    tasks = reader.read_tasks(load=LoadPlan(text=Loading.EAGER))
```

The read loads the values it needs in one batch before it returns. Arrays find their chunk
locations a batch of Signals at a time and load grouped by source release and file. The time
offsets of an axis load once. A loaded handle has `is_loaded` set and keeps its value in a compact
copy, so the decoded files and chunks it came from are not kept alive with it.

A plan applies only to the objects the read builds and to what they reference directly. A task's
plan reaches its text and annotations, the records it uses as inputs, candidates, or targets, the
Signals it targets, and the annotations it references. It does not reach tasks the read builds only
as another task's `from_tasks`, nor the Records you pass to `read_tasks(records)` or
`iter_tasks(records)`, which stay as they are. A read never changes objects you already hold. To
load more of them, read again with a larger plan; the new read returns new objects. A failed read
returns nothing.

`iter_tasks()` loads each batch before it yields the batch, as far as the plan reaches from the
batch's tasks. What a batch reaches only through its tasks' `from_tasks` parents, which is the
parents, their records, and their annotations, it builds but does not load, unless a task of the
batch uses it too. A later batch that yields such a parent, or uses such a record or annotation,
builds its own copy and loads it.

Values on lazy handles stay readable. `to_numpy()`, `read_steps()`, and a text handle's `read()`
read them through the reader, which reopens its files on demand after `close()`.

Writing a read dataset reuses its stored payloads, whether they were loaded or not: array chunks,
time offsets, and packed text keep their files in the release that stores them. The new release
therefore depends on its source even when every value was loaded, and its readers must be able to
find the source release.

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
| `read(load=...)` | Create the complete `TimeFDataset`, loading the values the plan names. |
| `iter_records()` | Yield Records without building a dataset. |
| `read_tasks(records, load=...)` | Build the tasks of selected Records, or every task, in one pass. |
| `iter_tasks(records, split=..., load=..., batch_size=...)` | Yield tasks for selected Records in bounded batches, optionally one split. |
| `read_task_table()` | Return the task table with public IDs. |
| `read_target_table()` | Return ordered task targets with public IDs. |
| `read_annotation_table()` | Return annotation occurrences with public IDs. |
| `verify()` | Hash every file the release reads, its own and its pinned payload source files. |
| `metadata` | Return the dataset metadata. |
| `schema` | Return the dataset schema. |
| `values_backend` | Return the selected values backend. |

## Opening a version

`open_local_version()` and a registry's `open_version()` open a version eagerly, but read only
metadata. Opening checks that:

- `manifest.json` is valid, with a supported format version;
- every file the manifest lists, `control.duckdb` included, is present with its recorded size;
- each payload source, a release whose payload files this version reuses, is found and still
  records each pinned file as the version pins it; and
- each pinned file of a local payload source is present with its recorded size.

Opening hashes nothing and reads no values. A remote version's file sizes come from its manifest,
and its reads and downloads check them.

A version lists every payload source it reads from, so opening resolves only those. A source's own
sources are not opened: a version that reads only files its source stores opens without the
releases that source reuses.

| Problem | Error | The message names |
| --- | --- | --- |
| A payload source is missing | `TimeNetDatasetNotFoundError` | The source and the root searched |
| A payload source differs from its pins | `TimeFFormatError` | The source and the first differing file |
| A file is missing or has another size | `TimeFFormatError` | The file, its expected size, and its actual size |

An invalid or unsupported manifest raises a validation error, and a missing one raises
`FileNotFoundError`.

### Where payload sources are found

Payload sources resolve relative to a registry root. By default, `open_local_version()` takes the
root that holds the version as `<root>/<org>/<name>/<version>`, so a version and its sources keep
working when they move to another root together. Pass `sources_root=` to find them under another
root. A registry's `open_version()` finds them in that registry.

A version without payload sources opens from any directory. A version with payload sources that is
not stored as `<root>/<org>/<name>/<version>` needs `sources_root=`, and opening it without one
raises `TimeFValidationError`.

## Errors and integrity

Values remain lazy. A values file that changes after opening fails when a Signal reads the affected
chunk. Errors include the relevant file or table context.

A catalog query raises `TimeFFormatError` when the catalog lacks a table, column, or type that the
read expects, and `TimeFReaderError` when the reader's connection closed while the read ran, through
`close()` or a fork in another thread. Other DuckDB errors, such as I/O failures, running out of
memory, or an interrupt, propagate unchanged.

Call `verify()` to hash every file before reading data: the version's own files, and the files it
pins on each payload source. It raises `TimeFFormatError` when a file is missing or does not match
its manifest record or pin.

## Round-trip guarantee

For valid writer input, the reader restores Records, Sources, Signals, annotations, and tasks. It
also restores task relationships and Signal values with their original dtype and shape. Public IDs
are the durable references. Entity IDs are ints. The dataset ID is its `org/name` string. The
reader does not keep the Python object identity from the writer process.

See the [API reference for `timenet.reader`](api/reader.md) for all members.
