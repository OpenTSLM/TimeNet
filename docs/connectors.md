---
icon: lucide/cable
description: "Write a BaseConnector to turn a raw source into a TimeFDataset."
tags:
  - guide
  - connectors
---

# Connectors

A connector is the unit of dataset integration. Each dataset has one connector class. A connector
fetches raw data and converts it into a [`TimeFDataset`](timef-dataset.md). A connector has no
knowledge of the registry, the engine, or other connectors. A child connector reads its parent
releases only through the readers that the engine passes to it. The consumer SDK never imports
connector code. On a local registry, `TimeNet.load` builds a missing dataset through the
`timenet.builders` entry point. Pass `auto_build=False` to turn this off.

`BaseConnector` is the connector contract in the `timenet` package. Concrete connectors ship in
`timenet-connectors`, beside their [dataset cards](manifest.md).

---

## `BaseConnector`

```python
from pathlib import Path
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset

class MyConnector(BaseConnector[MyRawRef]):
    def download(self, cache_dir: Path) -> list[MyRawRef]: ...
    def convert(self, raw_refs: list[MyRawRef]) -> TimeFDataset: ...
```

The two stages stay separate. This lets the engine drive
`download -> convert -> derive_schema -> store`.

| Method | Nature | Contract |
| --- | --- | --- |
| `download(cache_dir)` | I/O only | Fetch or find raw files and return lightweight references. The method is idempotent and does not parse data. |
| `convert(raw_refs)` | local work | Parse the references into a `TimeFDataset`. Do not use the network. |
| `compose(raw_refs, context)` | local work | Build a child dataset from the references and the opened parent releases. The default calls `convert(raw_refs)`. |

You must implement `convert`. Unless the dataset card declares parent releases, it is the only
required method. If the card declares parent releases, you implement `compose` instead, as
[Compose a child dataset](#compose-a-child-dataset) describes. `download` depends on I/O, so it has
two forms. You can override `download(cache_dir)` for a synchronous fetch. Alternatively, you can
define `async download_async(cache_dir)` to fetch artifacts at the same time. Implement only one of
these two forms. The engine always calls the synchronous `download()`. Its default implementation
runs `download_async` to completion. As a result, an async connector needs no event loop code of its
own.

`metadata()` is a concrete method that reads the dataset card. The engine owns storage and calls the
writer after conversion.

- `metadata()` reads and validates the dataset's [`dataset.yaml` card](manifest.md) from disk, through
  `DatasetMetadata.from_yaml`. This method does file I/O. Override it only to point to a different
  card. `metadata().dataset_id` must match the connector's built id.

A connector takes no constructor arguments. Configuration comes from environment variables that
`__init__` reads. `TRaw` is the reference type that the connector defines, for example a path, a small
dataclass, or an S3 key. The connector is generic over `TRaw`:
`class MyConnector(BaseConnector[MyRawRef])`.

If an assumption about your source data, downloads, or parents does not hold, raise
`TimeNetConnectorError` from `timenet.errors`.

---

## Reusing descriptors

Each Signal has exactly one owning Source. Do not attach one Signal to several Sources or Records.
TimeNet rejects that ambiguous ownership.

Signals can share immutable `TimeAxis` and `TimeSeriesSpec` objects. Annotation content can also be
reused. Each call to `annotate()` creates a separate occurrence for the selected owner.

`Annotation` and the Task classes are final. To reuse an annotation, keep a constant or write a
builder function, as [Reuse content](data-model/annotations.md#reuse-content) shows. To reuse a
prompt, write a function that builds the Task, as
[Built-in task types](data-model/tasks.md#built-in-task-types) shows.

---

## Assigning dataset splits

Use `timenet_connectors.splitting` during conversion to assign source samples to partitions.
Every implementation inherits from `Splitter[T]` and returns `DatasetSplits[T]`. Access partitions
by `Split` name; iteration over the result yields partition names, not positional train/test values.
The splitter keeps the source objects unchanged and preserves their order within each partition.

Preserve official partitions when the source provides them by setting `split` on each task
directly. No splitter is needed:

```python
from timenet.types.splits import Split

record = make_record(sample)
dataset.add_record(record=record)
task = make_task(sample, record, split=Split.TRAIN)
dataset.add_task(task=task)
```

Translate source-specific partition names into `Split` values in the connector.

For a dataset without official partitions, explicitly choose a category to balance and a grouping
key that prevents related samples from crossing partitions:

```python
from timenet.types.splits import Split
from timenet_connectors.splitting import StratifiedSplitter

def get_category(sample):
    return sample.category

def get_signal_id(sample):
    return sample.signal_id

splitter = StratifiedSplitter(
    train_folds=4,
    test_folds=1,
    stratify_by=get_category,
    group_by=get_signal_id,
    seed=42,
)
splits = splitter.split(samples)
train_samples = splits[Split.TRAIN]
test_samples = splits[Split.TEST]
print(splits.counts)
```

For three partitions, use `train_folds=8, validation_folds=1, test_folds=1`. Omitted validation is absent,
not an empty placeholder. Each configured count must be a positive integer.

`StratifiedSplitter` uses scikit-learn's
[`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
when `group_by` is absent, avoiding the dense group-by-category matrix of the grouped algorithm.
With `group_by`, it uses
[`StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
The counts directly specify how many folds each partition receives; their sum is the total
number of folds. For example, `train_folds=4, test_folds=1` uses five folds for an approximate 80/20 split;
`train_folds=8, validation_folds=1, test_folds=1` uses ten for 80/10/10. Fold assignment uses a fixed
train/test/validation order. Counts are used as given, without normalization: `train_folds=8, test_folds=2`
uses ten folds, while `train_folds=4, test_folds=1` uses five. Choose counts that the dataset's groups and
categories can support.

Stratification targets the same category proportions in each partition. A group can contain several
categories; all its samples stay together. Without `group_by`, each sample occurrence is independent.
Use the actual leakage boundary for grouping, such as a patient or source recording, rather than a
unique question or window ID. For temporal prediction, choose a time-based strategy instead.

Scikit-learn assigns samples or groups to folds, with `shuffle=True` and the configured seed. The wrapper
encodes category and group keys as scalar labels and maps the returned indices back to the original
samples. It contains no custom allocation or imbalance-scoring algorithm.

Nonempty input needs at least as many distinct groups as folds. Scikit-learn rejects inputs where
every category has fewer samples than folds; a rare category can produce a warning when other
categories have enough samples. Invalid inputs raise `TimeFValidationError`. Empty input returns
empty configured partitions. Sample counts and category proportions remain approximate when groups
have unequal sizes; inspect `splits.counts` and category distributions before publishing.

The same ordered input, callbacks, seed, and scikit-learn version reproduce the assignment; changing
input order or adding samples can change it. No global random state is modified. The connectors
package pins scikit-learn 1.9.1 because its allocation affects persisted dataset contents. Review
generated assignments and version affected datasets when upgrading the splitter implementation.

When constructing TimeNet tasks, copy the partition name onto each task:

```python
for split_name, samples_in_split in splits.items():
    for sample in samples_in_split:
        record = make_record(sample)
        dataset.add_record(record=record)
        task = make_task(sample, record, split=split_name)
        dataset.add_task(task=task)
```

Here `make_record` and `make_task` are the connector's own construction functions. A task refers to
its Record by object, so add the Record to the dataset before the task. Consumers then select stored
assignments through `dataset.get_train()`, `dataset.get_test()`, or `dataset.iter_tasks(split=...)`.
They do not run the splitter again.

TSQA dataset version `1.1.0` introduced stored splits. Version `1.2.0` generates new assignments
with scikit-learn 1.9.1: 80/10/10 train/validation/test using ten folds and seed 42, balanced by
the source `Task` category. Questions sharing identical parsed series values
stay together, regardless of JSON whitespace or integer/float notation. Channel layout is preserved;
a flat univariate series and the same series wrapped in one channel have the same grouping key.
This identifies exact duplicate signals, not overlapping windows or transformed versions of a signal.
These partitions are generated by TimeNet, not published benchmark splits from TSQA's authors.
Existing `1.0.0` and `1.1.0` artifacts stay unchanged. Publish `1.2.0` separately; do not overwrite
previously published splits or change a benchmark's pinned dataset version implicitly.

---

## Discovery and layout

The system finds connectors lazily, by dataset id. There is no central registry to maintain. A
concrete connector lives under `timenet_connectors/datasets/<org>/<name>/`. Use lowercase Python
package names. The package's `__init__.py` exposes `CONNECTOR`. Put `dataset.yaml` and an optional
`requirements.txt` beside it. The build tool imports only the requested connector package. Reusable
bases live under `timenet_connectors/bases/`. The next section describes the modules inside the
connector package.

## Connector layout

This section gives the default layout for connector code, so a reader finds the main flow in the
same place in each connector. The layout keeps three steps apart: read the source, build the
records, and build the tasks. Use it for a new connector. If another split reads more clearly for
one connector, use it and say why in the pull request. The responsibilities matter more than the
file names.

### A small connector

If a connector has fewer than about 250 lines, keep all of its code in `connector.py`. The number is
a guide, not a limit. Do not split one class or one closely related group of functions across files
only to stay under it. Put the code in this order, from top to bottom:

1. The constants, labels, and annotation keys.
2. The code that reads the source into typed rows.
3. The code that builds the records.
4. The code that builds the tasks.
5. The connector class.

The rules for a larger connector also apply inside this one file.

### A larger connector

If a connector has more lines, split it into these modules:

```
datasets/<org>/<name>/
  __init__.py       # re-exports CONNECTOR
  connector.py      # the connector class only: metadata, download, convert or compose
  source.py         # reads the raw files into typed rows and checks every source value
  records.py        # builds the records from the typed rows
  tasks.py          # builds the tasks from the records and the typed rows
  labels.py         # enums, label sets, annotation keys, and specs
  dataset.yaml
  requirements.txt  # optional
  tests/
```

If the dataset has no tasks, leave out `tasks.py`. If the connector has no fixed values, leave out
`labels.py`.

The imports between these modules go in one direction:

- `source.py` does not import `records.py` or `tasks.py`.
- `records.py` can import `source.py`.
- `tasks.py` can import `source.py` and `records.py`.
- Every module can import `labels.py`.

Keep connector class methods short. Use helper functions to read and check source data, build
records, and build tasks. A reader then sees the whole flow in a few lines.

### Source checks

The source checks live in `source.py`. In a small connector, the same helper functions live in
`connector.py`, and the rules in this section apply to them in the same way.

`source.py` turns each raw row into a typed row, for example a frozen dataclass. It checks each
value when it builds the row. If a value is not what the connector expects, it raises
`TimeNetConnectorError` with an error message that identifies the file and the row.

Record and task builders use checked, typed rows. Keep source checks in `source.py`, and let its
errors propagate. Each time the source is read, these checks run again.

A typed row holds small values only. Large values, such as signal arrays, stay in the source files.
The function that reads a large value also lives in `source.py`. It runs later, when the writer
loads the value, so it checks the values that it reads. For example, it checks the shape and the
data type. The rules for missing and non-finite values are separate:

- A missing value becomes null if the spec is nullable. If the spec is not nullable, a missing
  value raises `TimeNetConnectorError`. Never fill a missing value.
- If the source documents NaN as a missing value, treat it as one. Otherwise a NaN raises
  `TimeNetConnectorError`.
- Every value that is present must be finite. Infinity raises `TimeNetConnectorError`, also when
  the spec is nullable.

`records.py` only passes this function to `Signal.from_loader`. All code that reads the source is
then in `source.py`.

### Where to put each kind of code

| Code | Module |
| --- | --- |
| Facts about the source release, such as URLs, the revision, checksums, and expected counts | `source.py`, as constants next to the code that checks them. A fact that a base class reads from the connector class, such as `SOURCE_REVISION` for `BaseHuggingFaceConnector`, stays on the class. |
| Checks on one row | `source.py`, when it builds the typed row |
| Checks across rows, such as duplicates or contradictions | `source.py`, in a full scan of the source before `convert` builds records. Keep only the information needed for these checks, such as record keys. |
| Splits, folds, and classes that the source defines | `labels.py` if they are fixed values, `source.py` if the connector reads them from a file |
| Dataset annotations used by a task stream | `records.py`, in one function that `convert` calls before it sets the stream |
| Helpers for the users of the dataset | A module of their own, exported from `__init__.py` |

### The flow in `convert`

If the connector builds a list of tasks, `convert` reads like this:

```python
def convert(self, raw_refs: list[Path]) -> TimeFDataset:
    dataset = TimeFDataset(metadata=self.metadata())
    rows = tuple(iter_rows(raw_refs))  # source.py
    records = add_records(dataset, rows)  # records.py
    dataset.add_tasks(tasks=build_tasks(rows, records))  # tasks.py
    return dataset
```

`add_records` and `build_tasks` both read `rows`, so `rows` must be a tuple or a list. A generator
gives its rows to `add_records` only, and `build_tasks` then builds no tasks.

Task streams avoid keeping all tasks in memory. They must also avoid keeping all source rows in
memory. Read the source again each time the stream function is called:

```python
def convert(self, raw_refs: list[Path]) -> TimeFDataset:
    dataset = TimeFDataset(metadata=self.metadata())
    summary = check_rows(raw_refs)  # source.py: a full scan for the checks across rows
    records = add_records(dataset, iter_rows(raw_refs))  # records.py
    shared = add_stream_annotations(dataset, summary)  # records.py
    dataset.set_task_stream([AnswerTask], lambda: iter_tasks(raw_refs, records, shared))  # tasks.py
    return dataset
```

The task stream function can be called more than once, including by tests or later writes. Each
call must return a new iterator that rereads the source through `iter_rows` in `source.py`. The
stream keeps only the references, the records, and the shared annotations in memory.

Sometimes the loader of each record already holds its checked row, so the rows stay in memory
anyway. Then a reread saves no memory, and the stream can reuse the checked rows. Say so in a
comment next to `set_task_stream`.

Run all checks across rows in `check_rows`, before you call `set_task_stream`. The task stream
reads rows as needed instead of storing them all in a list. `check_rows`, `add_records`, and each
call of the stream read the source separately. Each pass checks its own rows again, but the checks
across rows run only once. Changes between reads can go undetected. TimeNet does not yet have a
shared helper to check that the source stayed unchanged.

The writer reads the dataset annotations before it reads the stream. Add every dataset annotation
used by the stream before you call `set_task_stream`.

The engine opens the parent releases and passes them to `compose` through `context.parents`. See
[Compose a child dataset](#compose-a-child-dataset). Use `source.py` to read the child's source
data, `records.py` to import and update parent records with `import_from`, and `tasks.py` to build
tasks from the typed rows and imported records.

### Tests

The tests in `tests/` follow the same split. Test `source.py` with bad source values, and make sure
that each one raises `TimeNetConnectorError`. Test the whole connector on a small fixture with a few
rows in the same format as the source files.

### Style

- Do not nest comprehensions. Use a loop or a named helper function.
- Use comments to explain reasons that are not clear from the code. Use plain words.
- Keep docstrings short. Say what a function returns and raises, not how it works line by line.

## Dependencies and credentials

A connector declares the libraries its source needs in a `requirements.txt` beside its
`dataset.yaml`. It is a plain pip requirements file:

```text
# Downloads the source dataset from the HuggingFace Hub.
huggingface_hub>=1.1
```

A build runs the connector in an environment built from that file. That environment is layered over
the same `timenet` and `timenet-connectors` that you run (see [Build & publish](build.md)).
A build installs nothing into your own environment. Two connectors that need incompatible libraries
do not collide.

Import those libraries lazily inside the connector anyway. If one is missing, raise a clear error.
That guard keeps `--no-isolation` usable while you write a connector.

List every dependency the connector needs, even one that another connector already names. There are
no shared requirement fragments. If several connectors share one file, an edit to that file can break
a connector that you did not check.

Credentials come from the environment. For the HuggingFace Hub, a token is read from `HF_TOKEN`
automatically (needed only for gated or private sources). Downloaded source files cache under
`<TIMENET_CACHE>` (see [client local state](client.md#local-state)).

A credentialed dataset (a PhysioNet DUA-gated one, for example) declares `access: credentialed`
and an `access_url` on its card. TimeNet never hosts such data, so it is build-your-own: get access
at the `access_url`, set the provider credential in the environment (the way `HF_TOKEN` already
works), and run `timenet-build build <id>` yourself. An `open`, `load`, or `download` of a
credentialed dataset from a hosted registry raises `TimeNetAccessError` instead of serving bytes.
The error names each `access_url` that the dataset needs, with the release that sets it. A child
dataset can need the access pages of its ancestors.

## Downloading artifacts

`timenet_connectors.download` has two async helpers. Each helper picks the backend from the scheme of
the URL. As a result, a connector never has to branch on `s3://` versus `http(s)://` itself:

- `download_files([Artifact(url, dest), ...])` downloads a list of files. The list can mix schemes freely.
  HTTP entries run at the same time, up to the limit of `max_concurrency`, and share one connection
  pool. S3 entries run one at a time. A single file uses a one-element list.
- `ensure_archive(url, target)` downloads a zip file and extracts it into `target`. This method is
  idempotent. A marker file records success. As a result, a later run reuses the extracted contents
  and skips the download. A successful extraction then deletes the archive.

Each `Artifact` takes optional `headers`, `cookies`, and a `sha256` value to validate the download.
`download_files` also takes batch-level `headers` and `cookies`. These apply to every HTTP request. The
per-artifact values merge over the batch-level values. S3 ignores all of these. HTTP downloads use
`httpx` and `aiofiles`, both base dependencies. They stream to disk and write atomically, through a
`.part` temporary file. If the SHA-256 value does not match, the download raises an error and leaves
nothing behind. HTTP downloads skip an existing destination. S3 downloads use boto3 and stay
synchronous. boto3 already parallelizes the transfer of a single object. `ensure_archive` also takes a
`filename` override, for URLs whose path has no usable name, for example a trailing slash or a
`/download` suffix. Call these helpers from `download_async`:

```python
from timenet_connectors.download import Artifact, ensure_archive, download_files

class MyConnector(BaseConnector[MyRawRef]):
    async def download_async(self, cache_dir):
        await download_files(
            [
                Artifact("https://host/a.csv", cache_dir / "a.csv"),
                Artifact("s3://bucket/b.csv", cache_dir / "b.csv"),
            ],
            # applied to every HTTP request
            headers={"Authorization": "Bearer …"},
        )
        await ensure_archive("https://host/records.zip", cache_dir)
        return [...]  # lightweight references into cache_dir
```

An ambient sink reports progress, through `timenet_connectors.download.progress`. The download helpers
emit `DownloadProgress` events, and the `timenet-build` CLI renders these events. As a result,
downloads show progress without a `progress` argument passed through the connector. On a terminal,
the events render as live progress bars, one row per file, that update at the same time. A Hugging Face
Hub download shows as one row for all its files. Piped output
falls back to throttled text lines. The `--quiet` flag silences all output.

PhysioNet connectors extend `BasePhysioNetConnector` for WFDB record I/O. They use `ensure_archive` to
pull their database archive.

## Example connectors

- `timenet/hello-world` is a synthetic, offline reference connector. It needs no network and produces
  a deterministic dataset for examples and round-trip tests. It covers shared descriptors, windowed
  Signals, annotations, and several task types.
- `chengsenwang/tsqa` is a time-series QA dataset. Each row's series becomes a Signal, and each
  row's question and answer become an `AnswerTask`. It downloads the Hub's parquet conversion of
  the dataset at a pinned commit, through `BaseHuggingFaceConnector`, so its `requirements.txt` names
  `huggingface_hub`.
- `physionet/ptb-xl` is a base dataset without tasks. It holds the PTB-XL recordings with their
  clinical annotations, for children to import from.
- The connector of `physionet/ecg-qa-cot` composes the dataset from `physionet/ptb-xl`. It imports
  the recordings and adds questions with chain-of-thought rationales.

```bash
timenet-build build timenet/hello-world             # offline, synthetic
timenet-build build chengsenwang/tsqa               # live, from the Hub
timenet-build build chengsenwang/tsqa --keep-cache  # keep the raw sources
```

A successful build removes the dataset's raw download cache, at `<TIMENET_CACHE>/<dataset_id>`. Only
`convert` reads the sources, and they are often several times the size of the dataset they produce.
To keep them, pass `--keep-cache`, or `keep_cache=True` to `timenet_connectors.build()`. This helps
while you write a connector for a large source, because each rebuild downloads the source again.

Keeping `download` and `convert` apart makes a connector testable offline. `convert` takes raw
references and does not touch the network. A test can therefore pass small local inputs or synthetic
references directly to `convert()` and skip `download`.

After the build, you can load and inspect a dataset with the SDK. See `examples/load_tsqa.py`. This
example loads a dataset and calls `describe()` to print its identity, its counts, its columns per
spec, and a record preview.

## Compose a child dataset

A child dataset reuses objects of other datasets, its parent releases. Its card declares each
parent release under `parents`, by a parent alias. A parent alias is the name that the child gives
the parent:

```yaml
parents:
  ptbxl: physionet/ptb-xl@1.0.0
```

The connector of a child overrides `compose(raw_refs, context)`. The default `compose` calls
`convert(raw_refs)`. If the card declares parents and the connector does not override `compose`,
the build raises before it opens a parent or downloads anything.

Before the connector runs, the build prepares each parent release in the output registry, as
[Prepare parent releases](build.md#prepare-parent-releases) describes. The engine then opens each
parent as a `TimeFReader` and passes the readers in a `BuildContext`, by parent alias. The connector
copies objects into the child with `dataset.import_from()`:

```python
from timenet.connectors.context import BuildContext
from timenet.dataset import TimeFDataset
from timenet.types.reference_keys import ParentAlias


def compose(self, raw_refs, context: BuildContext) -> TimeFDataset:
    dataset = TimeFDataset(metadata=self.metadata())
    ptbxl = context.parents[ParentAlias("ptbxl")]
    # Find parent Records by a key of the source in their metadata.
    selected = [
        record.id
        for record in ptbxl.iter_records(with_annotations=False)
        if record.metadata["ecg_id"] in {1, 2}
    ]
    # Import them by the IDs that the parent gives them.
    imported = dataset.import_from(ptbxl, records=selected)
    for copy in imported.records.values():
        copy.metadata["cohort"] = "selected"
    return dataset
```

[Importing from a parent](timef-dataset.md#importing-from-a-parent) describes what an import copies,
how a later import reuses the copies, and how to remove imported objects.

After `compose` returns, the engine records each declared parent in the dataset: its card and its
lineage. It does this also for a parent that the connector imported nothing from. A connector
therefore does not call `import_from()` only to record a parent. The card of the child keeps the
access that its author declared. The effective access level of the child is the strictest level of
its card and of the cards in its lineage, as [The parent](timef-dataset.md#the-parent) describes.

The copies keep handles to the stored values of the parent. The engine therefore writes the dataset
before it closes the parent readers. A test must do the same. Open the parents with
`open_parents()` from `timenet.engine`. Write the composed dataset inside the `with` block:

```python
from timenet.engine import open_parents, store_dataset
from timenet.registry import LocalRegistry

connector = MyConnector()
with open_parents(connector.metadata(), LocalRegistry(root)) as context:
    dataset = connector.compose(connector.download(cache_dir), context)
    store_dataset(dataset, root)
```

The written child stores no values for the copies. It reads them from the payload sources that its
manifest pins, as [Shared payloads](timef-format.md#shared-payloads) describes.

An import copies every annotation of the parent objects. The child decides which of them it
publishes. For example, a parent can carry an official split as annotations. A child that follows
that split keeps them. A child with a split of its own removes them and adds its own.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
