---
icon: lucide/code
description: "Browse a registry and load TimeF datasets from Python."
tags:
  - guide
  - client
---

# Client

`TimeNet` is the main Python entry point for dataset consumers. It wraps one
[registry](registry.md) and one local storage directory.

```python
from timenet.client import TimeNet
from timenet.types import Domain

client = TimeNet(".timenet-registry")

for metadata in client.search(domain=Domain.CARDIOLOGY):
    print(metadata.dataset_id)

dataset = client.load("timenet/hello-world")
values = dataset.records[0].signals[0].to_numpy()
```

## Choose a registry

Create a client with a registry object, path, or URL:

```python
from timenet.client import TimeNet

hosted = TimeNet()
local = TimeNet(".timenet-registry")
s3 = TimeNet("s3://my-bucket/timenet")
private = TimeNet("https://registry.example.com")
```

`TimeNet()` selects a registry in this order:

1. The `registry` argument.
2. The `TIMENET_REGISTRY` environment variable.
3. The hosted registry, `timenet://`.

The [getting-started example](get-started.md) uses an explicit local path so its build and load use
the same registry.

## Local state

TimeNet stores local state under `~/.cache/timenet` by default.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TIMENET_HOME` | `~/.cache/timenet` | Parent directory for local state. |
| `TIMENET_REGISTRY` | unset | Registry URL or local path. |
| `TIMENET_STORAGE` | `<home>/storage` | Complete versions downloaded from a registry. |
| `TIMENET_CACHE` | `<home>/cache` | Raw source files used during builds. |
| `TIMENET_TOKEN` | unset | Bearer token for a remote registry. |
| `TIMENET_ISOLATION` | `on` | Whether connector builds use isolated environments. |

An explicit argument takes precedence over an environment variable. An environment variable takes
precedence over its default.

The model for these values is `timenet.config.TimeNetSettings`.

## Browse datasets

The client provides these discovery methods:

| Method | Result |
| --- | --- |
| `list()` | Metadata for the latest version of every dataset. |
| `get(dataset_id, version=None)` | One dataset manifest. |
| `search(...)` | Metadata that matches registry filters. |

`search()` accepts filters for free text, domains, task classes, licenses, Signal specifications,
dataset IDs, and tags. See [Registry search](registry.md#search) for their matching rules.

## Open, load, or download

`open()` returns a [`TimeFReader`](timef-reader.md) for one version. Opening reads the manifest,
not the catalog or any value. The reader's reads build Records and tasks whose stored values stay
in storage until something reads them, unless a
[load plan](timef-reader.md#loading-values-at-read-time) loads them. Close the reader after use, or
use it as a context manager:

```python
with client.open("timenet/hello-world") as reader:
    dataset = reader.read()
    signal = dataset.records[0].signals[0]
    # Reads only the chunks that cover steps 0 to 100.
    window = signal.read_steps(0, 100)
```

`load()` opens the version, reads it with every value loaded, and closes the reader. The dataset
needs no storage afterwards:

```python
from timenet.reader import LoadPlan

dataset = client.load("timenet/hello-world")

# The same, spelled out:
with client.open("timenet/hello-world") as reader:
    dataset = reader.read(load=LoadPlan.ALL_EAGER)
```

`load()` keeps every Signal array, time offset, and text value in memory. For a large dataset, use
`open()` and read what you need, such as one task with `reader.get()` or the tasks' text alone with
a load plan.

To read every value of a large dataset without holding all of it in memory, stream its Records in
batches. `iter_records()` loads the values of each batch before it yields the batch. The stream
holds one batch at a time:

```python
with client.open("timenet/hello-world") as reader:
    for record in reader.iter_records(load=LoadPlan.ALL_EAGER, batch_size=500):
        values = record.signals[0].to_numpy()
```

For a local registry, the reader opens the version in place. A local read needs complete copies of
the version and of its payload sources: the releases whose payload files the version reads. For
each of these releases, `open()` and `load()` use the first of these steps that works:

1. Reuse a complete local copy.
2. Copy the release from the registry that `fetch_from` names.
3. Build the release from its installed connector, if `auto_build` is set. This is the default.

If no step works, the call raises an error that names the release and the keyword that can get it.
A build of the version prepares its declared parents in the same order.

```python
with TimeNet(".timenet-registry").open(
    "org/name",
    fetch_from="timenet://",
) as reader:
    records = reader.read().records
```

`fetch_from` takes a registry or its URI. `open_torch()` takes `auto_build` and `fetch_from` too.
Set `auto_build=False` to report a missing local release without starting a build.

A payload source builds with its own connector, at the exact version that the version pins. The
build publishes the source only if it writes the pinned files. See
[Rebuild a pinned release](build.md#rebuild-a-pinned-release).

A local copy is damaged if its manifest is present but a file is missing or has another size.
`fetch_from` repairs a damaged copy. Without `fetch_from`, the call raises `TimeNetFileError`, which
names the first missing or resized file. No build runs over a damaged copy.

Remote registries never run connector code. A remote client rejects `fetch_from`, because it reads
from its own registry.

An HTTP or S3 registry resolves the manifests of the payload sources when `open()`, `load()`, or
`open_torch()` runs. Reads then use complete copies in `TIMENET_STORAGE` where they exist. Otherwise
they read the remote files by range, as each read needs them. The first read of Records or tasks
downloads the catalog of the version and makes sure that it matches the manifest. A read never opens
the catalog of a parent release. A read also never fetches a parent release that is not a payload
source. See [Payload sources](registry.md#payload-sources).

A child dataset gives the objects that it imports from a parent IDs of its own. To read such a copy
by the ID that the parent gives it, pass the parent to `get()` with `parent=`:

```python
from timenet.dataset import Signal
from timenet.types.reference_keys import ParentAlias

with client.open("physionet/ecg-qa-cot") as reader:
    signal = reader.get(Signal, parent_signal_id, parent=ParentAlias("ptbxl"))
    window = signal.read_steps(0, 100)
```

`parent_signal_id` is the int ID that the parent gives the Signal. `parent=` takes the alias of the
parent as a `ParentAlias`, or its exact release as a `DatasetRef`. Without `parent=`, the lookup
uses the ID that the child gives the object.

`download()` copies the complete release and its payload sources into local storage:

```python
path = client.download("chengsenwang/tsqa@1.0.0")
```

Use `force=True` to replace an existing local copy. A later `open()` or `load()` reuses the complete
copies. A remote client still requests the manifests. To read without a network, open the storage
directory as a local registry with `TimeNet(storage_dir)`, and set `auto_build=False`. A remote read
keeps the byte ranges that it reads in memory only. The copies of `download()` stay on disk.

## Pin a version

Append `@<version>` to a dataset ID:

```python
client.get("chengsenwang/tsqa@1.0.0")
client.load("chengsenwang/tsqa@latest")
```

The `get`, `download`, `open`, `load`, and `open_torch` methods also accept `version=`. Do not
supply a version in both places.

## Open for PyTorch

`open_torch()` returns a read-only `torch.utils.data.Dataset`. Install the `torch` extra before you
use it. It reads through `open()`, one batch of records and their tasks at a time, so the dataset
does not have to fit in memory. The view holds an open reader. Close the view with `close()`, or
use it in a `with` block.

```python
from timenet.client import TimeNet
from timenet.types import read_value, read_values

client = TimeNet(".timenet-registry")
with client.open_torch("timenet/hello-world") as view:
    item = view[0]

    signal = item["series"][0]
    mask = item["series_masks"][0]
    task = item["tasks"][0]
    # read_value reads a text handle and passes other values, None included,
    # through unchanged; read_values does the same for each target.
    print(read_value(task.prompt), read_values(task.targets))
```

Each item also contains the int `record_id` and record-level `annotations`. Signal lengths and
task objects can vary between records. Use a `transform` or `collate_fn` that matches your model.

TimeNet imports PyTorch only when this adapter is used.

## Command line

The [`timenet` CLI](cli/timenet.md) provides the same discovery and download operations for shell
workflows.

---

See the [API reference for `timenet.client`](api/client.md) for every parameter and return type.
