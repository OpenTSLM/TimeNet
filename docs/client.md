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

For a local registry, the reader opens the version in place. If the version is missing, `open()` and
`load()` can build it from an installed connector.

```python
dataset = TimeNet(".timenet-registry").load(
    "timenet/hello-world",
    auto_build=True,
)
```

Set `auto_build=False` to report a missing local version without starting a build. Remote registries
never run connector code.

An HTTP or S3 registry reads from a complete copy of the version in `TIMENET_STORAGE` if one exists,
and otherwise reads the remote files by range, as each read needs them.

`download()` always copies the complete version into local storage:

```python
path = client.download("chengsenwang/tsqa@1.0.0")
```

Use `force=True` to replace an existing local copy.

## Pin a version

Append `@<version>` to a dataset ID:

```python
client.get("chengsenwang/tsqa@1.0.0")
client.load("chengsenwang/tsqa@latest")
```

The `get`, `download`, `open`, `load`, and `load_torch` methods also accept `version=`. Do not
supply a version in both places.

## Load for PyTorch

`load_torch()` returns a read-only `torch.utils.data.Dataset`. Install the `torch` extra before you
use it. It reads through `open()`, one batch of records and their tasks at a time, so the dataset
does not have to fit in memory.

```python
from timenet.client import TimeNet
from timenet.types import read_value, read_values

dataset = TimeNet(".timenet-registry").load_torch("timenet/hello-world")
item = dataset[0]

signal = item["series"][0]
mask = item["series_masks"][0]
task = item["tasks"][0]
# read_value reads a text handle and passes other values, None included,
# through unchanged; read_values does the same for each target.
print(read_value(task.prompt), read_values(task.targets))
```

Each item also contains `record_id` and record-level `annotations`. Signal lengths and task objects
can vary between records. Use a `transform` or `collate_fn` that matches your model.

TimeNet imports PyTorch only when this adapter is used.

## Command line

The [`timenet` CLI](cli/timenet.md) provides the same discovery and download operations for shell
workflows.

---

See the [API reference for `timenet.client`](api/client.md) for every parameter and return type.
