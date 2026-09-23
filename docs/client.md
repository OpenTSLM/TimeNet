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

## Load or download

`load()` returns a `TimeFDataset`. The hierarchy is in memory, but each Signal keeps a lazy values
loader.

```python
dataset = client.load("timenet/hello-world")
signal = dataset.records[0].signals[0]
window = signal.read_steps(0, 100)
```

For a local registry, the reader opens the version in place. If the version is missing, `load()` can
build it from an installed connector.

```python
dataset = TimeNet(".timenet-registry").load(
    "timenet/hello-world",
    auto_build=True,
)
```

Set `auto_build=False` to report a missing local version without starting a build. Remote registries
never run connector code.

For an HTTP registry, `load()` downloads the complete version into `TIMENET_STORAGE` first. A later
load reuses that committed copy. An S3 registry reads in place unless the storage directory already
contains a complete copy.

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

The `get`, `download`, `load`, and `load_torch` methods also accept `version=`. Do not supply a
version in both places.

## Load for PyTorch

`load_torch()` returns a read-only `torch.utils.data.Dataset`. Install the `torch` extra before you
use it.

```python
from timenet.client import TimeNet

dataset = TimeNet(".timenet-registry").load_torch("timenet/hello-world")
item = dataset[0]

signal = item["series"][0]
mask = item["series_masks"][0]
task = item["tasks"][0]
print(task.prompt, task.targets)
```

Each item also contains `record_id` and record-level `annotations`. Signal lengths and task objects
can vary between records. Use a `transform` or `collate_fn` that matches your model.

TimeNet imports PyTorch only when this adapter is used.

## Command line

The [`timenet` CLI](cli/timenet.md) provides the same discovery and download operations for shell
workflows.

---

See the [API reference for `timenet.client`](api/client.md) for every parameter and return type.
