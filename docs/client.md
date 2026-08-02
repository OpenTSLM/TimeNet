---
icon: lucide/code
description: "The TimeNet client: browse a registry and load datasets from Python."
tags:
  - guide
  - client
---

# Client

`TimeNet` is the single Python entry point for using TimeNet from code. It wraps a
[registry](registry.md) (the catalog) and a local storage path (the download cache). It never runs
connector code. Lives in `timenet.client`.

```python
from timenet.client import TimeNet
from timenet.types import Domain

client = TimeNet()   # default local registry (~/.cache/timenet/registry)

for meta in client.search(domain=Domain.CARDIOLOGY):
    print(meta.dataset_id)

dataset = client.load("timenet/hello-world")   # download if needed + read
values = dataset.samples[0].time_series[0].to_numpy()
```

## Construction

```python
TimeNet(registry=None, *, storage_path=None)
```

Registry selection order: the `registry` argument, then `$TIMENET_REGISTRY`, then the local default
registry (`<home>/registry`). `registry` accepts a `BaseRegistry`, a local path or `file://` URI, an
`s3://` URI, or a hosted `timenet://` / `http(s)://` URL:

```python
client = TimeNet("./local_registry")   # any directory a build wrote to
```

The `s3://` and remote backends are deferred. Constructing `TimeNet("timenet://")` succeeds, but every
call against it raises `NotImplementedError`, so today only local registries serve data; see
[Registry](registry.md).

## Configuration

All local state lives under `~/.cache/timenet/` by default. Setting the home relocates everything;
the per-area variables override just their own path. Precedence for any value is
**CLI flag / argument > environment variable > default**.

| Env var | Default | What |
| --- | --- | --- |
| `TIMENET_HOME` | `~/.cache/timenet` | Root; setting it relocates everything below. |
| `TIMENET_REGISTRY` | `<home>/registry` | The catalog to browse and pull from (local path or remote URL), and where `timenet-curate build` writes unless `--out` overrides it. A remote value makes `build` fail: there is nowhere local to write. |
| `TIMENET_STORAGE` | `<home>/storage` | Local copies that `download`/`load` fetch from the registry to read. |
| `TIMENET_CACHE` | `<home>/cache` | Raw sources fetched during curation (removed after a successful build). |

Configuration is a `pydantic-settings` model (`timenet.config.TimeNetSettings`), so new settings can be
added there.

## Methods

| Method | Description |
| --- | --- |
| `list()` | Every dataset's metadata. |
| `get(dataset_id, version=None)` | A dataset's [manifest](manifest.md). |
| `search(...)` | Filter datasets, mirrors [`registry.search`](registry.md#search). |
| `download(dataset_id, version=None, *, force=False)` | Copy a version's files into local storage; returns the directory. Idempotent unless `force`. |
| `load(dataset_id, version=None)` | `download` if needed, then read into a `TimeFDataset` with lazy per-series values. |
| `load_torch(dataset_id, version=None)` | `load`, wrapped in a read-only `torch.utils.data.Dataset` (needs the `torch` extra). |

## Versions

Pin a version by suffixing the id with `@<version>`; with no suffix (or `@latest`) you get the latest
committed version. This works everywhere an id is accepted, in both the SDK and the CLI:

```python
client.get("chengsenwang/tsqa@1.0.0")   # pinned
client.load("chengsenwang/tsqa")         # latest (default)
client.load("chengsenwang/tsqa@latest")  # latest, explicit
```

`get` / `download` / `load` / `load_torch` also accept an explicit `version=` argument. Passing both a
`@version` ref and `version=` is an error, and pinning a version that isn't committed raises
`DatasetNotFoundError`. `list` and `search` always report the latest version.

## PyTorch

`load_torch` returns a `TimeFTorchDataset`, a read-only, map-style `torch.utils.data.Dataset`. Each
item is a dict with the sample's `series` as dtype-preserving tensors with shape
`(n_steps, *value_shape)`, plus `sample_id`, `tasks`, and `annotations`.

```python
from timenet.client import TimeNet

# needs: pip install 'timenet[torch-gpu]'
ds = TimeNet().load_torch("chengsenwang/tsqa")
item = ds[0]
series, question = item["series"][0], item["tasks"][0].question
```

To feed a `DataLoader`, select what your model needs (the item's `tasks`/`annotations` are Python
objects, not tensors, and series lengths vary between samples), either with a `transform` on the dataset
or a `collate_fn` on the loader:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=8,
    collate_fn=lambda b: [(x["series"][0], x["tasks"][0].target) for x in b],
)
```

The torch module is imported lazily, so base users who never call `load_torch` don't need torch.

## Command line

Every method here has a shell equivalent. See the [`timenet` CLI](cli/timenet.md).

---

See the [API reference for `timenet.client`](api/client.md) for the full symbol listing.
