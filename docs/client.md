---
icon: lucide/code
description: "The TimeNet client: browse a registry and load datasets from Python."
tags:
  - guide
  - client
---

# Client

`TimeNet` is the single Python entry point to TimeNet. It wraps a [registry](registry.md) (the
catalog) and a local storage path (the download cache). Against a local registry, `load` builds a
dataset that the registry does not have. This needs an installed package that registers a connector
for the dataset id. See [Curate & publish](curation.md). Against a remote registry, `load` never
runs connector code. The module `timenet.client` contains `TimeNet`.

```python
from timenet.client import TimeNet
from timenet.types import Domain

client = TimeNet()   # default local registry (~/.cache/timenet/registry)

for meta in client.search(domain=Domain.CARDIOLOGY):
    print(meta.dataset_id)

# read in place through the registry, lazy values
dataset = client.load("timenet/hello-world")
values = dataset.samples[0].time_series[0].to_numpy()
```

## Construction

```python
TimeNet(registry=None, *, storage_path=None)
```

TimeNet selects the registry in this order: the `registry` argument, then the environment variable
`$TIMENET_REGISTRY`, then the local default registry (`<home>/registry`). The `registry` argument
can accept a `BaseRegistry` object, a local path, a `file://` URI, an `s3://` URI, or a hosted
`timenet://` or `http(s)://` URL:

```python
client = TimeNet("./local_registry")   # any directory a build wrote to
```

A `timenet://` or `http(s)://` URL opens a `RemoteRegistry`, which serves reads
(`list`/`search`/`get`/`download`/`load`). Only remote `store` and the `s3://` backend still raise
`NotImplementedError`; see [Registry](registry.md).

## Configuration

By default, TimeNet stores all local state under `~/.cache/timenet/`. If you set the home, TimeNet
relocates everything below it. Each per-area variable can override only its own path. The
precedence for any value is **CLI flag / argument > environment variable > default**.

| Env var | Default | What |
| --- | --- | --- |
| `TIMENET_HOME` | `~/.cache/timenet` | Root; setting it relocates everything below. |
| `TIMENET_REGISTRY` | `<home>/registry` | The catalog to browse and pull from (local path or remote URL), and where `timenet-curate build` writes unless `--out` overrides it. A remote value makes `build` fail: there is nowhere local to write. |
| `TIMENET_STORAGE` | `<home>/storage` | Local copies that `download` fetches from the registry as an explicit disk cache. |
| `TIMENET_CACHE` | `<home>/cache` | Raw sources fetched during curation (removed after a successful build). |
| `TIMENET_TOKEN` | _(unset)_ | Bearer token for a remote registry; unset reads anonymously (enough for public data). |
| `TIMENET_DOWNLOAD_MODE` | `on_demand` | How a remote `load` fetches bytes: `on_demand` (lazy range reads, cache-first) or `full` (download the whole version first). |
| `TIMENET_ISOLATION` | `on` | Whether a curation build runs in an environment built from the connector's requirements. `off` runs it in the current interpreter. |

The configuration is a `pydantic-settings` model, `timenet.config.TimeNetSettings`. You can add new
settings there.

## Methods

| Method | Description |
| --- | --- |
| `list()` | Returns the metadata for every dataset. |
| `get(dataset_id, version=None)` | Returns a dataset's [manifest](manifest.md). |
| `search(...)` | Filters datasets. This mirrors [`registry.search`](registry.md#search). |
| `download(dataset_id, version=None, *, force=False)` | Copies a version's files into local storage as an explicit disk cache, and returns the directory. This method is idempotent unless you set `force`. |
| `load(dataset_id, version=None, *, download=None)` | Reads a `TimeFDataset` with lazy per-series values, in place, through the registry's `open_version` handle. This does not download the whole dataset. `download` (`"full"` / `"on_demand"`) overrides the fetch mode for a remote registry; it is ignored for local. |
| `load_torch(dataset_id, version=None)` | Wraps `load` in a read-only `torch.utils.data.Dataset`. This needs the `torch` extra. |

## Remote loading

Against a remote registry, `load` fetches bytes one of two ways, set by `TIMENET_DOWNLOAD_MODE`
(default `on_demand`) or the per-call `download=` argument:

- `on_demand`: the reader range-reads Parquet footers and value slices straight from presigned URLs,
  pulling only the bytes a query touches. It is cache-first, reusing any complete files a prior `full`
  load or `download()` left in `TIMENET_STORAGE`.
- `full`: download the whole version into `TIMENET_STORAGE` first (in parallel, committed atomically),
  then read it locally. This is what `download()` does.

```python
client = TimeNet("timenet://")
# Uses $TIMENET_DOWNLOAD_MODE (on_demand by default).
client.load("chengsenwang/tsqa")
# Force a full download, then read locally.
client.load("chengsenwang/tsqa", download="full")
```

A Zarr-backed version always takes the `full` path: its store driver can't range-read presigned URLs.
Both modes are no-ops for a local registry, which already reads in place, and `download=` is ignored
there.

## Versions

You can pin a version by adding a suffix `@<version>` to the id. Without a suffix, or with
`@latest`, you get the latest committed version. This works everywhere that TimeNet accepts an id,
in the SDK and in the CLI:

```python
client.get("chengsenwang/tsqa@1.0.0")   # pinned
client.load("chengsenwang/tsqa")         # latest (default)
client.load("chengsenwang/tsqa@latest")  # latest, explicit
```

The methods `get`, `download`, `load`, and `load_torch` also accept an explicit `version=`
argument. If you pass both a `@version` reference and `version=`, TimeNet raises an error. If you
pin a version that is not committed, TimeNet raises `DatasetNotFoundError`. The methods `list` and
`search` always report the latest version.

## PyTorch

`load_torch` returns a `TimeFTorchDataset`. This is a read-only, map-style
`torch.utils.data.Dataset`. Each item is a dict. The dict contains the sample's `series` as
dtype-preserving tensors with shape `(n_steps, *value_shape)`, plus `sample_id`, `tasks`, and
`annotations`.

```python
from timenet.client import TimeNet

# needs: pip install 'timenet[torch]'
ds = TimeNet().load_torch("chengsenwang/tsqa")
item = ds[0]
series, question = item["series"][0], item["tasks"][0].question
```

To feed a `DataLoader`, select the fields that your model needs. Use a `transform` on the dataset,
or a `collate_fn` on the loader, for this selection. The item's `tasks` and `annotations` are
Python objects, not tensors. Series lengths also vary between samples.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=8,
    collate_fn=lambda b: [(x["series"][0], x["tasks"][0].target) for x in b],
)
```

TimeNet imports the torch module only when needed. If you never call `load_torch`, you do not need
torch installed.

## Command line

Every method in this page has an equivalent shell command. See the
[`timenet` CLI](cli/timenet.md).

---

See the [API reference for `timenet.client`](api/client.md) for the full symbol listing.
