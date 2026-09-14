---
icon: lucide/code
description: "The TimeNet client: browse a registry and open dataset versions from Python."
tags:
  - guide
  - client
---

# Client

`TimeNet` is the single Python entry point to TimeNet. It wraps a [registry](registry.md) (the
catalog) and a local storage path (the download cache). It lives in `timenet.client`.

```python
from timenet.client import TimeNet
from timenet.types import Domain

client = TimeNet()  # the hosted registry (timenet://)

for meta in client.search(domain=Domain.CARDIOLOGY):
    print(meta.dataset_id)

with client.open("timenet/hello-world") as reader:
    record = next(reader.iter_records(batch_size=64))
    values = reader.values(record.signals()[0].signal_id)
```

## Construction

```python
TimeNet(registry=None, *, storage_path=None)
```

TimeNet selects the registry in this order: the `registry` argument, then the environment variable
`$TIMENET_REGISTRY`, then the hosted TimeNet registry (`timenet://`). The `registry` argument
accepts a `BaseRegistry` object, a local path, a `file://` URI, an `s3://` URI, or a hosted
`timenet://` or `http(s)://` URL:

```python
client = TimeNet()                    # the hosted registry (timenet://)
client = TimeNet("./local_registry")  # any directory a writer published to
```

## Configuration

By default, TimeNet stores all local state under `~/.cache/timenet/`. If you set the home, TimeNet
relocates everything below it. Each per-area variable can override only its own path. The
precedence for any value is **explicit argument > environment variable > default**.

| Env var | Default | What |
| --- | --- | --- |
| `TIMENET_HOME` | `~/.cache/timenet` | Root; setting it relocates everything below. |
| `TIMENET_REGISTRY` | `<home>/registry` | The catalog to browse and pull from, as a local path or a remote URL. |
| `TIMENET_STORAGE` | `<home>/storage` | Local copies that `download` fetches from the registry, as an explicit disk cache. |
| `TIMENET_CACHE` | `<home>/cache` | Raw sources fetched while building a dataset. |
| `TIMENET_TOKEN` | _(unset)_ | Bearer token for a remote registry; unset reads anonymously, which is enough for public data. |

The configuration is a `pydantic-settings` model, `timenet.config.TimeNetSettings`. Add new settings
there.

## Methods

| Method | Description |
| --- | --- |
| `list()` | Returns the metadata of every dataset in the registry. |
| `get(dataset_id, version=None)` | Returns a dataset's [manifest](manifest.md). |
| `search(...)` | Filters datasets. This mirrors [`registry.search`](registry.md#search). |
| `download(dataset_id, version=None, *, force=False, progress_cb=None)` | Copies a version's files into local storage and returns the directory. It is idempotent unless you set `force`. |
| `open(dataset_id, version=None)` | Returns a [`TimeFReader`](timef-reader.md) over the version. Close it, or use it as a context manager. |

## Opening a version

A local or S3 registry reads in place. A remote registry materializes the version into
`$TIMENET_STORAGE` first, so a remote `open` fetches the whole version before it returns.

```python
client = TimeNet("timenet://")
with client.open("chengsenwang/tsqa") as reader:
    print(reader.counts())
```

From there, everything happens on the reader: hydrate records or tasks in batches, query annotations
in SQL, and pull values by signal id. See [TimeFReader](timef-reader.md) for the full surface, and
[Load into your stack](usage.md) for the pandas and PyTorch views.

## Restricted datasets

A credentialed or restricted dataset cannot be redistributed, so a hosted registry never serves its
bytes; `download` and `open` raise `TimeNetAccessError` and point at the dataset's access URL. Build
such a dataset yourself and read it from a local registry, where TimeNet serves only what you put
there.

## Versions

Pin a version by adding an `@<version>` suffix to the id. Without a suffix, or with `@latest`, you
get the latest committed version. This works everywhere TimeNet accepts an id:

```python
client.get("chengsenwang/tsqa@1.0.0")   # pinned
client.open("chengsenwang/tsqa")        # latest (default)
client.open("chengsenwang/tsqa@latest") # latest, explicit
```

`get`, `download`, and `open` also accept an explicit `version=` argument. Passing both a
`@version` suffix and `version=` raises `TimeFValidationError`. Pinning a version that is not
committed raises `TimeNetDatasetNotFoundError`. `list` and `search` always report the latest version.

---

See the [API reference for `timenet.client`](api/client.md) for the full symbol listing.
