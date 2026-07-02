# Client

`TimeNet` is the single Python entry point for using TimeNet from code. It wraps a
[registry](registry.md) (the catalog) and a local storage path (the download cache). It never runs
connector code. Lives in `timenet.client`.

```python
from timenet.client import TimeNet
from timenet.types import Domain

client = TimeNet()                                   # default public registry
client = TimeNet("~/.timenet/local")                 # a local registry
client = TimeNet("https://registry.timenet.io")      # a remote registry

for meta in client.search(domain=Domain.CARDIOLOGY):
    print(meta.dataset_id)

dataset = client.load("hello_world")                 # download if needed + read
values = dataset.samples[0].time_series[0].to_numpy()
```

## Construction

```python
TimeNet(registry=None, *, storage_path=None)
```

Registry selection order: the `registry` argument, then `$TIMENET_REGISTRY`, then the local default
registry. `registry` accepts a `BaseRegistry`, a URL, a `file://` URI, or a local path.

## Configuration

All local state lives under `~/.cache/timenet/` by default, mirroring HuggingFace's `HF_HOME`
hierarchy. Precedence for any value is **CLI flag / argument > environment variable > default**.

| Env var | Default | What |
| --- | --- | --- |
| `TIMENET_HOME` | `~/.cache/timenet` | Root; setting it relocates everything below. |
| `TIMENET_STORAGE` | `<home>/storage` | Downloaded/loaded datasets (like `HF_DATASETS_CACHE`). |
| `TIMENET_CACHE` | `<home>/cache` | Curation raw sources + Hub downloads (like `HF_HUB_CACHE`). |
| `TIMENET_REGISTRY` | `<home>/registry` | The registry to use (path or URL). |

Configuration is a `pydantic-settings` model (`timenet.config.TimeNetSettings`), so new settings can be
added there.

## Methods

| Method | Description |
| --- | --- |
| `list()` | Every dataset's metadata. |
| `get(dataset_id, version=None)` | A dataset's [manifest](manifest.md). |
| `search(...)` | Filter datasets — mirrors [`registry.search`](registry.md#search). |
| `download(dataset_id, version=None, *, force=False)` | Copy a version's files into local storage; returns the directory. Idempotent unless `force`. |
| `load(dataset_id, version=None)` | `download` if needed, then read into a `TimeFDataset` with lazy per-series values. |

## CLI

The `timenet` console script mirrors the SDK (built with [Typer](https://typer.tiangolo.com)):

```bash
export TIMENET_REGISTRY=https://registry.timenet.io
timenet list
timenet search --query ecg --domain cardiology --limit 10
timenet info hello_world
timenet download hello_world --storage ~/.timenet/storage
```

Registry selection precedence: `--registry` > `$TIMENET_REGISTRY` > default. `search` flags map
one-to-one to the SDK's `search` arguments and are repeatable for list values (`--spec` -> `time_series_spec`,
`--id` -> `dataset_id`).
