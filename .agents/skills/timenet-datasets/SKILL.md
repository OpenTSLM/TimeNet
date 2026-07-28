---
name: timenet-datasets
description: Use when finding, searching, inspecting, downloading, or loading TimeNet datasets, via the `timenet` CLI or the Python `TimeNet` client (list, search, info, download, load, load_torch). Covers dataset ids, version pinning, and registry/storage configuration.
---

# Using TimeNet datasets

TimeNet's consumer side is the `timenet` CLI and the `timenet.client.TimeNet` SDK. Both browse a
**registry** (the catalog of curated datasets) and fetch dataset versions into local storage. The CLI
is a thin mirror of the SDK, so anything below works the same either way.

Producing datasets (writing connectors, running `timenet-curate`) is a separate concern. For that, use
the `add-dataset-connector` skill.

## Dataset ids and versions

- A dataset id is `org/name` (HuggingFace-style, exactly one slash), for example `chengsenwang/tsqa`.
- Pin a version with `<id>@<version>` or `<id>@latest`, e.g. `chengsenwang/tsqa@1.0.0`. You can also
  pass the version as a separate argument. Don't do both (it raises).
- Omitting the version means the latest.

## CLI

The CLI needs the `cli` extra (`timenet[cli]`); `make sync` already installs it. Run it with
`uv run timenet <command>`.

| Command | What it does |
| --- | --- |
| `timenet list` | Table of every dataset in the registry (id, name, license, domains). |
| `timenet search [filters]` | Filter datasets. See flags below. |
| `timenet info <id> [version]` | Manifest summary: version, name, license, domains, specs, task types, and counts (samples / annotations / chunks). |
| `timenet download <id> [version]` | Fetch a version's parquet to local storage; prints the target directory to stdout. |
| `timenet cache info` | Cached datasets with sizes, plus the raw download cache size. |
| `timenet cache clear` | Delete downloads and the raw cache. `--all` also removes the local registry; `-y`/`--yes` skips the prompt. |

`search` flags (repeat a flag to pass several values; all are ANDed across kinds):

- `-q`/`--query` free-text over name/description/tags
- `--domain` (must be a valid `Domain`)
- `--task` (`classification`, `answer`, `scalar_prediction`, `temporal_localization`, `forecasting`,
  `ts_editing`, `ts_generation`, `ts_correspondence`)
- `--license` (must be a valid `License`)
- `--spec` a `time_series_spec` type
- `--id` restrict to specific dataset ids
- `--tag`
- `--limit` (default 100)

Global: `-q`/`--quiet` suppresses status output. `-r`/`--registry` picks the registry (else
`$TIMENET_REGISTRY`, else the local default). `download` takes `--storage` (else `$TIMENET_STORAGE`).

## Python API

```python
from timenet.client import TimeNet

tn = TimeNet()                       # registry: arg > $TIMENET_REGISTRY > local default
tn.list()                            # list[DatasetMetadata]
tn.get("chengsenwang/tsqa")          # Manifest (add a version or use id@version)
tn.search(domain=..., task=AnswerTask, tag="ecg", limit=50)   # filters are scalar-or-list, ANDed
path = tn.download("chengsenwang/tsqa")                   # -> local <storage>/<id>/<version>/ dir
dataset = tn.load("chengsenwang/tsqa")                    # download-if-needed, then read into memory
```

- `TimeNet(registry=None, *, storage_path=None)`. `registry` accepts a `BaseRegistry`, a URL, a
  `file://` URI, or a local path.
- `load(...)` returns a `TimeFDataset` with lazy per-series loaders. Call `dataset.describe()` for a
  text summary; iterate `dataset.samples` and `dataset.tasks`; get values with
  `sample.time_series[i].to_numpy()` / `.to_arrow()` (they load only when asked).
- `load_torch(...)` returns a read-only PyTorch `Dataset`; needs the `torch` extra (`timenet[torch]`).
- `search` task filter takes the task **class** (e.g. `from timenet.types import AnswerTask`), not a string.

## End-to-end recipe

```python
from timenet.client import TimeNet
from timenet.types import AnswerTask

tn = TimeNet()
hits = tn.search(task=AnswerTask, limit=10)          # find candidates
tn.get(hits[0].dataset_id)                        # inspect the manifest
ds = tn.load(hits[0].dataset_id)                  # load into memory
ds.describe()                                     # identity, counts, per-spec columns, sample preview
first = ds.samples[0].time_series[0].to_numpy()   # pull raw values lazily
```

CLI equivalent: `uv run timenet search --task answer` then `timenet info <id>` then
`timenet download <id>`. See `examples/load_tsqa.py` for a runnable version.

## Configuration

All local state lives under `TIMENET_HOME` (default `~/.cache/timenet`). Env vars, prefix `TIMENET_`:

- `TIMENET_HOME` relocates everything.
- `TIMENET_REGISTRY` selects the registry (URL or path).
- `TIMENET_STORAGE` where downloads are cached (default `<home>/storage`).
- `TIMENET_CACHE` raw source / Hub download cache (default `<home>/cache`).

Precedence for any value: explicit argument or CLI flag, then env var, then default.

## Further reading

`docs/client.md` (the SDK and consumer CLI) and `docs/registry.md` (registry contract and search
semantics). The SDK lives in `packages/timenet/src/timenet/client.py`; the CLI in
`packages/timenet/src/timenet/cli/app.py`.
