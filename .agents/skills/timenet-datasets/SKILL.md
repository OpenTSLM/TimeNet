---
name: timenet-datasets
description: Use when finding, searching, inspecting, downloading, or loading TimeNet datasets, via the `timenet` CLI or the Python `TimeNet` client (list, search, info, download, open, load, open_torch). Covers dataset ids, version pinning, the list of sources, and the settings of the local and remote registries and the cache.
---

# Using TimeNet datasets

TimeNet's consumer side is the `timenet` CLI and the `timenet.TimeNet` SDK. Both find, read and
download TimeNet datasets: they read dataset versions from local and remote **registries**
(catalogues of built datasets). A local registry can also serve to store locally built or downloaded
datasets. The CLI is a thin mirror of the SDK, so anything below works the same either way.

Producing datasets (writing connectors, running `timenet-build`) is a separate concern. For that, use
the `add-dataset-connector` skill.

## Dataset ids and versions

- A dataset id is `org/name` (HuggingFace-style, exactly one slash), for example `chengsenwang/tsqa`.
- Pin a version with `<id>@<version>` or `<id>@latest`, e.g. `chengsenwang/tsqa@1.0.0`. You can also
  pass the version as a separate argument. Don't do both (it raises).
- Omitting the version means the newest version of the first source that serves the dataset, so a
  local copy reads offline. `@latest` means the highest version of all sources.

## CLI

The CLI needs the `cli` extra (`timenet[cli]`); `make sync` already installs it. Run it with
`uv run timenet <command>`.

| Command | What it does |
| --- | --- |
| `timenet list` | Table of every dataset of the sources at its highest version, connector cards included (id, name, license, domains). |
| `timenet search [filters]` | Filter the datasets of the sources, connector cards included. See flags below. |
| `timenet info <id>[@version]` | Manifest summary of the release that `open()` reads: version, name, license, domains, specs, task types, and counts (records / annotations / chunks). A dataset that only a connector provides shows its card and the build command. |
| `timenet download <id>[@version]` | Copy a release and its data files into the local registry, and print the release directory to stdout. Never builds. Prints "Nothing to do" if the local registry holds everything. |
| `timenet publish <id>[@version] --dest` `URL` or `PATH`, optionally `--source` `URL` or `PATH` | Copy a complete release and its payload sources from the local registry (or `--source`) to another registry, and print the release to stdout. Same release: nothing happens. Another release under the version: an error. |
| `timenet local info` | Path and size of the cache, then one line each for the downloads and the builds in the local registry (count and size). |
| `timenet local clear [--yes]` | Delete the cache and every release in the local registry. The prompt counts the builds. |
| `timenet local cache info` / `clear [--yes]` | Path and size of the cache (`$TIMENET_CACHE`), or delete its raw source files (`downloads/`) and control databases (`control/`) that carry TimeNet's `CACHEDIR.TAG`. |
| `timenet local downloads info` / `clear [--yes]` | List the downloads in the local registry (dataset, version, source, size, total), or delete them. |
| `timenet local builds info` / `clear [--yes]` | List the builds in the local registry (dataset, version, source, size, total), or delete them. |

A download is a release whose provenance record says `download`. A build is a release whose
record says `build`, or a release without a record. Releases can depend on each other (parent-child
relationships). In that case `downloads clear` and `builds clear` might keep releases that are
depended on, and print `Kept <release>: <release> needs it.` for each. Because `local clear` deletes
everything, no dependency relationships can persist, and everything is removed. A symlinked dataset
or release folder in the local registry stops every clear, `local clear` included, before it
deletes anything. Do not run a clear while a build or a download runs.

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

Global: `-q`/`--quiet` suppresses status output; it goes before the command.

Sources: `list`, `search`, `info` and `download` take three flags after the command. Each flag wins
over its environment variable, and the variable wins over its default. A registry flag that names no
registry (a blank value, an unknown scheme, a URL for `--local-registry`) exits with code 2 and
names the flag. A bad `TIMENET_*` variable exits with code 1 and names the variable.

- `--local-registry PATH`: the local registry, where downloads and builds land and which a lookup
  checks first (`$TIMENET_LOCAL_REGISTRY`, else `~/.timenet/registry`).
- `--remote-registry URL|PATH`: a remote registry, as a URL or a local path, the second source that
  a lookup checks, if the local registry does not have the dataset (`$TIMENET_REMOTE_REGISTRY`,
  else `timenet://`).
- `--auto-build`/`--no-auto-build`: whether installed connectors build a dataset that neither the
  local nor the remote registry has (`$TIMENET_AUTO_BUILD`, else true).

`list` and `search` merge every source, `info` resolves like `open()`, and `download` copies into
the local registry. Of these three flags, `publish` and the `local` commands other than
`local cache` take only `--local-registry`. The `local cache` commands take none of them.

## Python API

```python
import timenet

tn = timenet.TimeNet()               # the default list of sources, see below
tn.list()                            # list[DatasetMetadata]
tn.get_manifest("chengsenwang/tsqa") # Manifest (add a version or use id@version)
tn.search(domain=..., task=AnswerTask, tag="ecg", limit=50)   # filters are scalar-or-list, ANDed
path = tn.download("chengsenwang/tsqa")                   # -> <local registry>/<id>/<version>/ dir
with tn.open("chengsenwang/tsqa") as reader:              # TimeFReader; values stay in storage
    dataset = reader.read()
dataset = tn.load("chengsenwang/tsqa")                    # every value read into memory
```

- `timenet.TimeNet(sources=None)`. `sources` is an ordered list. Each item is a registry object, a
  URL, a `file://` URI, a local path, or `timenet.InstalledConnectors()`. Pass a list even for one
  registry: `timenet.TimeNet(sources=[".timenet-registry"])`. The order of the list is the read
  order: a release that two sources hold comes from the first one, so put the local registry first
  to read offline. An exact version comes from the first source that has it, no version gives the
  newest version of the first source that serves the dataset, and `@latest` is the highest version
  of any source.
- The first `timenet.LocalRegistry(path)` of the list that is not `read_only=True` gets the
  downloads and builds. A path in the list opens such a registry too. A list with
  `InstalledConnectors()` needs a writable local registry, and only such a list builds on a read.
  A remote registry reads only its own files.
- `TimeNet()` uses `timenet.client.default_sources()`: the local registry `$TIMENET_LOCAL_REGISTRY`,
  then `$TIMENET_REMOTE_REGISTRY`, then `InstalledConnectors()` unless `TIMENET_AUTO_BUILD=false`.
  The first entry is always the local registry; `default_local_registry()` returns it alone.
- `list()` and `search()` merge every source of the list, connector cards included: one row per
  dataset at the highest version, sorted by ID. `search()` filters these merged rows and stops at
  `limit` matches. It reads a manifest for `task` or `time_series_spec` only for a row whose card
  matches. A connector never matches these two filters. A registry without a catalog (S3) is left
  out with a `CatalogWarning`, and a list without any catalog raises `TimeNetNoCatalogError`. A card
  that cannot be read is left out with a `ConnectorCardWarning`. To ask one registry alone, use
  `open_registry("timenet://").search(DatasetFilter.of(...))`, with `DatasetFilter` from
  `timenet.types`.
- `get_manifest()` returns the manifest of the release that `open()` reads, without a read of other
  files. A registry that lists a release without serving it gives its listed manifest. A release
  that only a connector provides raises `TimeNetNotBuiltError` (no manifest before a build). The
  error holds the card of the connector as `error.card`.
  `get_card()` returns the card, also from a connector.
- `download()` needs a writable local registry in the list. It resolves like `open()`. The release
  and each payload source that the local registry lacks or holds damaged come from the first
  registry other than the local registry that provides it, so they can come from different
  registries. If the local registry holds a complete copy of the release, only the missing payload
  sources are copied. It never builds: a release that only a connector provides raises
  `TimeNetNotBuiltError`, whose hint names `timenet.build()`. `force=True` copies again, and
  `progress=` takes the events of `timenet.progress`. The hint of a read of a damaged copy names the
  `download()` or the build that replaces it.
- `open(...)` returns a `TimeFReader`; close it or use it in a `with` block. `reader.read()` returns
  a `TimeFDataset` whose values load only when asked: `record.signals[i].to_numpy()` /
  `.to_arrow()`. `reader.get(AnswerTask, task_id)` reads one object by its int ID.
- `load(...)` returns a `TimeFDataset` with every value in memory and its reader closed. Prefer
  `open()` for large datasets. Call `dataset.describe()` for a text summary; iterate
  `dataset.records` and `dataset.tasks`.
- `open_torch(...)` returns a read-only PyTorch `Dataset` that reads one batch at a time; needs the
  `torch` extra (`timenet[torch]`). It holds an open reader: close it or use it in a `with` block.
- `search` task filter takes the task **class** (e.g. `from timenet.types import AnswerTask`), not a string.

## End-to-end recipe

```python
import timenet
from timenet.types import AnswerTask

tn = timenet.TimeNet()
hits = tn.search(task=AnswerTask, limit=10)          # find candidates
tn.get_manifest(hits[0].dataset_id)               # inspect the manifest
with tn.open(hits[0].dataset_id) as reader:       # values stay in storage
    ds = reader.read()
    ds.describe()                                 # identity, counts, per-spec columns, preview
    first = ds.records[0].signals[0].to_numpy()   # reads this one signal's values
```

CLI equivalent: `uv run timenet search --task answer` then `timenet info <id>` then
`timenet download <id>`. See `examples/load_tsqa.py` for a runnable version.

## Configuration

Env vars, prefix `TIMENET_`. Each sets one thing; none holds a list:

- `TIMENET_LOCAL_REGISTRY` the first source that a lookup checks, and the place where
  downloads and builds land (default `~/.timenet/registry`).
- `TIMENET_REMOTE_REGISTRY` a remote registry, as a URL or a local path: the second source that a
  lookup checks, if the local registry does not have the dataset (default `timenet://`).
- `TIMENET_AUTO_BUILD` whether the default list ends with `InstalledConnectors()` (default `true`).
- `TIMENET_CACHE` raw downloads of builds in `downloads/` and control databases in `control/`
  (default `~/.cache/timenet`).
- `TIMENET_TOKEN` bearer token of a remote registry; `TIMENET_ISOLATION` (default `on`).

Precedence for any value: CLI flag, then env var, then default. An empty value is an error. Before
`timenet local cache clear` or `timenet local clear` deletes, and before a build removes its raw
downloads, TimeNet refuses a cache that is the home folder or a folder above it, or that overlaps
the local registry. In the CLI, a registry flag that names no registry exits with code 2, and a bad
variable with code 1. A `sources` list that you pass ignores `TIMENET_LOCAL_REGISTRY`,
`TIMENET_REMOTE_REGISTRY` and `TIMENET_AUTO_BUILD`.

## Further reading

`docs/client.md` (the SDK and consumer CLI) and `docs/registry.md` (registry contract and search
semantics). The SDK lives in `packages/timenet/src/timenet/client.py`; the CLI in
`packages/timenet/src/timenet/cli/app.py`.
