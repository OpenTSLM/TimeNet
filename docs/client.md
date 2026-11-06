---
icon: lucide/code
description: "Find, read and download TimeNet datasets from Python."
tags:
  - guide
  - client
---

# Client

`TimeNet` is the main Python entry point for dataset consumers. It reads releases through an
ordered list of sources: [registries](registry.md) and the installed connectors.

```python
import timenet
from timenet.types import Domain

client = timenet.TimeNet(sources=[".timenet-registry"])

for metadata in client.search(domain=Domain.CARDIOLOGY):
    print(metadata.dataset_id)

dataset = client.load("timenet/hello-world")
values = dataset.records[0].signals[0].to_numpy()
```

## Choose the sources

A client takes an ordered list of sources. A source is a local or remote registry, or
`InstalledConnectors()`:

```python
import timenet

client = timenet.TimeNet(sources=[
    timenet.LocalRegistry(".timenet-registry", store=True),
    "https://registry.example.com",
    "timenet://",
    timenet.InstalledConnectors(),
])
```

- A string is a registry: a URL (`timenet://`, `https://...`, `s3://...`), a `file://` URI, or a
  local path. A `Path` is a local path. `sources` takes a list, even for one source.
- `InstalledConnectors()` stands for every installed package that registers a builder backend,
  such as `timenet-connectors`. If only this source provides a release, a read builds it. Without
  it in the list, a client never runs connector code.
- The store is the one `LocalRegistry` with `store=True`, where builds land. A client accepts at
  most one store, and a list with `InstalledConnectors()` needs one. A `LocalRegistry` without the
  flag is only read, for example a mirror on a network drive.
- A registry URL in the list gets the store of the list. If the store holds a complete copy of the
  same release, with the same manifest apart from the build environment, the registry reads that
  copy instead of its remote files. Without a store in the list, it reads only its own files.

An exact version comes from the first source that provides it, and "latest" is the highest
version that any source provides. Only "not found" and "not served" fall through to the next
source, and any other error stops the call. So if a registry in the list does not answer, a request
for "latest" fails. If the store holds a version of the dataset, the hint of that error names it.

A client that only reads takes a list without a store:

```python
import timenet

hosted = timenet.TimeNet(sources=["timenet://"])
local = timenet.TimeNet(sources=[".timenet-registry"])
s3 = timenet.TimeNet(sources=["s3://my-bucket/timenet"])
```

`TimeNet()` uses the default list of `timenet.client.default_sources()`. If no variable is set,
the list is:

```python
[
    LocalRegistry("~/.timenet/registry", store=True),
    "timenet://",
    InstalledConnectors(),
]
```

`TIMENET_LOCAL_REGISTRY` moves the store, `TIMENET_REMOTE_REGISTRY` replaces `timenet://`, and
`TIMENET_AUTO_BUILD=false` removes `InstalledConnectors()` (see [Settings](#settings)). The
arguments `local_registry`, `remote_registry` and `auto_build` of `default_sources()` override
them, and `timenet.client.default_store()` returns the store alone. `default_sources()` gives the
remote registry its store. A list that you pass as `sources` ignores these three variables.

The [getting-started example](get-started.md) builds a dataset into the default store and reads it
with `TimeNet()`.

## Settings

These environment variables configure TimeNet. Each variable sets one thing.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TIMENET_LOCAL_REGISTRY` | `~/.timenet/registry` | The store: a local registry, as a local path or a `file://` URI. It is the first source that a lookup checks, and the place where downloads and builds land. |
| `TIMENET_REMOTE_REGISTRY` | `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the local registry does not have the dataset. |
| `TIMENET_AUTO_BUILD` | `true` | Whether installed connectors build a dataset that neither the local nor the remote registry has. If `true`, the default list ends with `InstalledConnectors()`. |
| `TIMENET_CACHE` | `~/.cache/timenet` | Raw downloads of builds in `downloads/`, and control databases of remote reads in `control/`. |
| `TIMENET_TOKEN` | unset | Bearer token for a remote registry. |
| `TIMENET_ISOLATION` | `on` | Whether connector builds use isolated environments. |

The store lies outside `~/.cache`, because some tools delete files there without asking. The cache
holds only files that a build or a read can make again.

An explicit argument, for example a CLI flag, wins over its variable, and the variable wins over its
default. An empty value is an error, and so are a cache directory that overlaps the store and a
cache directory that is your home folder or a folder above it.

When TimeNet creates the `downloads` or the `control` folder of the cache, it writes a
[`CACHEDIR.TAG`](https://bford.info/cachedir/) file into it, as the Cache Directory Tagging
Specification describes. `timenet local cache clear` and `timenet local clear` delete only a cache
folder with this file. So a `TIMENET_CACHE` that points at a wrong folder is never wiped. Backup
tools that follow the specification skip the cache.

`timenet.config.settings()` reads and checks every variable each time that it runs, also a variable
that an argument overrides. The model for these values is `timenet.config.TimeNetSettings`.

## Browse datasets

`list()` and `search()` answer for the whole list of sources. They return one row per dataset, at
the highest version that a source lists, sorted by dataset ID. The card of each installed connector
counts as a row. So `list()` also shows a dataset that only a connector provides.

| Method | Result |
| --- | --- |
| `list()` | Metadata for the highest version of every dataset across the sources. |
| `search(...)` | Metadata that matches registry filters, across the sources. |

`search()` accepts filters for free text, domains, task classes, licenses, Signal specifications,
dataset IDs, and tags. See [Registry search](registry.md#search) for their matching rules. Each
registry applies the filters, and the card of each installed connector passes the same filters. A
connector never matches `task` or `time_series_spec`, because both filters read the schema of a
manifest. `limit` applies after the merge.

The calls ask the sources one after another, in order. If two sources list a dataset at the same
version, the row of the earlier source stays. A registry without a catalog, such as an S3 registry,
adds nothing. If the card of a connector cannot be read, the call leaves the dataset out with a
`ConnectorCardWarning`. An error of a source, for example a registry that cannot be reached, stops
the call.

To query one registry alone, open it and ask it:

```python
from timenet.registry import open_registry

hits = open_registry("timenet://").search(query="ecg")
```

## Inspect a release

These methods describe the release that `open()` reads. They resolve the release through the
sources as `open()` does (see [Choose the sources](#choose-the-sources)), and copy and build
nothing.

| Method | Result |
| --- | --- |
| `get_manifest(dataset_id, version=None)` | The [manifest](manifest.md) of the release. |
| `get_card(dataset_id, version=None)` | The card of the release. |
| `plan_read(dataset_id, version=None)` | The plan of the read, one step for each release. |

`get_manifest()` reads only the manifest, and checks no other file. A registry can list a
release without serving its files, for example a credentialed release on the hosted registry. Then
`get_manifest()` returns the listed manifest, which shows the access level and the access page of
each card. A release gets its manifest when it is built. So for a release that only
`InstalledConnectors()` provides, `get_manifest()` raises `TimeNetNotBuiltError`. Its hint names
`get_card()` and [`timenet.build()`](build.md#programmatic-builds), with `store=` if the store of
the client is not the default one.

`get_card()` returns the card of the release. The card comes from the manifest, or from the
installed connector for a release that is not built yet:

```python
import timenet
from timenet.errors import TimeNetNotBuiltError

client = timenet.TimeNet()
try:
    card = client.get_manifest("chengsenwang/tsqa").metadata
except TimeNetNotBuiltError:
    card = client.get_card("chengsenwang/tsqa")
print(card.name, card.license)
```

`plan_read()` returns the plan that `open()` runs, as a `timenet.Plan`. It copies and builds
nothing and returns no reader. Like `open()`, it checks the file sizes of a release that it reads
from a local registry.

Each step of the plan names a release, an action, and the source that provides the release
(`entry`). A `"read"` step opens the release where it is. A `"copy"` or `"build"` step puts the
release into the store first:

```python
for step in client.plan_read("chengsenwang/tsqa").steps:
    print(step.release, step.action, step.entry)
```

`plan_read()` checks the plan as `open()` does. If a registry lists the release but does not serve
its files, `plan_read()` raises the `TimeNetAccessError` of `open()`, which names the access level
and the access page.

`list()` can show a version that a registry lists but does not serve. This version can be higher
than the version that `open()` reads, because a version that is not served does not count for
"latest".

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

`open()`, `load()`, and `open_torch()` read each release where it is, and copy nothing. They ask
the sources by the rules of [Choose the sources](#choose-the-sources), so if an installed connector
declares a newer version than the store holds, the call builds the newer version. Each payload
source, a release whose payload files the version reads, resolves in the same way at its pinned
version. So a version and its payload sources can come from different registries.

If no source provides a release, the call raises `TimeNetDatasetNotFoundError`, which names each
source. Its hint names a registry to add. If an installed connector builds the release but the list
has no `InstalledConnectors()`, the hint names that source too, and a store if the client has none.
If no connector package is installed, it names one, such as `timenet-connectors`. If the installed
connector declares another version than the one requested, the call raises
`TimeNetVersionUnavailableError` before the build starts.

```python
import timenet

client = timenet.TimeNet(sources=[
    timenet.LocalRegistry(".timenet-registry", store=True),
    "timenet://",
])
with client.open("org/name") as reader:
    records = reader.read().records
```

This list has no `InstalledConnectors()`, so a missing release raises an error instead of starting
a build.

If only `InstalledConnectors()` provides a release, the call builds it into the store first and
reads it there. The build copies or builds each missing parent from the same sources, as
[Prepare parent releases](build.md#prepare-parent-releases) describes. A payload source builds at
the exact release that the version pins, and the build publishes it only if it writes the pinned
files. See [Rebuild a pinned release](build.md#rebuild-a-pinned-release). To build a release
without a read, call [`timenet.build()`](build.md#programmatic-builds).

A local copy is damaged if its manifest is present but a file is missing or has another size.
Before the call reads or builds anything, it checks each release that it opens in a local registry,
and raises `TimeNetFileError` for a damaged copy. The call never reads past a damaged copy to a
later source, and never builds over it. For a damaged copy in the store, the hint of the error
names one way to replace it that works: `download()` from a registry that provides the release, a
forced `timenet.build()`, or a registry with an intact copy to add.

When `open()`, `load()`, or `open_torch()` reads from an HTTP or S3 registry, it reads the
complete copy of the same release in the store, if the store holds one. Otherwise it reads the
remote files by range. The first read of Records or tasks downloads the catalog of the version and
makes sure that it matches the manifest. A read never fetches a parent release that is not a
payload source. See [Payload sources](registry.md#payload-sources).

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

`download()` copies the complete release and its payload sources into the store, and returns the
release directory there:

```python
path = client.download("chengsenwang/tsqa@1.0.0")
```

The release resolves as in `open()`. The call copies the release and each payload source that the
store lacks or holds damaged. Each one comes from the first registry other than the store that
provides it, so they can come from different registries. If the store holds a complete copy of the
release, the call copies only the payload sources that the store lacks, and does not compare the
files of the release again. If the store holds everything, the call copies nothing.

A local registry over the directory of the store counts as the store, also through a symlink or a
name in other letter case on a volume that ignores case. So `download(force=True)` never copies the
store onto itself.

A download never builds. For a release that only `InstalledConnectors()` provides, it raises
`TimeNetNotBuiltError`: `<release> is not built yet: only an installed connector provides it.` The
hint names `timenet.build('<id>')`. If a registry holds another version of the dataset, the hint
also names it, as `download('<id>@<version>')`. An exact version that no source provides, while an
installed connector builds another version, raises `TimeNetVersionUnavailableError`. A client
without a store raises `TimeFValidationError`.

A download also repairs a damaged copy of the release or of a payload source in the store, with a
new copy from a registry. If no registry other than the store provides the release, the call raises
`TimeNetFileError`, whose hint says how to replace the damaged copy. Use `force=True` to copy a
complete release again. This needs a registry other than the store that provides the release.
Without one, the call raises `TimeNetDatasetNotFoundError`.

A published version cannot change. So if the store holds the release with other files or another
card than the registry that the call copies from, the call raises `TimeNetVersionConflictError`,
also with `force=True`. The hint says
`Delete <store>/<id>/<version> to take the copy of <registry>.` The call compares the release only
with `force=True`, or for a damaged copy.

`progress=` takes a function that receives the progress events of `timenet.progress`. For each
release that the call copies, the function receives `StepStarted`, then `BytesMoved` events as the
files arrive, then `StepFinished`:

```python
from timenet.progress import BytesMoved

def show(event):
    if isinstance(event, BytesMoved):
        print(f"{event.url}: {event.done} of {event.total} bytes")

client.download("chengsenwang/tsqa", progress=show)
```

The store comes first in the default list, so a later `open()` or `load()` of the same version
reads the copy in the store. A remote read keeps the byte ranges that it reads in memory only, while
the copies of `download()` stay on disk.

## The store

The store keeps a provenance record for each release that a copy or a build puts into it. The
record says where the release comes from:

```json
{
  "origin": "copy",
  "source": "https://registry.timenet.ai",
  "at": "2026-11-03T14:02:11Z"
}
```

`origin` is `copy` for a release that the store copied from a registry, and `build` for a release
that an installed connector built. `source` names the registry, or the connector package and its
version. `at` is the time in UTC. A URL in the record has no user name, password or query, so no
credential lands in it. A release without a record counts as built.

The record is the file `.origin.json` in the release directory. It belongs to the store, not to the
release. The manifest does not list it, and `publish()` never copies it. To read it, call
`timenet.registry.origin.read_origin(store, release)`.

The record makes a release a download or a build. A download is a release whose record says
`copy`. A build is a release whose record says `build`, or a release without a record.
[`timenet local downloads clear`](cli/timenet.md#downloads-and-builds-in-the-store) deletes the
downloads, and `timenet local builds clear` deletes the builds. Each command keeps every release
that a release that stays needs: its payload sources, and the parents of a build. So the store
stays self-contained. `timenet local clear` deletes every release and the cache. Run a clear only
when no build or download runs.

## Pin a version

Append `@<version>` to a dataset ID:

```python
client.get_manifest("chengsenwang/tsqa@1.0.0")
client.load("chengsenwang/tsqa@latest")
```

The `get_manifest`, `get_card`, `plan_read`, `download`, `open`, `load`, and `open_torch` methods
also accept `version=`. Do not supply a version in both places.

## Open for PyTorch

`open_torch()` returns a read-only `torch.utils.data.Dataset`. Install the `torch` extra before you
use it. It reads through `open()`, one batch of records and their tasks at a time, so the dataset
does not have to fit in memory. The view holds an open reader. Close the view with `close()`, or
use it in a `with` block.

```python
import timenet
from timenet.types import read_value, read_values

client = timenet.TimeNet(sources=[".timenet-registry"])
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
