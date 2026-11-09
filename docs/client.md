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
    timenet.LocalRegistry(".timenet-registry"),
    "https://registry.example.com",
    "timenet://",
    timenet.InstalledConnectors(),
])
```

- `InstalledConnectors()` stands for every installed package that registers a builder backend,
  such as `timenet-connectors`. If only this source provides a release, a read builds it. Without
  it in the list, a client never runs connector code.
- The first `LocalRegistry` of the list that is not `read_only=True` is the local registry where
  downloads and builds land. A path in the list opens such a registry too. The client only reads a
  later local registry. `LocalRegistry(path, read_only=True)` is never written, for example a
  mirror on a network drive. A list with `InstalledConnectors()` needs a writable local registry.
- A remote registry reads only its own files.

The order of the list is the read order. A release that two sources hold is read from the first
one. So to read offline, put the local registry first in the list.

Without a version, the first source that serves the dataset gives its newest version. An exact
version comes from the first source that provides it, while `@latest` is the highest version that
any source provides. Only "not found" and "not served" fall through to the next source, and any
other error stops the call. So if a registry in the list does not answer, a request for `@latest`
fails.

A client that must never write takes `read_only=True`. A list of remote registries alone has no
local registry, so it never writes either:

```python
import timenet

local = timenet.TimeNet(sources=[timenet.LocalRegistry(".timenet-registry", read_only=True)])
hosted = timenet.TimeNet(sources=["timenet://"])
s3 = timenet.TimeNet(sources=["s3://my-bucket/timenet"])
```

`TimeNet()` uses the default list of `timenet.client.default_sources()`. If no variable is set,
the list is:

```python
[
    LocalRegistry("~/.timenet/registry"),
    "timenet://",
    InstalledConnectors(),
]
```

`TIMENET_LOCAL_REGISTRY` moves the local registry, `TIMENET_REMOTE_REGISTRY` replaces `timenet://`,
and `TIMENET_AUTO_BUILD=false` removes `InstalledConnectors()` (see [Settings](#settings)). The
arguments `local_registry`, `remote_registry` and `auto_build` of `default_sources()` override
them, and `timenet.client.default_local_registry()` returns the local registry alone.

## Settings

These environment variables configure TimeNet. Each variable sets one thing.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TIMENET_LOCAL_REGISTRY` | `~/.timenet/registry` | The local registry, as a local path or a `file://` URI. It is the first source that a lookup checks, and the place where downloads and builds land. |
| `TIMENET_REMOTE_REGISTRY` | `timenet://` | A remote registry, as a URL or a local path: the second source that a lookup checks, if the local registry does not have the dataset. |
| `TIMENET_AUTO_BUILD` | `true` | Whether installed connectors build a dataset that neither the local nor the remote registry has. If `true`, the default list ends with `InstalledConnectors()`. |
| `TIMENET_CACHE` | `~/.cache/timenet` | Raw downloads of builds in `downloads/`, and control databases of remote reads in `control/`. |
| `TIMENET_TOKEN` | unset | Bearer token for a remote registry. |
| `TIMENET_ISOLATION` | `on` | Whether connector builds use isolated environments. |

An explicit argument, for example a CLI flag, wins over its variable, and the variable wins over its
default.

## Browse datasets

`list()` and `search()` answer for the whole list of sources. They return one row per dataset, at
the highest version that a source lists, sorted by dataset ID. The card of each installed connector
counts as a row. So `list()` also shows a dataset that only a connector provides.

| Method | Result |
| --- | --- |
| `list()` | Metadata for the highest version of every dataset across the sources. |
| `search(...)` | Metadata that matches registry filters, across the sources. |

`search()` accepts filters for free text, domains, task classes, licenses, Signal specifications,
dataset IDs, and tags. See [Registry search](registry.md#search) for their matching rules. A
connector never matches `task` or `time_series_spec`, because both filters read the schema of a
manifest.

The calls ask the sources one after another, in order. If two sources list a dataset at the same
version, the row of the earlier source stays. A listed version can be newer than the version that
`open()` reads without a version. To query one registry alone, open it and ask it:

```python
from timenet.registry import open_registry
from timenet.types import DatasetFilter

hits = open_registry("timenet://").search(DatasetFilter(query=("ecg",)))
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
`InstalledConnectors()` provides, `get_manifest()` raises `TimeNetNotBuiltError`.

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
the sources by the rules of [Choose the sources](#choose-the-sources). So without a version, the
call reads the newest version of the first source that serves the dataset, for example the local
registry. With `@latest`, if an installed connector declares a newer version than the local
registry holds, the call builds the newer version.

```python
import timenet

client = timenet.TimeNet(sources=[
    timenet.LocalRegistry(".timenet-registry"),
    "timenet://",
])
with client.open("org/name") as reader:
    records = reader.read().records
```

This list has no `InstalledConnectors()`, so a missing release raises an error instead of starting
a build.

If only `InstalledConnectors()` provides a release, the call builds it into the local registry
first and reads it there. The build copies or builds each missing parent from the same sources, as
[Prepare parent releases](build.md#prepare-parent-releases) describes.

When `open()`, `load()`, or `open_torch()` reads from an HTTP or S3 registry, it reads the remote
files by range.

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

`download()` copies the complete release and its payload sources into the local registry, and
returns the release directory there:

```python
path = client.download("chengsenwang/tsqa@1.0.0")
```

The release resolves as in `open()`. The call copies the release and each payload source that the
local registry lacks or holds damaged. Each one comes from the first other registry that provides
it, so they can come from different registries. If the local registry holds a complete copy of the
release, the call copies only the payload sources that it lacks.

A download never builds. For a release that only `InstalledConnectors()` provides, it raises
`TimeNetNotBuiltError`.

A published version cannot change. So if the local registry holds the release with other files or
another card than the registry that the call copies from, the call raises
`TimeNetVersionConflictError`.

`progress=` takes a function that receives the progress events of `timenet.progress`. For each
release that the call copies, the function receives `StepStarted`, then `BytesMoved` events as the
files arrive, then `StepFinished`:

```python
from timenet.progress import BytesMoved

def show(event):
    if isinstance(event, BytesMoved):
        print(f"{event.subject}: {event.done} of {event.total} bytes")

client.download("chengsenwang/tsqa", progress=show)
```

The local registry comes first in the default list, so a later `open()` or `load()` of the same
version reads the copy in the local registry. A remote read keeps the byte ranges that it reads in
memory only, while the copies of `download()` stay on disk.

## The local registry

The local registry where downloads and builds land keeps a provenance record for each release that
a copy or a build puts into it. The record says where the release comes from:

```json
{
  "kind": "download",
  "source": "https://registry.timenet.ai",
  "at": "2026-11-03T14:02:11Z"
}
```

`kind` is `download` for a release copied from a registry, and `build` for a release
that an installed connector built. `source` names the registry, or the connector package and its
version. To read it, call `store.read_origin(release)` on the `LocalRegistry`. It returns an
`Origin` from `timenet.registry.origin`, whose `kind` is an `OriginKind`.

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

See the [API reference for `timenet.client`](api/timenet.client.md) for every parameter and return type.
