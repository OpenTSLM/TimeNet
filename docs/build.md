---
icon: lucide/factory
description: "Build a connector into TimeF and publish the version to a registry."
tags:
  - guide
  - build
---

# Build and publish

A build turns a [connector](connectors.md)'s source data into an immutable TimeF version. The
version contains `manifest.json`, `control.duckdb`, and Parquet or Zarr Signal values.

Use `timenet-build` for normal builds:

```bash
timenet-build build timenet/hello-world
```

The release lands in the store, `$TIMENET_LOCAL_REGISTRY` (default `~/.timenet/registry`), where
`TimeNet()` reads. `--local-registry` selects another store. `--publish-to` publishes the release
and its payload sources to another registry after the build:

```bash
# Publish through the hosted registry API.
timenet-build build org/name --publish-to timenet://

# Publish to S3.
timenet-build build org/name --publish-to s3://bucket/prefix
```

See the [`timenet-build` reference](cli/build.md) for all options.

## Isolated build environments

Each connector can declare dependencies in its own `requirements.txt`. By default, the build tool
creates an isolated `uv` environment for those dependencies. Connectors with incompatible
requirements do not affect each other or your current environment.

Use `--no-isolation` while developing a connector in the current environment:

```bash
timenet-build build org/name --local-registry .timenet-registry --no-isolation
```

`TIMENET_ISOLATION=off` sets the same default for every build, also for `timenet.build()` in
Python. `--isolation` and `--no-isolation` win over it for one build. The manifest records the
Python and package versions used for the build.

If an isolated build cannot make its environment, it raises `TimeNetIsolationError`. Examples are a
missing `uv` executable, or a base package that is not installed. The hint names `--no-isolation`
in the CLI, and `TIMENET_ISOLATION=off` in Python.

## Pipeline

The engine runs these stages:

1. Read and validate the connector's `dataset.yaml` card.
2. Open the parent releases that the card declares. The store that the build writes to must hold
   them, as [Prepare parent releases](#prepare-parent-releases) describes.
3. Download raw artifacts into the build cache.
4. Compose the parents and the raw artifacts into a `TimeFDataset`. Record each parent in it.
5. Derive the dataset schema.
6. Write the complete TimeF version into the store. Then close the parent releases.
7. Remove the raw cache after success, unless `--keep-cache` is set.

`--force` builds a version again in the store. Without it, the build reuses the committed version,
and a status line says so. A damaged copy of the requested release in the store gets a new build,
also without `--force`. A forced build replaces the old version only after the new one is
complete, so a failed build keeps the old one. A published version never changes, so with
`--publish-to`, `--force` builds the release again in the store only. The publish then copies
nothing for the same files, and raises `TimeNetVersionConflictError` for other files.

The connector selects Parquet or Zarr through its `values_backend` attribute. Neither the build
command nor the Python API can change it.

## Programmatic builds

Build and publish from Python with the producer functions of `timenet`:

```python
import timenet

plan = timenet.build("org/name")  # into the store, parents through the list
timenet.publish("org/name", dest="s3://bucket/registry")
```

`timenet.build()` builds a release with the installed connector that claims its dataset ID, in an
[isolated environment](#isolated-build-environments). It returns the plan that ran: one step for
each release that it copied or built, in order. Each argument that you leave out comes from the
environment at the time of the call:

- `store` comes from `$TIMENET_LOCAL_REGISTRY`.
- `parents_from` comes from `$TIMENET_REMOTE_REGISTRY`, then `InstalledConnectors()`. With
  `TIMENET_AUTO_BUILD=false`, the connectors stay out.
- With both arguments, the store and the sources come only from the arguments. The cache,
  isolation and token settings still apply.

So a default build lands where `TimeNet()` reads. `store` takes a local directory, a `file://` URI
or a `LocalRegistry`, and it must not overlap the cache. `parents_from` lists, in order, where the
parents and payload sources that the store lacks come from (see
[Prepare parent releases](#prepare-parent-releases)), so an empty list allows no copy and no build
besides the requested release.

A version in `org/name@1.2.0` must be the version that the card of the connector declares, and the
build makes sure of it before anything runs. `force=True` builds the release again, even if the
store holds it. `open_access_only=True` refuses a plan that needs a release without open access,
with `TimeNetAccessError`, before any step runs. `keep_cache=True` keeps the raw downloads, and
`progress` receives the progress events of `timenet.progress`.

The errors of `build()` and `plan_build()` carry a hint in Python terms. For a missing parent, the
hint names a registry to add to `parents_from`. If an installed connector builds the parent, it
also names `InstalledConnectors()`.

```python
import timenet

plan = timenet.build(
    "org/name@1.2.0",
    store=".timenet-registry",
    parents_from=["timenet://", timenet.InstalledConnectors()],
)
for step in plan.steps:
    print(step.release, step.action)
```

`timenet.plan_build()` takes the arguments of `build()` without `keep_cache` and `progress`. It
returns the plan that `build()` runs with them, and copies and builds nothing.
`timenet-build build --dry-run` prints the same plan as a tree (see
[Output](cli/build.md#output)).

`timenet.publish()` copies a complete release and the payload sources that it needs from `source`,
by default the store, to `dest`, and returns the release. Without a version in the dataset ID, it
copies the latest version of `source`. The [`timenet publish`](cli/timenet.md#commands) command does
the same from the shell.

A published version cannot change. If `dest` holds the same release, `publish()` copies nothing.
The same release has the same manifest apart from the build environment: the metadata, the parent
cards, the lineage, and the file records. Another release under the version in `dest` raises
`TimeNetVersionConflictError`, and its hint asks for a new version. A local `dest` with a missing or
damaged file gets the release again.

If the default store lacks the release, the hint of the error names the call that puts it there:
`timenet.TimeNet().download('<id>')` if the remote registry holds it, or `timenet.build('<id>')` if
an installed connector builds it.

To build a connector object that no package installs, use the engine. `run_pipeline()` builds one
connector into a local directory. The directory must already hold the parents that its card
declares, for example from `timenet.build()` of each parent:

```python
from pathlib import Path

from timenet.engine import run_pipeline

version_dir = run_pipeline(connector, Path(".timenet-registry"))
```

`run_pipeline()` also accepts `force`, `keep_cache`, `cache_dir`, `pinned_by`, and `progress_cb`, a
callback for the progress of the writer. To publish the result, pass the directory to
`timenet.publish()` as `source`.

A registry that is not local holds only releases with access declared `open`. Before
`timenet.publish()` copies anything to such a registry, it checks the card and the lineage of each
release. A release without open access raises `TimeNetAccessError`. This includes a child whose card
declares `access: open` but whose parent is credentialed.

The error names the release and its effective access level, which is the strictest level of its
card and of the cards in its lineage. It also names each `access_url` that a user needs, with the
release that sets it. For a card, this is the `access_url` of the card.

## Prepare parent releases

A connector card can declare exact parent releases under `parents`. The connector reads these
releases when it composes the child dataset. Before the build downloads anything, it prepares each
parent release in the store that it builds into. It also prepares the payload sources of each
parent. A payload source is a release whose payload files the parent reads.

A missing release comes from the first source of the default list that provides it. The list is
the store, then the remote registry, then the installed connectors (see
[Settings](client.md#settings)). So for each release, the build uses the first of these steps that
works:

1. Reuse a complete copy in the store. `--local-registry` selects the store.
2. Copy the release from the remote registry. `--remote-registry` selects it, else
   `$TIMENET_REMOTE_REGISTRY` (default `timenet://`).
3. Build the release from its own connector. `--no-auto-build`, or `TIMENET_AUTO_BUILD=false`,
   removes this step.

If no step works, the build stops before any download, and the error names the missing release.
Its hint names `--remote-registry`, and also `--auto-build` if an installed connector builds the
release. The build prepares each parent before its children. If the store already holds the
requested version and `--force` is not set, the build only prepares the missing payload sources of
that version, with the same steps, and does not run the connector.

A copy takes a release with every payload source that it needs, at every depth, so the store never
holds a release without its payload sources. If the registry to copy from lacks a payload source,
the build gets it first with the same steps. A payload source that no step gets also stops the
build before any download.

```bash
# Copy missing parents from an S3 registry, and build none of them.
timenet-build build org/name \
    --remote-registry s3://bucket/prefix --no-auto-build
```

A parent builds like the requested dataset, in the environment of its own connector, but never
with `--force`. Each release builds through the builder backend that claims its dataset ID: the part
of an installed package that builds the datasets of its connectors. The `TimeNet` client builds a
missing release in the same way, as [Open, load, or download](client.md#open-load-or-download)
describes.

The build never builds over a parent or a payload source that the store holds. If a file of such a
copy is missing or has another size, the build copies the release again from a registry. The build
raises `TimeNetFileError` if no registry provides the release, if that copy fails, or if a copied
file does not match the size and checksum in its manifest.

If the list ends with the installed connectors, a missing payload source builds as a pinned
release. The build publishes the payload source only if it writes the files that the reading
release pins. See [Rebuild a pinned release](#rebuild-a-pinned-release) for the check and the two
fixes.

A registry that is not local holds only releases with access declared `open`. So for such a
`--publish-to`, the build makes sure that each release is open before any download, including the
releases that the store holds.

A registry that is not local serves the files of a release only if its access is `open`. If it
only lists a release, for example a credentialed one, the build builds the release from its own
connector. With `--no-auto-build`, the build stops before any download with `TimeNetAccessError`,
which names the page where you get access.

[`timenet.build()`](#programmatic-builds) takes the same steps from one list, `parents_from`. The
first entry that provides a release wins: a registry gives a copy, and `InstalledConnectors()` gives
a build. The release lands in the store with each payload source that it reads, so
`TimeNet().load()` then reads it from the store:

```python
import timenet

timenet.build(
    "org/name@1.2.0",
    parents_from=["timenet://", timenet.InstalledConnectors()],
)
dataset = timenet.TimeNet().load("org/name@1.2.0")
```

## Rebuild a pinned release

A release can read payload files that another release stores. The reading release pins each of
these files by path, size, and checksum (see [Shared payloads](timef-format.md#shared-payloads)).
A rebuild of the stored release must write the same files.

Pass the pins to `run_pipeline()` as `pinned_by`, keyed by the reading release. The writer builds
the release in a staging directory. Before it publishes the release, it compares each pinned file
with the file that it wrote. If a pinned file differs, the writer deletes the staging directory and
raises `TimeFFormatError`. Nothing is published.

The error names two fixes:

- Fetch the exact release from a registry that holds it.
- Rebuild the reading release with force, so that it pins the new files. The releases that pin the
  old files of the reading release then fail to open.

## Authoring loop

1. Add a connector package and dataset card under
   `timenet_connectors/datasets/<org>/<name>/`.
2. Build into an explicit local registry.
3. Load that same registry with the SDK and inspect the result.
4. Run the connector and format checks.
5. Publish the complete version to its destination registry.

The local check keeps build and load paths explicit:

```python
import timenet

client = timenet.TimeNet(sources=[".timenet-registry"])
with client.open("org/name") as reader:
    reader.read().describe()
```

See [Connectors](connectors.md) for the `download()` and `convert()` contract. See the
[`timenet.engine` API](api/engine.md) for the programmatic interface.
