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

The release lands in the local registry (default `~/.timenet/registry`), where
`TimeNet()` reads from. `--local-registry` selects another local registry. `--publish-to` publishes the release to another registry after the build:

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

Most parameters can also be specified as environment variables. For example, `$TIMENET_LOCAL_REGISTRY` & `TIMENET_ISOLATION=off`. These apply to every CLI command, and the Python SDK also honors them. 

For each build, the used Python and package and TimeNet versions are recorded to the build's manifest.

## Pipeline

The engine runs these stages:

1. Read and validate the connector's `dataset.yaml` card.
2. Prepare the parent datasets that the card declares, in the local registry. Before the build
   runs, the plan downloads or builds each missing parent there, as
   [Prepare parent releases](#prepare-parent-releases) describes.
3. Download raw artifacts into the build cache.
4. Compose the parents and the raw artifacts into a `TimeFDataset`. Record each parent in it.
5. Derive the dataset schema.
6. Write the complete TimeF version into the local registry.
7. Remove the raw cache after success, unless `--keep-cache` is set.

By default a build reuses an existing version already in the local registry and a status line says so. Use `--force` to skip this behavior. A forced build replaces the old version only after the new one is
complete, so a failed build keeps the old one. A published version never changes, so with
`--publish-to`, `--force` only affects the build to the local registry. The publish then only checks the rebuild against the already published version.

## Programmatic builds

Build and publish from Python with the producer functions of `timenet`:

```python
import timenet

# Build into the local registry.
timenet.build("org/name")

# Read the new release, like any other.
dataset = timenet.TimeNet().load("org/name")

# Copy the release and its payload sources to a registry.
timenet.publish("org/name", dest="s3://bucket/registry")
```

`timenet.build()` builds a release with the installed connector that claims its dataset ID, in an
[isolated environment](#isolated-build-environments). It returns the plan that ran: one step for
each release that it copied or built, in order. Each argument that you leave out comes from the
environment at the time of the call:

- `store` comes from `$TIMENET_LOCAL_REGISTRY`.
- `parents_from` comes from `$TIMENET_REMOTE_REGISTRY`, then `InstalledConnectors()`. With
  `TIMENET_AUTO_BUILD=false`, the connectors stay out.

So a default build lands where `TimeNet()` reads. `store` takes a local directory, a `file://` URI
or a `LocalRegistry`. `parents_from` lists, in order, where the parents and payload sources that the local registry lacks come from (see [Prepare parent releases](#prepare-parent-releases)).

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

`timenet.publish()` copies a complete release and the payload sources that it needs from `source`,
by default the local registry, to `dest`. Without a version in the dataset ID, it
copies the latest version of `source`. It mirrors the [`timenet publish`](cli/timenet.md#commands) CLI command.

A published version cannot change. If `dest` already holds the same release, `publish()` copies nothing.


A remote registry holds only releases with access declared `open`. Before
`timenet.publish()` copies anything to such a registry, it checks the card and the lineage of each
release.

## Prepare parent releases

A connector card can declare exact parent releases under `parents`. The connector reads these
releases when it composes the child dataset, preparing each parent release in the local registry that it builds into.

When resolving dependencies, the build uses the first of these steps that
works:

1. Reuse a complete copy in the local registry. `--local-registry` selects it.
2. Copy the release from the remote registry. `--remote-registry` selects it, else
   `$TIMENET_REMOTE_REGISTRY` (default `timenet://`).
3. Build the release from its own connector. `--no-auto-build`, or `TIMENET_AUTO_BUILD=false`,
   removes this step.

If no step works, the build stops before any download, and the error names the missing release.

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
