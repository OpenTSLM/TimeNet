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
timenet-build build timenet/hello-world --out .timenet-registry
```

The output can be a local directory or a writable registry URL.

```bash
# Publish through the hosted registry API.
timenet-build build org/name --out timenet://

# Publish to S3.
timenet-build build org/name --out s3://bucket/prefix
```

See the [`timenet-build` reference](cli/build.md) for all options.

## Isolated build environments

Each connector can declare dependencies in its own `requirements.txt`. By default, the build tool
creates an isolated `uv` environment for those dependencies. Connectors with incompatible
requirements do not affect each other or your current environment.

Use `--no-isolation` while developing a connector in the current environment:

```bash
timenet-build build org/name --out .timenet-registry --no-isolation
```

`TIMENET_ISOLATION=off` provides the same setting. The manifest records the Python and package
versions used for the build, and the git commit of `timenet` and `timenet-connectors`. See
[`build_env`](manifest.md) for the fields.

## Pipeline

The engine runs these stages:

1. Read and validate the connector's `dataset.yaml` card. If the output registry is not local, the
   card must declare `access: open`, and the TimeNet code must be traceable to a commit, as
   [Traceable builds](#traceable-builds) describes.
2. Open the parent releases that the card declares. The output registry must hold them, as
   [Prepare parent releases](#prepare-parent-releases) describes.
3. Download raw artifacts into the build cache.
4. Compose the parents and the raw artifacts into a `TimeFDataset`. Record each parent in it.
5. Derive the dataset schema.
6. Write or publish the complete TimeF version. Then close the parent releases. Before a publish to
   a registry that is not local, the build makes sure that each card in the lineage also declares
   `access: open`.
7. Remove the raw cache after success, unless `--keep-cache` is set.

`--force` rebuilds or republishes a version that already exists. Without this flag, the pipeline
returns the committed version and skips expensive work. A forced local build replaces the old
version only once the new one is complete, so a failed rebuild keeps it.

The connector selects Parquet or Zarr by default. Override it when you need to inspect another
backend:

```bash
timenet-build build timenet/hello-world \
    --out .timenet-registry \
    --values-backend zarr
```

## Programmatic builds

Use `run_pipeline()` for a local directory:

```python
from pathlib import Path

from timenet.engine import run_pipeline

version_dir = run_pipeline(
    connector,
    Path(".timenet-registry"),
    values_backend="parquet",
)
```

Use `publish_pipeline()` with a writable registry:

```python
from timenet.engine import publish_pipeline
from timenet.registry import open_writable_registry

registry = open_writable_registry("s3://bucket/prefix")
version = publish_pipeline(connector, registry)
```

Both functions accept `force`, `keep_cache`, `cache_dir`, `values_backend`, `pinned_by`, and a
writer progress callback.

A registry that is not local holds only releases with access declared `open`. `publish_pipeline()`
checks the card before it downloads anything. A credentialed or restricted card raises
`TimeNetAccessError` for such a registry. After the connector composes the dataset,
`publish_pipeline()` checks the dataset again with its lineage, before it writes. If the card of a
child declares `access: open` but its parent is credentialed, the second check raises the error.

The error names the release and its effective access level, which is the strictest level of its
card and of the cards in its lineage. It also names each `access_url` that a user needs, with the
release that sets it. For a card, this is the `access_url` of the card.

## Traceable builds

A release in a registry that is not local outlives the machine that built it. Its manifest must
name the exact code that built it, so a publish to such a registry needs a known git commit for
each installed `timenet` and `timenet-connectors` distribution, and no uncommitted changes in their
package directories. The [manifest](manifest.md) describes where each install finds its commit.
There is no override. Commit the changes first, or build to a local directory, which records the
state and does not refuse.

The rule applies to every way of publishing:

- `timenet-build build --out <remote>` refuses before uv builds the isolated environment.
- `publish_pipeline()` refuses before it downloads anything.
- `S3Registry.store()` and `RemoteRegistry.store()` refuse before they write the release.

A build reads git once, when it starts, and records that state in the manifest. Before the upload,
the registry reads git again. If the code changed while the build ran, for example because you
switched branches, the registry raises `TimeNetBuildError` and uploads nothing. Build again.

The check does not make sure that the commit exists on GitHub. Push the commit before you publish,
so that other people can find the code that a manifest names.

## Prepare parent releases

A connector card can declare exact parent releases under `parents`. The connector reads these
releases when it composes the child dataset. Before the build downloads anything, it prepares each
parent release in the output registry. It also prepares the payload sources of each parent. A
payload source is a release whose payload files the parent reads.

For each release, the build uses the first of these steps that works:

1. Reuse a complete copy in the output registry.
2. Copy the release from the registry that `--fetch-from` names.
3. If `--build-parents` is set, build the release from its own connector.

If no step works, the build stops before any download. The error names the missing release. The
build prepares each parent before its children. If the output already holds the requested version
and `--force` is not set, the build prepares only the missing payload sources of that version. It
uses the same steps in the same order. The build does not run the connector of the requested
dataset.

```bash
timenet-build build org/name --out .timenet-registry \
    --fetch-from timenet:// --build-parents
```

A parent builds like the requested dataset, in the environment of its own connector. A parent never
builds with `--force`. If the builder backend of another installed package claims a parent, that
backend builds it.

The build never builds over a release that the output holds. If a file of the copy in the output
is missing or has another size, the build copies the release again from `--fetch-from`. If the build
cannot copy it, the build raises `TimeNetFileError`. If a copied file does not match the size and
checksum that its manifest records, the build raises `TimeNetFileError`.

If `--build-parents` is set, a missing payload source builds as a pinned release. The build
publishes the payload source only if it writes the files that the reading release pins. See
[Rebuild a pinned release](#rebuild-a-pinned-release) for the check and the two fixes.

`--fetch-from` needs a local `--out`. An output that is not local holds only releases with access
declared `open`. So the build checks the access of each release before any download. A copy from a
registry that is not local also needs access declared `open`.

`timenet_connectors.build()` and `timenet_connectors.load()` take the same options as keywords.
`fetch_from` takes a registry or its URI. `build_parents=True` builds missing parents.

```python
import timenet_connectors

dataset = timenet_connectors.load(
    "org/name",
    fetch_from="timenet://",
    build_parents=True,
)
```

## Rebuild a pinned release

A release can read payload files that another release stores. The reading release pins each of
these files by path, size, and checksum (see [Shared payloads](timef-format.md#shared-payloads)).
A rebuild of the stored release must write the same files.

Pass the pins to `run_pipeline()` or `publish_pipeline()` as `pinned_by`, keyed by the reading
release. The writer builds the release in a staging directory. Before it publishes the release, it
compares each pinned file with the file that it wrote. If a pinned file differs, the writer deletes
the staging directory and raises `TimeFFormatError`. Nothing is published.

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
from timenet.client import TimeNet

with TimeNet(".timenet-registry").open("org/name") as reader:
    reader.read().describe()
```

See [Connectors](connectors.md) for the `download()` and `convert()` contract. See the
[`timenet.engine` API](api/engine.md) for the programmatic interface.
