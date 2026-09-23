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
versions used for the build.

## Pipeline

The engine runs these stages:

1. Read and validate the connector's `dataset.yaml` card.
2. Download raw artifacts into the build cache.
3. Convert local artifacts into a `TimeFDataset`.
4. Derive the dataset schema.
5. Write or publish the complete TimeF version.
6. Remove the raw cache after success, unless `--keep-cache` is set.

`--force` rebuilds or republishes a version that already exists. Without this flag, the pipeline
returns the committed version and skips expensive work.

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

Both functions accept `force`, `keep_cache`, `cache_dir`, `values_backend`, and a writer progress
callback.

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

dataset = TimeNet(".timenet-registry").load("org/name")
dataset.describe()
```

See [Connectors](connectors.md) for the `download()` and `convert()` contract. See the
[`timenet.engine` API](api/engine.md) for the programmatic interface.
