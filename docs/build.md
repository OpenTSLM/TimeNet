---
icon: lucide/factory
description: "Build: run a connector through the pipeline and publish a dataset to a registry."
tags:
  - guide
  - build
---

# Build & publish

Build turns a [connector](connectors.md)'s raw source into a stored TimeF version. A version is a
`manifest.json`, Parquet control tables, and a Parquet or Zarr values plane. TimeNet writes the version
into a [registry](registry.md). Build runs on your machine. Today it publishes to a local registry. A
hosted backend is planned. The command [`timenet-build build`](cli/build.md) drives it. This page
explains what happens underneath.

## The pipeline

Each build runs in its own environment. Before the pipeline starts, `timenet-build` resolves the
connector's `requirements.txt` (see [Connectors](connectors.md)). Then it re-runs itself under `uv`.
The build environment layers those requirements over the same `timenet` and `timenet-connectors`
that the parent runs. Two connectors that need incompatible libraries no longer collide. The
[manifest](manifest.md) records the build environment as `build_env`.

To run the pipeline in the current interpreter instead, pass `--no-isolation` or set
`TIMENET_ISOLATION=off`. Use this while you write a connector. The programmatic entry point
`timenet_connectors.build()` works the same way. It is isolated by default. It runs in-process when
`TIMENET_ISOLATION=off`.

The engine runs one connector through four stages in `timenet.engine.run_pipeline`:

```python
from timenet.engine import run_pipeline

run_pipeline(
    connector, root, *,
    cache_dir=None, clean_cache=False, progress_cb=None, force=False,
)
```

1. cache: create `cache_dir`. The default is `<TIMENET_CACHE>/<dataset_id>`. If you set
   `clean_cache=True`, the engine removes `cache_dir` after a successful build.
2. download: `connector.download(cache_dir)` fetches the raw references. Only this stage touches the
   network.
3. convert: `connector.convert(raw_refs)` builds an in-memory [`TimeFDataset`](timef-dataset.md).
4. derive_schema and store: the engine derives the schema first. It then calls `store_dataset()`,
   which streams the dataset through [`TimeFWriter`](timef-writer.md) and returns the committed
   version directory.

`run_pipeline` is idempotent. If a version is already committed, it short-circuits, unless you pass
`force=True`. Distributed (Ray-backed) scheduling is out of scope for now.

## Publishing

For a local registry, `store` is the publish step. The output directory is itself a valid local
registry. [`WritableRegistry.store`](registry.md#writing-to-a-registry) is the general primitive. The S3
and hosted backends will implement it. Publishing to those backends arrives when they do.

## The authoring loop

Building a dataset follows one path:

1. Add a connector at `datasets/<org>/<name>/` in `timenet-connectors`. Its `__init__.py` exposes a
   [`BaseConnector`](connectors.md) as `CONNECTOR`.
2. Put its [dataset card](manifest.md), `dataset.yaml`, beside it. When the connector loads the card,
   TimeNet validates it against the packaged `dataset-card.schema.json`.
3. Build it with [`timenet-build build`](cli/build.md).
4. Verify the dataset: point the SDK at the output directory. The output directory is itself a valid
   local registry.
5. When a hosted backend is available, publish the dataset.

See [Connectors](connectors.md) to learn how to write the `download` and `convert` steps. See the
[`timenet.engine` API](api/engine.md) for the full symbol listing.
