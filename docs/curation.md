---
icon: lucide/factory
description: "Curation: run a connector through the pipeline and publish a dataset to a registry."
tags:
  - guide
  - curation
---

# Curate & publish

Curation turns a [connector](connectors.md)'s raw source into a stored dataset: a `manifest.json` plus
its parquet, written into a [registry](registry.md). It runs on your machine and publishes to a local
registry today, with a hosted backend planned. The command that drives it is
[`timenet-curate build`](cli/curate.md); this page covers what happens underneath.

## The pipeline

The engine runs one connector through four stages, in `timenet.engine.run_pipeline`:

```python
from timenet.engine import run_pipeline

run_pipeline(connector, root, *, cache_dir=None, clean_cache=False, progress_cb=None, force=False)
```

1. cache: create `cache_dir` (defaults to `<TIMENET_CACHE>/<dataset_id>`). With `clean_cache=True` it is
   removed after a successful build.
2. download: `connector.download(cache_dir)` fetches the raw references. This is the only stage that
   touches the network.
3. convert: `connector.convert(raw_refs)` builds an in-memory [`TimeFDataset`](timef-dataset.md).
4. derive_schema and store: derive the schema, then `connector.store()` streams it through
   [`TimeFWriter`](timef-writer.md) and returns the committed version directory.

`run_pipeline` is idempotent: an already-committed version short-circuits unless you pass `force=True`.
Distributed (Ray-backed) scheduling is out of scope for now.

## Publishing

For a local registry, `store` is the publish step, since the output directory is itself a valid local
registry. [`WritableRegistry.store`](registry.md#writing-to-a-registry) is the general primitive the S3
and hosted backends will implement, so publishing to those arrives when they do.

## The authoring loop

Building a dataset follows one path:

1. Add a connector at `datasets/<org>/<name>/` in `timenet-connectors`. Its `__init__.py` exposes a
   [`BaseConnector`](connectors.md) as `CONNECTOR`.
2. Put its [dataset card](manifest.md), `dataset.yaml`, beside it. The card is validated against the
   packaged `dataset-card.schema.json` when the connector loads it.
3. Build it with [`timenet-curate build`](cli/curate.md).
4. Verify by pointing the SDK at the output directory, which is itself a valid local registry.
5. Publish once a hosted backend is available.

See [Connectors](connectors.md) for how to write the `download` and `convert` steps, and the
[`timenet.engine` API](api/engine.md) for the full symbol listing.
