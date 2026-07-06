---
icon: lucide/cog
description: "The curation runtime that turns a connector into a stored dataset."
tags:
  - guide
  - engine
---

# Engine

The curation runtime that turns a [connector](connectors.md) into a stored dataset. It publishes into a
local registry today; the registry's [`WritableRegistry.store`](registry.md#writing-to-a-registry) is the
general publish primitive that the S3 and remote backends will implement. Lives in `timenet.engine`.

## Pipeline

```python
from timenet.engine import run_pipeline

run_pipeline(connector, root, *, cache_dir=None, clean_cache=False, progress_cb=None, force=False)
```

`run_pipeline` is idempotent — an already-committed version short-circuits unless `force=True` — and runs
four stages:

1. **cache** — create `cache_dir` (defaults to `<TIMENET_CACHE>/<dataset_id>`); `clean_cache=True` removes
   it after a successful build.
2. **download** — `connector.download(cache_dir)` fetches raw references.
3. **convert** — `connector.convert(raw_refs)` builds a `TimeFDataset`.
4. **derive_schema + store** — derive the schema, then `connector.store()` streams it through
   [`TimeFWriter`](timef-writer.md) and returns the committed version directory.

For a local registry, `store` *is* the publish — the output directory is itself a valid local registry.
Publishing to an S3 or remote registry lands when those backends implement `store`.

Distributed (Ray-backed) scheduling is out of scope for the current implementation.

---

See the [API reference for `timenet.engine`](api/engine.md) for the full symbol listing.
