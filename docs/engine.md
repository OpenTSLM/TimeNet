# Engine

The curation runtime that turns a [connector](connectors.md) into a stored dataset. It only writes
local files; publishing to a remote registry is a separate step. Lives in `timenet.engine`.

## Pipeline

```python
from timenet.engine import run_pipeline

version_dir = run_pipeline(connector, root, cache_dir=None, progress_cb=None)
```

`run_pipeline` runs four stages:

1. **cache** — create `cache_dir` (defaults to `<root>/.cache/<dataset_id>`).
2. **download** — `connector.download(cache_dir)` fetches raw references.
3. **convert** — `connector.convert(raw_refs)` builds a `TimeFDataset`.
4. **derive_schema + store** — derive the schema, then `connector.store()` streams it through
   [`TimeFWriter`](timef-writer.md) and returns the committed version directory.

For a local registry, `store` *is* the publish (the output directory is itself a valid local registry).
Publishing to a remote registry happens afterward via the curation CLI.

Distributed (Ray-backed) scheduling is out of scope for the current implementation.
