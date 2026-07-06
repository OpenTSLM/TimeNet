---
icon: lucide/cable
description: "Write a BaseConnector to turn a raw source into a TimeFDataset."
tags:
  - guide
  - connectors
---

# Connectors

The unit of dataset integration: one connector class per dataset. A connector fetches raw data and
converts it into a [`TimeFDataset`](timef-dataset.md). It has no knowledge of the registry, the engine,
or any other connector, and the consumer SDK never runs it.

The contract, `BaseConnector`, lives in the `timenet` package (`timenet.connectors`). Concrete
connectors live in the `timenet-connectors` repo alongside their [dataset card](manifest.md).

---

## `BaseConnector`

```python
from pathlib import Path
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset

class MyConnector(BaseConnector[MyRawRef]):
    def download(self, cache_dir: Path) -> list[MyRawRef]: ...
    def convert(self, raw_refs: list[MyRawRef]) -> TimeFDataset: ...
```

Two abstract stages, kept distinct so the engine can drive
`download -> convert -> derive_schema -> store`:

| Method | Nature | Contract |
| --- | --- | --- |
| `download(cache_dir)` | I/O only | Fetch/discover raw files, return lightweight references. Idempotent; no parsing. |
| `convert(raw_refs)` | CPU only | Parse references into a `TimeFDataset` with lazy Arrow loaders. No network. |

`metadata()` and `store()` are concrete methods you inherit, not stages you implement:

- `metadata()` reads and validates the dataset's [`dataset.yaml` card](manifest.md) from disk
  (`DatasetMetadata.from_yaml`), so it does file I/O; override it only to point at a different card.
  `metadata().dataset_id` must match the id the connector is curated under.
- `store()` writes the dataset through a [`TimeFWriter`](timef-writer.md), deriving the schema first if
  absent, and returns the committed version directory; most connectors never override it.

Connectors take **no constructor arguments** — configuration comes from environment variables read in
`__init__`. `TRaw` is whatever reference type the connector defines (a path, a small dataclass, an S3
key). Generic via PEP 695: `class MyConnector(BaseConnector[MyRawRef])`.

---

## Sharing data

To share time-series data across samples, attach the **same** `TimeSeries` instance (or two instances
with the same explicit `time_series_id`) to each sample. The writer dedupes by `time_series_id`, so the
bytes are stored once. The same applies to annotations, which the writer dedupes by `id`.

---

## Discovery and layout

Connectors are found **lazily by dataset id** — there is no central registry to maintain. A concrete
connector lives in its own folder at `datasets/<org>/<name>/` (lowercase Python package names): the
package's `__init__.py` exposes a module-level `CONNECTOR` and a `dataset.yaml` card sits beside it, so
`timenet-curate build <org>/<name>` imports just that package. Reusable bases live under `bases/`. Each
connector declares its own id in `metadata()`; ids are lowercase `org/name`.

## Optional dependencies and credentials

A connector may need libraries or credentials its source requires. Declare heavy libraries as an
**optional extra** and import them lazily inside the connector so base users don't have to install them;
a missing library should raise an actionable error. Credentials come from the environment — for the
HuggingFace Hub, a token is read from `HF_TOKEN` automatically (needed only for gated/private sources).
Downloaded source files cache under `<TIMENET_CACHE>` (see [client config](client.md#configuration)).

## Example connectors

- **`timenet/hello-world`** — a synthetic, offline reference connector. It needs no network and produces
  a fully deterministic dataset, so it doubles as the round-trip fixture: two modalities over a shared
  data source, a series shared across samples, a windowed sample, a chunk-split-sized series, all three
  annotation shapes (one shared), and a `ClassificationTask -> QATask` chain plus a `LabelingTask`. Its
  dataset card, `dataset.yaml`, sits beside it in `datasets/timenet/hello_world/`.
- **`chengsenwang/tsqa`** — a time-series QA dataset: each row's series becomes a `TimeSeries` and its
  question/answer a `QATask`. Needs the `huggingface` extra
  (`pip install 'timenet-connectors[huggingface]'`) since it downloads from the Hub.

```bash
timenet-curate build timenet/hello-world               # offline, synthetic
timenet-curate build chengsenwang/tsqa                 # live download from the Hub into the local registry
timenet-curate build chengsenwang/tsqa --keep-cache    # keep the raw sources for a faster rebuild
```

A successful build removes the dataset's raw download cache (`<TIMENET_CACHE>/<dataset_id>`), since the
sources are only needed during conversion; pass `--keep-cache` to retain them. For offline work and
tests, `TIMENET_TESTING=1` serves the connector's checked-in fixture, and `TIMENET_ROW_LIMIT=N` caps
rows for a quick sample.

Once built, load and inspect a dataset with the SDK — see `examples/load_tsqa.py`, which loads a dataset
and calls `describe()` to print its identity, counts, per-spec columns, and a sample preview.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
