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
from timenet.types import DatasetMetadata

class MyConnector(BaseConnector[MyRawRef]):
    def metadata(self) -> DatasetMetadata: ...
    def download(self, cache_dir: Path) -> list[MyRawRef]: ...
    def convert(self, raw_refs: list[MyRawRef]) -> TimeFDataset: ...
```

Three abstract stages, kept distinct so the engine can drive
`download -> convert -> derive_schema -> store`:

| Method | Nature | Contract |
| --- | --- | --- |
| `metadata()` | pure | Return `DatasetMetadata`; derivable from constructor state, no I/O. `metadata().dataset_id` must match the id the connector is curated under. |
| `download(cache_dir)` | I/O only | Fetch/discover raw files, return lightweight references. Idempotent; no parsing. |
| `convert(raw_refs)` | CPU only | Parse references into a `TimeFDataset` with lazy Arrow loaders. No network. |

Connectors take **no constructor arguments** — configuration comes from environment variables read in
`__init__`. `TRaw` is whatever reference type the connector defines (a path, a small dataclass, an S3
key). Generic via PEP 695: `class MyConnector(BaseConnector[MyRawRef])`.

`store()` (writing the dataset to disk) is not part of the connector — it is the engine/writer's job and
lands with [`TimeFWriter`](timef-writer.md).

---

## Sharing data

To share time-series data across samples, attach the **same** `TimeSeries` instance (or two instances
with the same explicit `time_series_id`) to each sample. The writer dedupes by `time_series_id`, so the
bytes are stored once. The same applies to annotations, which the writer dedupes by `id`.

---

## Discovery and layout

Connectors are found **lazily by dataset id** — there is no central registry to maintain. A concrete
connector lives at `datasets/<org>/<name>.py` (lowercase Python package names) and exposes a
module-level `CONNECTOR`; `timenet-curate build <org>/<name>` imports just that module. Reusable bases
live under `bases/`. Each connector declares its own id in `metadata()`; ids are lowercase `org/name`.

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
  dataset card, `hello_world.yaml`, sits beside it.
- **`chengsenwang/tsqa`** — a time-series QA dataset: each row's series becomes a `TimeSeries` and its
  question/answer a `QATask`. `TIMENET_ROW_LIMIT` caps rows for large runs.

```bash
TIMENET_TESTING=1 timenet-curate build chengsenwang/tsqa   # offline, uses the connector's fixture
timenet-curate build chengsenwang/tsqa                     # live download
```
