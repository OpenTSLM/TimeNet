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

Two stages, kept distinct so the engine can drive
`download -> convert -> derive_schema -> store`:

| Method | Nature | Contract |
| --- | --- | --- |
| `download(cache_dir)` | I/O only | Fetch/discover raw files, return lightweight references. Idempotent; no parsing. |
| `convert(raw_refs)` | CPU only | Parse references into a `TimeFDataset` with lazy Arrow loaders. No network. |

`convert` is the only method you must implement. `download` is I/O-bound, so it comes in two shapes:
override `download(cache_dir)` for a synchronous fetch, or `async download_async(cache_dir)` to fetch
artifacts concurrently. Implement exactly one. The engine always calls the synchronous `download()`,
whose default drives `download_async` to completion, so an async connector needs no event-loop wiring
of its own.

`metadata()` and `store()` are concrete methods you inherit, not stages you implement:

- `metadata()` reads and validates the dataset's [`dataset.yaml` card](manifest.md) from disk
  (`DatasetMetadata.from_yaml`), so it does file I/O; override it only to point at a different card.
  `metadata().dataset_id` must match the id the connector is curated under.
- `store()` writes the dataset through a [`TimeFWriter`](timef-writer.md), deriving the schema first if
  absent, and returns the committed version directory; most connectors never override it.

Connectors take no constructor arguments. Configuration comes from environment variables read in
`__init__`. `TRaw` is whatever reference type the connector defines (a path, a small dataclass, an S3
key). Generic via PEP 695: `class MyConnector(BaseConnector[MyRawRef])`.

---

## Sharing data

To share time-series data across samples, attach the same `TimeSeries` instance (or two instances
with the same explicit `time_series_id`) to each sample. The writer dedupes by `time_series_id`, so the
bytes are stored once. The same applies to annotations, which the writer dedupes by `id`.

---

## Discovery and layout

Connectors are found lazily by dataset id: there is no central registry to maintain. A concrete
connector lives in its own folder at `datasets/<org>/<name>/` (lowercase Python package names): the
package's `__init__.py` exposes a module-level `CONNECTOR` and a `dataset.yaml` card sits beside it, so
`timenet-curate build <org>/<name>` imports just that package. Reusable bases live under `bases/`. Each
connector declares its own id in `metadata()`; ids are lowercase `org/name`.

## Optional dependencies and credentials

A connector may need libraries or credentials its source requires. Declare heavy libraries as an
**optional extra** and import them lazily inside the connector so base users don't have to install them;
a missing library should raise a clear error. Credentials come from the environment. For the
HuggingFace Hub, a token is read from `HF_TOKEN` automatically (needed only for gated/private sources).
Downloaded source files cache under `<TIMENET_CACHE>` (see [client config](client.md#configuration)).

## Downloading artifacts

`timenet_connectors.download` has two async helpers that pick the backend from each URL's scheme,
so a connector never branches on `s3://` vs `http(s)://` itself:

- `fetch_files([Artifact(url, dest), ...])` downloads a list, mixing schemes freely: HTTP entries run
  concurrently (bounded by `max_concurrency`, sharing one connection pool), S3 entries one at a time. A
  single file is a one-element list.
- `ensure_archive(url, target)` downloads a zip and extracts it into `target`, idempotently: a marker
  file records success, so a re-run reuses the extracted contents and skips the download.

Each `Artifact` takes optional `headers`, `cookies`, and a `sha256` to verify the download; `fetch_files`
also takes batch-level `headers`/`cookies` applied to every HTTP request, with the per-artifact ones
merged over them (all ignored for S3). HTTP downloads (`aiohttp` + `aiofiles`, both base deps) stream to
disk and write atomically through a `.part` temp file (a SHA-256 mismatch raises and leaves nothing
behind), skipping an existing destination. S3 downloads use boto3 and stay synchronous, since boto3
already parallelizes a single object's transfer. `ensure_archive` also takes a `filename` override for
URLs whose path has no usable name (a trailing slash or `/download` suffix). Call them from
`download_async`:

```python
from timenet_connectors.download import Artifact, ensure_archive, fetch_files

class MyConnector(BaseConnector[MyRawRef]):
    async def download_async(self, cache_dir):
        await fetch_files(
            [
                Artifact("https://host/a.csv", cache_dir / "a.csv"),
                Artifact("s3://bucket/b.csv", cache_dir / "b.csv"),
            ],
            headers={"Authorization": "Bearer …"},  # applied to every HTTP request
        )
        await ensure_archive("https://host/records.zip", cache_dir)
        return [...]  # lightweight references into cache_dir
```

Progress is reported through an ambient sink (`timenet_connectors.download.progress`): the download helpers
emit `DownloadProgress` events and the `timenet-curate` CLI renders them, so downloads show progress with
no `progress` argument threaded through the connector. On a terminal they render as live progress bars,
one row per file updating in parallel; piped output falls back to throttled text lines, and `--quiet`
silences them.

PhysioNet connectors extend `BasePhysioNetConnector` for WFDB record I/O and use `ensure_archive` to pull
their database archive.

## Example connectors

- `timenet/hello-world` is a synthetic, offline reference connector. It needs no network and produces
  a fully deterministic dataset, so it doubles as the round-trip fixture: two modalities over a shared
  data source, a series shared across samples, a windowed sample, a chunk-split-sized series, all three
  annotation shapes (one shared), and a `ClassificationTask -> AnswerTask` chain plus a scalar
  prediction, a temporal localization, and a scoped classification. Its
  dataset card, `dataset.yaml`, sits beside it in `datasets/timenet/hello_world/`.
- `chengsenwang/tsqa` is a time-series QA dataset: each row's series becomes a `TimeSeries` and its
  question/answer an `AnswerTask`. Needs the `huggingface` extra
  (`pip install 'timenet-connectors[huggingface]'`) since it downloads from the Hub.

```bash
timenet-curate build timenet/hello-world             # offline, synthetic
timenet-curate build chengsenwang/tsqa               # live, from the Hub
timenet-curate build chengsenwang/tsqa --keep-cache  # keep the raw sources
```

A successful build removes the dataset's raw download cache (`<TIMENET_CACHE>/<dataset_id>`), since the
sources are only needed during conversion; pass `--keep-cache` to retain them.

Keeping `download` and `convert` apart is what makes a connector testable offline: `convert` takes raw
references and touches no network, so a test hands it a checked-in fixture and skips `download`
entirely. See `packages/timenet-connectors/tests/fixtures/` and the `_convert()` helpers beside them.

Once built, load and inspect a dataset with the SDK. See `examples/load_tsqa.py`, which loads a dataset
and calls `describe()` to print its identity, counts, per-spec columns, and a sample preview.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
