---
icon: lucide/cable
description: "Write a BaseConnector to turn a raw source into a TimeFDataset."
tags:
  - guide
  - connectors
---

# Connectors

A connector is the unit of dataset integration. Each dataset has one connector class. A connector
fetches raw data and converts it into a [`TimeFDataset`](timef-dataset.md). A connector has no
knowledge of the registry, the engine, or other connectors. The consumer SDK never runs a connector.

`BaseConnector` is the contract for a connector. It lives in the `timenet` package
(`timenet.connectors`). Concrete connectors live in the `timenet-connectors` repo, next to their
[dataset card](manifest.md).

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

The two stages stay separate. This lets the engine drive
`download -> convert -> derive_schema -> store`.

| Method | Nature | Contract |
| --- | --- | --- |
| `download(cache_dir)` | I/O only | Fetch or find raw files and return lightweight references. The method is idempotent and does not parse data. |
| `convert(raw_refs)` | CPU only | Parse the references into a `TimeFDataset` with lazy Arrow loaders. The method does not use the network. |

You must implement `convert`. It is the only required method. `download` depends on I/O, so it has
two forms. You can override `download(cache_dir)` for a synchronous fetch. Alternatively, you can
define `async download_async(cache_dir)` to fetch artifacts at the same time. Implement only one of
these two forms. The engine always calls the synchronous `download()`. Its default implementation
runs `download_async` to completion. As a result, an async connector needs no event loop code of its
own.

`metadata()` and `store()` are concrete methods. A connector inherits them. They are not stages that
you implement.

- `metadata()` reads and validates the dataset's [`dataset.yaml` card](manifest.md) from disk, through
  `DatasetMetadata.from_yaml`. This method does file I/O. Override it only to point to a different
  card. `metadata().dataset_id` must match the connector's curated id.
- `store()` writes the dataset through a [`TimeFWriter`](timef-writer.md). If the schema is absent, it
  derives the schema first. It returns the committed version directory. Most connectors never
  override it.

A connector takes no constructor arguments. Configuration comes from environment variables that
`__init__` reads. `TRaw` is the reference type that the connector defines, for example a path, a small
dataclass, or an S3 key. The connector is generic through PEP 695:
`class MyConnector(BaseConnector[MyRawRef])`.

---

## Sharing data

To share time-series data across samples, attach the same `TimeSeries` instance to each sample. You
can also attach two instances that have the same explicit `time_series_id`. The writer removes
duplicates by `time_series_id`, so it stores the bytes only once. The writer also removes duplicate
annotations, by `id`.

---

## Discovery and layout

The system finds connectors lazily, by dataset id. There is no central registry to maintain. A
concrete connector lives in its own folder, at `datasets/<org>/<name>/` (lowercase Python package
names). The package's `__init__.py` exposes a module-level `CONNECTOR`, and a `dataset.yaml` card sits
beside it. As a result, `timenet-curate build <org>/<name>` imports only that package. Reusable bases
live under `bases/`. Each connector declares its own id in `metadata()`. An id is a lowercase
`org/name` pair.

## Optional dependencies and credentials

A connector can need libraries or credentials that its source requires. Declare heavy libraries as an
**optional extra**. Import them lazily inside the connector, so that base users do not have to install
them. A missing library must raise a clear error. Credentials come from the environment. For the
HuggingFace Hub, the connector reads a token from `HF_TOKEN` automatically. Only gated or private
sources need this token. Downloaded source files cache under `<TIMENET_CACHE>` (see
[client config](client.md#configuration)).

## Downloading artifacts

`timenet_connectors.download` has two async helpers. Each helper picks the backend from the scheme of
the URL. As a result, a connector never has to branch on `s3://` versus `http(s)://` itself:

- `fetch_files([Artifact(url, dest), ...])` downloads a list of files. The list can mix schemes freely.
  HTTP entries run at the same time, up to the limit of `max_concurrency`, and share one connection
  pool. S3 entries run one at a time. A single file uses a one-element list.
- `ensure_archive(url, target)` downloads a zip file and extracts it into `target`. This method is
  idempotent. A marker file records success. As a result, a later run reuses the extracted contents
  and skips the download.

Each `Artifact` takes optional `headers`, `cookies`, and a `sha256` value to validate the download.
`fetch_files` also takes batch-level `headers` and `cookies`. These apply to every HTTP request. The
per-artifact values merge over the batch-level values. S3 ignores all of these. HTTP downloads use
`aiohttp` and `aiofiles`, both base dependencies. They stream to disk and write atomically, through a
`.part` temporary file. If the SHA-256 value does not match, the download raises an error and leaves
nothing behind. HTTP downloads skip an existing destination. S3 downloads use boto3 and stay
synchronous. boto3 already parallelizes the transfer of a single object. `ensure_archive` also takes a
`filename` override, for URLs whose path has no usable name, for example a trailing slash or a
`/download` suffix. Call these helpers from `download_async`:

```python
from timenet_connectors.download import Artifact, ensure_archive, fetch_files

class MyConnector(BaseConnector[MyRawRef]):
    async def download_async(self, cache_dir):
        await fetch_files(
            [
                Artifact("https://host/a.csv", cache_dir / "a.csv"),
                Artifact("s3://bucket/b.csv", cache_dir / "b.csv"),
            ],
            # applied to every HTTP request
            headers={"Authorization": "Bearer …"},
        )
        await ensure_archive("https://host/records.zip", cache_dir)
        return [...]  # lightweight references into cache_dir
```

An ambient sink reports progress, through `timenet_connectors.download.progress`. The download helpers
emit `DownloadProgress` events, and the `timenet-curate` CLI renders these events. As a result,
downloads show progress without a `progress` argument passed through the connector. On a terminal,
the events render as live progress bars, one row per file, that update at the same time. Piped output
falls back to throttled text lines. The `--quiet` flag silences all output.

PhysioNet connectors extend `BasePhysioNetConnector` for WFDB record I/O. They use `ensure_archive` to
pull their database archive.

## Example connectors

- `timenet/hello-world` is a synthetic, offline reference connector. It needs no network and produces
  a fully deterministic dataset, so it also serves as the round-trip fixture. It covers two modalities
  over one shared data source, a series shared across samples, and a windowed sample. It also covers a
  series sized to force a chunk split, and all three annotation shapes, with one shared. Beyond these,
  it covers a `ClassificationTask -> AnswerTask` chain, a scalar prediction, a temporal localization,
  and a scoped classification. Its dataset card, `dataset.yaml`, sits beside it in
  `datasets/timenet/hello_world/`.
- `chengsenwang/tsqa` is a time-series QA dataset. Each row's series becomes a `TimeSeries`, and each
  row's question and answer become an `AnswerTask`. This connector needs the `huggingface` extra
  (`pip install 'timenet-connectors[huggingface]'`), because it downloads data from the Hub.

```bash
timenet-curate build timenet/hello-world             # offline, synthetic
timenet-curate build chengsenwang/tsqa               # live, from the Hub
timenet-curate build chengsenwang/tsqa --keep-cache  # keep the raw sources
```

A successful build removes the dataset's raw download cache, at `<TIMENET_CACHE>/<dataset_id>`. The
build needs the sources only during conversion. To keep the sources, pass `--keep-cache`.

Keeping `download` and `convert` apart makes a connector testable offline. `convert` takes raw
references and does not touch the network. As a result, a test can hand it a checked-in fixture and
skip `download` entirely. See `packages/timenet-connectors/tests/fixtures/` and the `_convert()`
helpers next to them.

After the build, you can load and inspect a dataset with the SDK. See `examples/load_tsqa.py`. This
example loads a dataset and calls `describe()` to print its identity, its counts, its columns per
spec, and a sample preview.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
