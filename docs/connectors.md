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

`BaseConnector` is the connector contract in the `timenet` package. Concrete connectors ship in
`timenet-connectors`, beside their [dataset cards](manifest.md).

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
| `convert(raw_refs)` | local work | Parse the references into a `TimeFDataset`. Do not use the network. |

You must implement `convert`. It is the only required method. `download` depends on I/O, so it has
two forms. You can override `download(cache_dir)` for a synchronous fetch. Alternatively, you can
define `async download_async(cache_dir)` to fetch artifacts at the same time. Implement only one of
these two forms. The engine always calls the synchronous `download()`. Its default implementation
runs `download_async` to completion. As a result, an async connector needs no event loop code of its
own.

`metadata()` is a concrete method that reads the dataset card. The engine owns storage and calls the
writer after conversion.

- `metadata()` reads and validates the dataset's [`dataset.yaml` card](manifest.md) from disk, through
  `DatasetMetadata.from_yaml`. This method does file I/O. Override it only to point to a different
  card. `metadata().dataset_id` must match the connector's built id.

A connector takes no constructor arguments. Configuration comes from environment variables that
`__init__` reads. `TRaw` is the reference type that the connector defines, for example a path, a small
dataclass, or an S3 key. The connector is generic through PEP 695:
`class MyConnector(BaseConnector[MyRawRef])`.

---

## Reusing descriptors

Each Signal has exactly one owning Source. Do not attach one Signal to several Sources or Records.
TimeNet rejects that ambiguous ownership.

Signals can share immutable `TimeAxis` and `TimeSeriesSpec` objects. Annotation content can also be
reused. Each call to `annotate()` creates a separate occurrence for the selected owner.

---

## Discovery and layout

The system finds connectors lazily, by dataset id. There is no central registry to maintain. A
concrete connector lives under `timenet_connectors/datasets/<org>/<name>/`. Use lowercase Python
package names. The package's `__init__.py` exposes `CONNECTOR`. Put `dataset.yaml` and an optional
`requirements.txt` beside it. The build tool imports only the requested connector package. Reusable
bases live under `timenet_connectors/bases/`.

## Dependencies and credentials

A connector declares the libraries its source needs in a `requirements.txt` beside its
`dataset.yaml`. It is a plain pip requirements file:

```text
# Downloads the source dataset from the HuggingFace Hub.
huggingface_hub>=0.24
```

A build runs the connector in an environment built from that file. That environment is layered over
the same `timenet` and `timenet-connectors` that you run (see [Build & publish](build.md)).
A build installs nothing into your own environment. Two connectors that need incompatible libraries
do not collide.

Import those libraries lazily inside the connector anyway. If one is missing, raise a clear error.
That guard keeps `--no-isolation` usable while you write a connector.

List every dependency the connector needs, even one that another connector already names. There are
no shared requirement fragments. If several connectors share one file, an edit to that file can break
a connector that you did not check.

Credentials come from the environment. For the HuggingFace Hub, a token is read from `HF_TOKEN`
automatically (needed only for gated or private sources). Downloaded source files cache under
`<TIMENET_CACHE>` (see [client local state](client.md#local-state)).

A credentialed dataset (a PhysioNet DUA-gated one, for example) declares `access: credentialed`
and an `access_url` on its card. TimeNet never hosts such data, so it is build-your-own: get access
at the `access_url`, set the provider credential in the environment (the way `HF_TOKEN` already
works), and run `timenet-build build <id>` yourself. A `load` or `download` of a credentialed
dataset from a hosted registry raises with the `access_url` instead of serving bytes.

## Downloading artifacts

`timenet_connectors.download` has two async helpers. Each helper picks the backend from the scheme of
the URL. As a result, a connector never has to branch on `s3://` versus `http(s)://` itself:

- `download_files([Artifact(url, dest), ...])` downloads a list of files. The list can mix schemes freely.
  HTTP entries run at the same time, up to the limit of `max_concurrency`, and share one connection
  pool. S3 entries run one at a time. A single file uses a one-element list.
- `ensure_archive(url, target)` downloads a zip file and extracts it into `target`. This method is
  idempotent. A marker file records success. As a result, a later run reuses the extracted contents
  and skips the download. A successful extraction then deletes the archive.

Each `Artifact` takes optional `headers`, `cookies`, and a `sha256` value to validate the download.
`download_files` also takes batch-level `headers` and `cookies`. These apply to every HTTP request. The
per-artifact values merge over the batch-level values. S3 ignores all of these. HTTP downloads use
`httpx` and `aiofiles`, both base dependencies. They stream to disk and write atomically, through a
`.part` temporary file. If the SHA-256 value does not match, the download raises an error and leaves
nothing behind. HTTP downloads skip an existing destination. S3 downloads use boto3 and stay
synchronous. boto3 already parallelizes the transfer of a single object. `ensure_archive` also takes a
`filename` override, for URLs whose path has no usable name, for example a trailing slash or a
`/download` suffix. Call these helpers from `download_async`:

```python
from timenet_connectors.download import Artifact, ensure_archive, download_files

class MyConnector(BaseConnector[MyRawRef]):
    async def download_async(self, cache_dir):
        await download_files(
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
emit `DownloadProgress` events, and the `timenet-build` CLI renders these events. As a result,
downloads show progress without a `progress` argument passed through the connector. On a terminal,
the events render as live progress bars, one row per file, that update at the same time. Piped output
falls back to throttled text lines. The `--quiet` flag silences all output.

PhysioNet connectors extend `BasePhysioNetConnector` for WFDB record I/O. They use `ensure_archive` to
pull their database archive.

## Example connectors

- `timenet/hello-world` is a synthetic, offline reference connector. It needs no network and produces
  a deterministic dataset for examples and round-trip tests. It covers nested Sources, shared
  descriptors, windowed Signals, annotations, and several task types.
- `chengsenwang/tsqa` is a time-series QA dataset. Each row's series becomes a Signal, and each
  row's question and answer become an `AnswerTask`. It downloads data from the Hub, so its
  `requirements.txt` names `huggingface_hub`.

```bash
timenet-build build timenet/hello-world             # offline, synthetic
timenet-build build chengsenwang/tsqa               # live, from the Hub
timenet-build build chengsenwang/tsqa --keep-cache  # keep the raw sources
```

A successful build removes the dataset's raw download cache, at `<TIMENET_CACHE>/<dataset_id>`. Only
`convert` reads the sources, and they are often several times the size of the dataset they produce.
To keep them, pass `--keep-cache`, or `keep_cache=True` to `timenet_connectors.build()`. This helps
while you write a connector for a large source, because each rebuild downloads the source again.

Keeping `download` and `convert` apart makes a connector testable offline. `convert` takes raw
references and does not touch the network. A test can therefore pass small local inputs or synthetic
references directly to `convert()` and skip `download`.

After the build, you can load and inspect a dataset with the SDK. See `examples/load_tsqa.py`. This
example loads a dataset and calls `describe()` to print its identity, its counts, its columns per
spec, and a record preview.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
