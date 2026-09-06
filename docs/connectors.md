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

## Adding a new dataset

Do not start from a blank `connector.py`. The repository ships an agent skill,
`add-dataset-connector`, under `.agents/skills/`. It takes a link to a source and gives back a built
connector, and it encodes decisions that earlier connectors had to find the hard way. Invoke it by
name, or with `/add-dataset-connector`.

It runs six phases with one gate, and its `SKILL.md` is the authority on what each does. What matters
before you start is that **two of those steps are yours, not the agent's**:

- **You confirm the dataset card**, before any design begins. `license` is a legal claim about
  somebody else's data, not a field to infer from a web page.
- **You rule at the gate.** The skill stops before writing any connector code and puts each open
  assumption to you one at a time. An assumption is a question the source does not answer, and the
  answer belongs to a person.

Everything the two of you decide ends up in a `README.md` beside the connector, one entry per
assumption and per inconsistency the release ships, each with its evidence, the decision, and a state
of **Handled**, **Not built**, or **Open**. That file is what tells somebody a year later why the
data looks the way it does.

Two things about the method are worth knowing even if you never run the skill:

- **A head is not a `cat`.** Most sources are binary — EDF, WFDB, parquet, `.xls`, HDF5 — so each
  connector ships a `head()` per raw file type, and those are how anybody, human or agent, looks at
  the raw source. Run them against the next release of a dataset and a changed shape shows at once.
- **A head cannot find what is odd.** Odd is a fact about the set, not about any one file: one
  recording in a hundred with a different record length, a scaling factor that varies per file where
  you assumed a constant, a handful of table rows disagreeing with their headers. Only a census over
  every file finds those, and the design has to answer for each one.

The skill's references carry the rules it applies, and they are worth reading on their own:
`fidelity.md` (what a connector may and may not do to its source), `layout.md` (how the modules
divide), `discovery.md` (heads, censuses, the map), and `connector-anatomy.md` (the contract, the
bases, the task types).

A finished connector is reviewed with the second skill, `review-connector-stack`, which checks it
against these same rules before it merges — and reports what it could not check rather than passing
it in silence.

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
  card. `metadata().dataset_id` must match the connector's built id.
- `store()` writes the dataset through a [`TimeFWriter`](timef-writer.md). If the schema is absent, it
  derives the schema first. It returns the committed version directory. Most connectors never
  override it.

A connector takes no constructor arguments. Where a connector needs configuration, it reads it from
the environment rather than from a parameter; `__init__` itself only checks that the subclass
implements `download` or `download_async`. `TRaw` is the reference type that the connector defines,
for example a path, a small dataclass, or an S3 key. The connector is generic through PEP 695:
`class MyConnector(BaseConnector[MyRawRef])`.

---

## Sharing data

To share time-series data across records, attach the same `TimeSeries` instance to each record. You
can also attach two instances that have the same explicit `time_series_id`. The writer removes
duplicates by `time_series_id`, so it stores the bytes only once. The writer also removes duplicate
annotations, by `id`.

---

## Discovery and layout

The system finds connectors lazily, by dataset id. There is no central registry to maintain. A
concrete connector lives in its own folder, at `datasets/<org>/<name>/` (lowercase Python package
names). The package's `__init__.py` exposes a module-level `CONNECTOR`, and a `dataset.yaml` card sits
beside it, next to a `requirements.txt` when the connector needs libraries of its own. As a result,
`timenet-build build <org>/<name>` imports only that package. Reusable bases live under `bases/`.
Each connector declares its own id in `metadata()`. An id is a lowercase `org/name` pair.

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

Import those libraries lazily, inside the function that uses them, and raise a clear error when one
is missing. That guard keeps `--no-isolation` usable while you write a connector; a top-level import
breaks it. The one exception is a shared base that only its declaring connector imports, such as
`bases/edf/reader.py` or `bases/excel.py`: nothing else reaches it, so there is nobody to protect,
and it imports at the top of the module.

**Declare a requirement twice.** Once in the connector's `requirements.txt`, which the build
installs into the environment the build runs in, and once in the `dev` group of the root
`pyproject.toml`, which is what puts it in your own environment. `make sync` is
`uv sync --all-groups --all-extras`, and a `requirements.txt` is neither a group nor an extra, so
syncing alone will not install it.
Without the second declaration, `ty` reports the lazy import as unresolved and the connector's own
tests cannot run.

List every dependency the connector needs, even one that another connector already names. There are
no shared requirement fragments. If several connectors share one file, an edit to that file can break
a connector that you did not check.

Credentials come from the environment. For the HuggingFace Hub, a token is read from `HF_TOKEN`
automatically (needed only for gated or private sources). Downloaded source files cache under
`<TIMENET_CACHE>` (see [client config](client.md#configuration)).

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
  a fully deterministic dataset, so it also serves as the round-trip fixture. It covers two modalities
  over one shared data source, a series shared across records, and a windowed record. It also covers a
  series sized to force a chunk split, and all three annotation shapes, with one shared. Beyond these,
  it covers a `ClassificationTask -> AnswerTask` chain, a scalar prediction, a temporal localization,
  and a scoped classification. Its dataset card, `dataset.yaml`, sits beside it in
  `datasets/timenet/hello_world/`.
- `chengsenwang/tsqa` is a time-series QA dataset. Each row's series becomes a `TimeSeries`, and each
  row's question and answer become an `AnswerTask`. It downloads data from the Hub, so its
  `requirements.txt` names `huggingface_hub`. It is the worked example of a connector that fits in
  one module.
- `physionet/ecg-qa-cot` pairs PTB-XL 12-lead ECGs with chain-of-thought question answering. One
  record is one recording, and its many questions stream as `AnswerTask`s rather than being held in
  memory.
- `physionet/sleep-edfx` is whole-night polysomnography with expert sleep scoring. It is the worked
  example of the divided shape: `connector.py` orchestrates, and `tables.py`, `metadata.py`,
  `annotations.py`, `tasks.py`, `specs.py` and `keys.py` each hold one job. Its `README.md` beside
  the connector is the worked example of a connector README, with an entry for each of the six
  inconsistencies the release ships.

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
references and does not touch the network, so a test can hand it a fixture and skip `download`
entirely.

**A test writes its own fixture. The repository ships no dataset bytes.** There is no `fixtures/`
directory under `datasets/`. `chengsenwang/tsqa` holds hand-written rows shaped like Hub rows;
`physionet/sleep_edfx` writes a synthetic two-recording release into `tmp_path`. A comment above
each fixture says it is invented and how a reader can tell.

Split the connector so that most of its tests need no fixture at all. A module that turns rows into
facts, or facts into annotations, takes values and gives values, so its test passes literals and
imports neither the source library nor a temporary file. Only the module that opens the file needs
one.

After the build, you can load and inspect a dataset with the SDK. See `examples/load_tsqa.py`. This
example loads a dataset and calls `describe()` to print its identity, its counts, its columns per
spec, and a record preview.

---

See the [API reference for `timenet.connectors`](api/connectors.md) for the full symbol listing.
