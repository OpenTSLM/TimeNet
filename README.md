# TimeNet

*Download and explore time-series datasets through one standardized format.*

> [!NOTE]
> This is a pre-release version and is subject to change. We are actively working on
> improvements around performance and integrations, and welcome community contributions.

[![PyPI](https://img.shields.io/pypi/v/timenet)](https://pypi.org/project/timenet/)
[![Docs](https://img.shields.io/badge/docs-docs.timenet.ai-1f6feb)](https://docs.timenet.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OpenTSLM/TimeNet/blob/main/LICENSE)

Time-series data is fragmented. TimeNet standardizes it. Every dataset used to ship in its own
shape, forcing teams to rewrite the same loading code again and again. TimeF replaces that with
one shared format and one set of tools to find, download, and load any dataset the same way,
whether it holds ECGs, accelerometer traces, or market prices.

TimeNet hands you the data and stops there. Training, inference, and modeling are up to you.

We're actively growing TimeNet: adding datasets, integrating time-series ML models, and building
connectors to data processing libraries. Contributions in any of these areas are welcome.

Full documentation: <https://docs.timenet.ai/>

## How it fits together

![TimeNet architecture diagram](https://raw.githubusercontent.com/OpenTSLM/TimeNet/main/docs/assets/architecture.svg)

A connector turns a raw source into a manifest plus parquet and publishes it to a registry. The
client reads the manifest from the registry and loads the data. Reading never runs connector code,
so everything a consumer needs to interpret the parquet lives in the manifest.

- `BaseConnector` is the only contract a new data source must satisfy.
- `TimeFDataset` is the in-memory model a connector populates during `convert()`.
- `TimeFWriter` serializes a populated `TimeFDataset` to disk.
- `TimeFReader` reads a TimeF version directory back into a `TimeFDataset`.

## Components

The project is a [uv](https://docs.astral.sh/uv/) workspace with two packages under `packages/`,
plus the registry they read from and write to.

| Part | What it is | Ships |
| --- | --- | --- |
| `timenet` | the SDK and CLI | the TimeF format, reader/writer, registry client, engine, `BaseConnector` |
| `timenet-connectors` | the producer package | connector recipes, dataset cards, and the `timenet-build` CLI |
| registry | a served location | compiled manifests plus parquet; can be public, a private internal one, or a local directory |

See the [architecture guide](https://docs.timenet.ai/architecture.html) for the full map, and the
[concepts page](https://docs.timenet.ai/concepts.html) for the terminology.

## Install

Requires Python 3.11 or newer (tested on 3.11 to 3.13).

```bash
uv add timenet            # core: TimeF format, reader/writer, registry client
uv add 'timenet[cli]'     # add the timenet console command
uv add 'timenet[torch]'   # add load_torch (PyTorch Dataset); works with any torch build
```

Once installed, the CLI is available as `timenet`. See [Get started](https://docs.timenet.ai/get-started.html)
to load your first dataset.

## License

TimeNet is released under the [MIT License](https://github.com/OpenTSLM/TimeNet/blob/main/LICENSE).

### Dataset licenses

The MIT License covers TimeNet's own code, not the datasets it fetches. Each dataset keeps its
upstream license. Check the `license` and `source_url` fields on a dataset's card to see what applies
and where the data comes from. Some sources, such as PhysioNet, only grant credentialed access, so
follow their terms when you download. See
[Dataset licensing](https://docs.timenet.ai/catalog/licensing/) for the full note.
