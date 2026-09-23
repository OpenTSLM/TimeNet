# TimeNet

*Find, build, and load time-series datasets through one shared format.*

> [!NOTE]
> TimeNet is a pre-release project. Its public API and the TimeF format can change before 1.0.

[![PyPI](https://img.shields.io/pypi/v/timenet)](https://pypi.org/project/timenet/)
[![Docs](https://img.shields.io/badge/docs-docs.timenet.ai-1f6feb)](https://docs.timenet.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OpenTSLM/TimeNet/blob/main/LICENSE)

Time-series datasets use many incompatible layouts. TimeF gives them one storage format. TimeNet
provides one API to find, build, download, and load them.

TimeNet handles the data layer. Model training, inference, and evaluation stay in your chosen
framework.

## Quick start

TimeNet needs Python 3.11 or newer.

```bash
uv add 'timenet[cli]'
uv tool install timenet-connectors
timenet-build build timenet/hello-world --out .timenet-registry
```

Load the local build with Python:

```python
from timenet.client import TimeNet

dataset = TimeNet(".timenet-registry").load("timenet/hello-world")
dataset.describe()

signal = dataset.records[0].signals[0]
values = signal.to_numpy()
```

This example uses a local registry, so it works without registry credentials or network access.

See the [Get started guide](https://docs.timenet.ai/get-started.html) for installation options and a
longer example.

## How it fits together

![TimeNet architecture diagram](https://raw.githubusercontent.com/OpenTSLM/TimeNet/main/docs/assets/architecture.svg)

A connector converts a raw source into a TimeF version. The version contains:

- `manifest.json`, which identifies the dataset and lists its files
- `control.duckdb`, which stores the hierarchy and relationships
- a Parquet or Zarr values plane, which stores Signal values

A registry stores immutable TimeF versions. The client reads a version without importing its
connector.

The repository is a [uv](https://docs.astral.sh/uv/) workspace with two packages:

| Package | Purpose |
| --- | --- |
| `timenet` | The TimeF model, reader, writer, registries, SDK, and consumer CLI. |
| `timenet-connectors` | Dataset connectors and the `timenet-build` producer CLI. |

Read the [architecture guide](https://docs.timenet.ai/architecture.html) for the complete design.

## License

TimeNet uses the [MIT License](https://github.com/OpenTSLM/TimeNet/blob/main/LICENSE).

This license covers the TimeNet code, not the datasets that connectors fetch. Each dataset keeps
its upstream license and access terms. Read the
[dataset licensing guide](https://docs.timenet.ai/catalog/licensing.html) before redistribution.
