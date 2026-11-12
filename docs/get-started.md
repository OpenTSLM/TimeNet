---
icon: lucide/rocket
description: "Install TimeNet and load your first dataset from a local registry."
tags:
  - getting-started
---

# Get started

This guide builds a small offline dataset and loads it through the Python client.

## Install the client

TimeNet needs Python 3.11 or newer. Install the core package, then add only the extras that you use.

=== "uv (recommended)"

    ```bash
    uv add timenet
    uv add 'timenet[cli]'    # add the timenet command
    uv add 'timenet[torch]'  # add the PyTorch adapter
    ```

=== "pip"

    ```bash
    pip install timenet
    pip install 'timenet[cli]'
    pip install 'timenet[torch]'
    ```

The `torch` extra accepts any PyTorch build. Install a CPU or accelerator build that matches your
environment before you add the extra.

## Install the build tool

The `timenet-build` command ships in the separate connectors package. Install it as an isolated tool:

```bash
uv tool install timenet-connectors
```

Use `pipx install timenet-connectors` if you use pipx for command-line tools.

## Build the example dataset

Build the offline `timenet/hello-world` dataset:

```bash
timenet-build build timenet/hello-world
```

The dataset is synthetic, so the connector downloads no data. The build writes this version into a
local registry:

```text
~/.timenet/registry/timenet/hello-world/1.0.0/
├── manifest.json
├── control.duckdb
└── time_series/
```

## Load the dataset

The default client reads the local registry first. An exact version that the local registry holds
needs no network access:

```python
import pandas as pd

import timenet

client = timenet.TimeNet()
dataset = client.load("timenet/hello-world@1.0.0")
dataset.describe()

signal = dataset.records[0].signals[0]
frame = pd.DataFrame({signal.name: signal.to_numpy()})
print(frame.head())
```

`load()` reads the whole dataset with every value in memory. For a dataset too large for that,
`client.open()` returns a reader whose reads leave Signal values in storage until `to_arrow()`,
`to_numpy()`, or `read_steps()` reads them; see [Client](client.md#open-load-or-download).

!!! tip "pandas is optional"
    The DataFrame step needs pandas (`uv add pandas`). Remove that step for an Arrow or NumPy
    workflow.

Continue with these pages:

- [Client](client.md) covers search, version pins, local builds, and PyTorch.
- [Data model](data-model/index.md) explains Records, Sources, Signals, Annotations, and Tasks.
- [Connectors](connectors.md) explains how to add a dataset.
- [Build and publish](build.md) explains local and remote registries.
