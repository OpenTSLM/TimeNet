---
icon: lucide/rocket
description: "Install TimeNet and load your first dataset from a registry."
tags:
  - getting-started
---

# Get started

## Install

TimeNet needs Python 3.11 or newer. The core install stays lean; the CLI and PyTorch loader are
extras you opt into.

=== "uv (recommended)"

    Add it to your project with [uv](https://docs.astral.sh/uv/):

    ```bash
    uv add timenet            # core: TimeF format, reader/writer, registry client
    uv add 'timenet[cli]'     # add the timenet console command
    uv add 'timenet[torch]'   # add load_torch (PyTorch Dataset)
    ```

=== "pip"

    ```bash
    pip install timenet
    pip install 'timenet[cli]'
    pip install 'timenet[torch]'
    ```

=== "Global CLI"

    Install the CLIs anywhere, each in its own isolated environment:

    ```bash
    uv tool install 'timenet[cli]'       # the `timenet` command
    uv tool install timenet-connectors   # `timenet-curate` (for connector authors)
    # or, with pipx:  pipx install 'timenet[cli]'
    ```

To work on TimeNet or author connectors, clone the repo and sync with uv:

```bash
git clone https://github.com/AI-X-Labs/TimeNet.git
cd TimeNet
uv sync --all-groups --all-extras
```

## Load a dataset

!!! note "No public registry yet"
    There's no hosted registry to pull from, so build the offline `timenet/hello-world` dataset
    into a local registry first. It needs no network and comes from `timenet-connectors`.

```bash
timenet-curate build timenet/hello-world --out ./local_registry
```

That output directory is itself a valid registry. Point the [`TimeNet`](client.md) client at it and
load the dataset:

```python
import pandas as pd
from timenet.client import TimeNet

dataset = TimeNet("./local_registry").load("timenet/hello-world")
dataset.describe()                                # identity, counts, a quick preview

# Each channel converts to Arrow or NumPy, so it drops straight into pandas:
series = dataset.samples[0].time_series[0]
df = pd.DataFrame({series.channel: series.to_numpy()})
print(df.head())
```

!!! tip "pandas is optional"
    The DataFrame step uses pandas (`uv add pandas`) — it isn't a TimeNet dependency. Drop it for a
    pure-NumPy workflow.

`load` reads the dataset into a [`TimeFDataset`](timef-dataset.md) with lazy per-series values;
`to_arrow()` / `to_numpy()` on a [`TimeSeries`](timef-dataset.md) pull the values on demand. See
[Client](client.md) for search, version pinning, PyTorch, and the CLI, and
[Connectors](connectors.md) / [Curation](curation.md) to build your own datasets.
