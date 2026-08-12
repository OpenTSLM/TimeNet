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
    uv add timenet                # core: TimeF format, reader/writer, registry
    uv add 'timenet[cli]'         # add the timenet console command
    uv add 'timenet[torch]'       # load_torch; reuses your torch, or pulls the default build
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
    uv tool install timenet-connectors   # `timenet-curate` (connector authors)
    # or, with pipx:  pipx install 'timenet[cli]'
    ```

The `torch` extra accepts any torch build. If you already have a CUDA torch (say, for training), it
is reused as-is. For a small CPU-only torch, install it from the PyTorch CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To work on TimeNet or author connectors, clone the repo and sync with uv:

```bash
git clone https://github.com/OpenTSLM/TimeNet.git
cd TimeNet
uv sync --all-groups --all-extras
```

## Load a dataset

!!! info "No public registry yet"
    There's no hosted registry to pull from, so build the offline `timenet/hello-world` dataset
    into a local registry first. It needs no network and comes from `timenet-connectors`.

```bash
timenet-curate build timenet/hello-world
```

That writes into your local registry, which is where the [`TimeNet`](client.md) client looks by
default. Load the dataset:

```python
import pandas as pd
from timenet.client import TimeNet

dataset = TimeNet().load("timenet/hello-world")
dataset.describe()  # identity, counts, a quick preview

# Each channel converts to Arrow or NumPy, so it drops straight into pandas:
series = dataset.samples[0].time_series[0]
df = pd.DataFrame({series.channel: series.to_numpy()})
print(df.head())
```

!!! tip "pandas is optional"
    The DataFrame step uses pandas (`uv add pandas`). It isn't a TimeNet dependency. Drop it for a
    pure-NumPy workflow.

`load` reads the dataset into a [`TimeFDataset`](timef-dataset.md) with lazy per-series values;
`to_arrow()` / `to_numpy()` on a [`TimeSeries`](timef-dataset.md) pull the values on demand. See
[Client](client.md) for search, version pinning, PyTorch, and the CLI, and
[Connectors](connectors.md) / [Curation](curation.md) to build your own datasets.
