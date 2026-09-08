---
icon: lucide/rocket
description: "Install TimeNet and load your first dataset from a registry."
tags:
  - getting-started
---

# Get started

## Install

TimeNet needs Python 3.11 or newer. The core install stays small. The CLI and the PyTorch loader
are extras. You can add them.

=== "uv (recommended)"

    Add TimeNet to your project with [uv](https://docs.astral.sh/uv/):

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

    Install the CLIs anywhere. Each CLI gets its own isolated environment:

    ```bash
    uv tool install 'timenet[cli]'       # the `timenet` command
    uv tool install timenet-connectors   # `timenet-build` (connector authors)
    # or, with pipx:  pipx install 'timenet[cli]'
    ```

The `torch` extra accepts any torch build. If you already have a CUDA torch (for example, for
training), you keep it as-is. For a small CPU-only torch, install it from the PyTorch CPU index
first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To work on TimeNet or author connectors, clone the repo and sync with uv:

```bash
git clone https://github.com/OpenTSLM/TimeNet.git
cd TimeNet
make sync  # install the dev environment (workspace + extras)
```

## Load a dataset

!!! info "No public registry yet"
    There is no hosted registry yet. First build the offline `timenet/hello-world` dataset into
    a local registry. The build needs no network. The dataset comes from `timenet-connectors`.

```bash
timenet-build build timenet/hello-world
```

The build writes into your local registry. The [`TimeNet`](client.md) client looks there by
default. Now load the dataset:

```python
import pandas as pd
from timenet.client import TimeNet

dataset = TimeNet().load("timenet/hello-world")
dataset.describe()  # identity, counts, a quick preview

# Each signal converts to Arrow or NumPy, so it drops straight into pandas:
series = dataset.records[0].time_series[0]
df = pd.DataFrame({series.signal: series.to_numpy()})
print(df.head())
```

!!! tip "pandas is optional"
    The DataFrame step uses pandas (`uv add pandas`). pandas is not a TimeNet dependency. For a
    pure-NumPy workflow, remove it.

`load` reads the dataset into a [`TimeFDataset`](timef-dataset.md) with lazy per-series values.
`to_arrow()` and `to_numpy()` on a [`TimeSeries`](timef-dataset.md) pull the values on demand. See
[Client](client.md) for search, version pinning, PyTorch, and the CLI. See
[Connectors](connectors.md) and [Build](build.md) to build your own datasets.
