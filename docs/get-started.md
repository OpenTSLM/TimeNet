---
icon: lucide/rocket
description: "Install TimeNet and read your first dataset."
tags:
  - getting-started
---

# Get started

## Install

TimeNet needs Python 3.11 or newer. The core install stays small. The PyTorch loader, the Zarr
values backend, and the S3 registry are extras.

=== "uv (recommended)"

    Add TimeNet to your project with [uv](https://docs.astral.sh/uv/):

    ```bash
    uv add timenet              # core: TimeF format, reader/writer, registry
    uv add 'timenet[torch]'     # timenet.torch; reuses your torch build
    uv add 'timenet[zarr]'      # read and write a Zarr values plane
    uv add 'timenet[s3]'        # an s3:// registry
    ```

=== "pip"

    ```bash
    pip install timenet
    pip install 'timenet[torch]'
    pip install 'timenet[zarr]'
    pip install 'timenet[s3]'
    ```

The `torch` extra accepts any torch build. If you already have a CUDA torch (for training, say) you
keep it as-is. For a small CPU-only torch, install it from the PyTorch CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To work on TimeNet itself, clone the repo and sync with uv:

```bash
git clone https://github.com/OpenTSLM/TimeNet.git
cd TimeNet
make sync  # install the dev environment (workspace + extras)
```

## Write a dataset

There is no hosted registry yet, so start by writing one. A dataset is a
[`DeclarativeDataset`](timef-writer.md): records holding sources, sources holding signals.

```python
from fractions import Fraction

import numpy as np

from timenet.control_plane import (
    DeclarativeDataset, Record, Signal, Source, TimeFWriter,
)
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Access, DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg,
)

spec = TimeSeriesSpec(
    spec_type="ecg-voltage",
    name="ECG voltage",
    unit_value=ureg.Unit("mV"),
    dtype="float32",
)
metadata = DatasetMetadata(
    dataset_id="demo/hello-world",
    dataset_version=Version.parse("1.0.0"),
    name="Hello world",
    description="One record, one lead.",
    license=License.MIT,
    domains=(Domain.CARDIOLOGY,),
    access=Access.OPEN,
)

dataset = DeclarativeDataset(metadata=metadata)
signal = Signal(
    id="record-000-lead-i",
    name="I",
    values=np.sin(np.arange(5_000, dtype=np.float32) / 50),
    time_axis=RegularAxis(period_us=Fraction(1_000_000, 500)),
    spec=spec,
)
monitor = Source(id="monitor", name="Bedside monitor", signals=[signal])
dataset.add_record(Record(id="record-000", sources=[monitor]))

with TimeFWriter("./local_registry", metadata) as writer:
    writer.write(dataset)
```

The output directory is itself a valid local registry, committed at
`./local_registry/demo/hello-world/1.0.0/`.

## Read it back

```python
from timenet.client import TimeNet

with TimeNet("./local_registry").open("demo/hello-world") as reader:
    record = reader.record("record-000")
    signal = record.signals()[0]
    values = reader.values(signal.signal_id)
    print(signal.name, values.dtype, values.shape)
```

`reader.values()` reads one signal. For a whole corpus, walk it in batches with
`reader.iter_records()` and read each batch's values in one call to `reader.values_for()`. That is
what the [pandas and PyTorch views](usage.md) do.

!!! tip "pandas and torch are optional"
    Neither is a TimeNet dependency. `timenet.pandas` imports pandas inside the function that builds
    a frame, and `timenet.torch` imports torch on first use, so a NumPy-only workflow needs neither.

Next: [the client](client.md) for search and version pinning, [TimeFReader](timef-reader.md) for the
full read surface, and [the data model](data-model/index.md) for what a record can hold.
