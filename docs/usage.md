---
icon: lucide/plug
description: "Load a TimeNet dataset into pandas, polars, Spark, or PyTorch."
tags:
  - usage
  - pandas
  - polars
  - spark
  - torch
---

# Usage

Every dataset loads the same way, then hands off to your framework of choice. Two entry points:

- `TimeNet().load("org/name")` returns an in-memory [`TimeFDataset`](timef-dataset.md) with lazy
  per-series values. Use it for single-node work (pandas, polars, torch).
- `TimeNet().download("org/name")` returns the local version directory of parquet files. Use it for
  Spark and other engines that read parquet directly.

Recipe status:

- [x] pandas: load a sample's series into a `DataFrame`
- [x] polars: `pl.from_arrow` over `to_arrow()`
- [x] PyTorch: `load_torch` plus a `DataLoader`
- [ ] Spark: planned

Each series carries its own `channel`, `sampling_rate_hz`, and `t_start_s`, and reads its values
lazily through `to_arrow()` / `to_numpy()`. The framework recipes below all start from one loaded
sample.

=== "pandas"

    ```python
    import numpy as np
    import pandas as pd
    from timenet.client import TimeNet

    dataset = TimeNet().load("chengsenwang/tsqa")   # download if needed, then read
    series = dataset.samples[0].time_series[0]

    values = series.to_numpy()                       # 1-D np.ndarray of channel values
    t_s = series.t_start_s + np.arange(len(values)) / series.sampling_rate_hz
    frame = pd.DataFrame({"t_s": t_s, series.channel: values})
    ```

=== "polars"

    ```python
    import polars as pl
    from timenet.client import TimeNet

    dataset = TimeNet().load("chengsenwang/tsqa")
    series = dataset.samples[0].time_series[0]

    # pl.from_arrow reads the Arrow array into a polars Series without a copy.
    column = pl.from_arrow(series.to_arrow())
    frame = pl.DataFrame({series.channel: column})
    ```

=== "Spark"

    !!! warning "Planned"
        No Spark recipe yet. `TimeNet().download("chengsenwang/tsqa")` returns the local version
        directory of parquet files, which `spark.read.parquet` can point at directly, but the
        documented recipe lands later.

=== "PyTorch"

    ```python
    from torch.utils.data import DataLoader
    from timenet.client import TimeNet

    ds = TimeNet().load_torch("chengsenwang/tsqa")   # pip install 'timenet[torch]'
    item = ds[0]
    series, question = item["series"][0], item["tasks"][0].question

    # Series lengths vary between samples, so batch with a collate_fn that picks
    # out what the model needs.
    loader = DataLoader(
        ds,
        batch_size=8,
        collate_fn=lambda batch: [(x["series"][0], x["tasks"][0].answer) for x in batch],
    )
    ```
