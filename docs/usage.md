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
- `TimeNet().download("org/name")` returns the local TimeF version directory. Its control tables are
  Parquet; its values plane is Parquet or Zarr, as recorded in the manifest.

Example status:

- [x] pandas: load a sample's series into a `DataFrame`
- [x] polars: `pl.from_arrow` over `to_arrow()`
- [x] PyTorch: `load_torch` plus a `DataLoader`
- [ ] Spark: planned

Each series carries its own `channel`, `sampling_rate_hz`, and `t_start_s`, and reads its values
lazily through `to_arrow()` / `to_numpy()`. The framework examples below all start from one loaded
sample.

=== "pandas"

    ```python
    import numpy as np
    import pandas as pd
    from timenet.client import TimeNet

    # download if needed, then read
    dataset = TimeNet().load("chengsenwang/tsqa")
    series = dataset.samples[0].time_series[0]

    values = series.to_numpy()  # shape: (n_steps, *series.spec.value_shape)
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

    !!! planned "Planned"
        No Spark recipe yet. `TimeNet().download("chengsenwang/tsqa")` returns the local version
        directory. Spark can read its Parquet control tables directly. Reading series values depends
        on the manifest's values backend: Parquet values are accessible to Parquet tooling, while
        Zarr values need a Zarr-aware reader.

=== "PyTorch"

    ```python
    from torch.utils.data import DataLoader
    from timenet.client import TimeNet

    # needs: pip install 'timenet[torch-gpu]'
    ds = TimeNet().load_torch("chengsenwang/tsqa")
    item = ds[0]
    series, question = item["series"][0], item["tasks"][0].question

    # Series lengths vary between samples, so batch with a collate_fn that picks
    # out what the model needs.
    loader = DataLoader(
        ds,
        batch_size=8,
        collate_fn=lambda batch: [
            (x["series"][0], x["tasks"][0].target) for x in batch
        ],
    )
    ```

## Example: train a classifier end-to-end

One script, the whole loop: curate a dataset, load it, train a model. The `timenet/test-mean` demo is
deliberately simple. Each sample is one noisy signal, labeled `above_zero` or `below_zero` by whether
its mean is positive, so a classifier only has to recover that sign.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from timenet.client import TimeNet
import timenet_connectors

# Curate the connector's dataset into the local registry (the producer
# side), then load it back.
timenet_connectors.build("timenet/test-mean")
dataset = TimeNet().load("timenet/test-mean")

# Pair each sample's values with its target. Materialization is deferred by
# default (Arrow); ask for output="numpy" since scikit-learn needs it.
x, y = dataset.to_features_and_targets(output="numpy")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, stratify=y, random_state=0
)
model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
print(f"test accuracy: {model.score(x_test, y_test):.3f}")   # -> 1.000
```

`to_features_and_targets` defers materialization: `output="arrow"` (the default) hands back a
`FixedSizeListArray` and a string array with no NumPy copy; the example asks for `output="numpy"` because
scikit-learn needs it. It also takes `features="series"` to return one variable-length sequence per
sample (a `ListArray` / object array) instead of the rectangular `"timestep"` matrix. `task` is inferred
here because `test-mean` has a single task type; pass `task=...` when a dataset carries several.

The full runnable version is
[`examples/test_mean_classifier.py`](https://github.com/OpenTSLM/TimeNet/blob/main/examples/test_mean_classifier.py).

!!! tip "scikit-learn is optional"
    It backs this example only and isn't a TimeNet dependency: `pip install scikit-learn`, then
    `python examples/test_mean_classifier.py`. TimeNet hands you the values as NumPy or Arrow; the
    model on top is your choice.
