---
icon: lucide/plug
description: "Load TimeF Signals into pandas, Polars, PyTorch, or another array library."
tags:
  - usage
  - pandas
  - polars
  - torch
---

# Use a dataset

Every TimeF dataset loads through the same client. Each Record contains a Source hierarchy, and each
Source contains Signals.

```python
from timenet.client import TimeNet

dataset = TimeNet(".timenet-registry").load("timenet/hello-world")
record = dataset.records[0]
signal = record.signals[0]
```

The hierarchy loads first. Signal values load only when an array method reads them.

## Arrow and NumPy

Use `to_arrow()` to preserve Arrow types and nullable values:

```python
arrow_values = signal.to_arrow()
```

Use `to_numpy()` when the Signal has no null values:

```python
numpy_values = signal.to_numpy()
```

For nullable data, use `to_numpy_and_mask()`. The mask is true for present timesteps.

```python
values, present = signal.to_numpy_and_mask()
observed = values[present]
```

Use `read_steps()` to read one half-open range:

```python
first_second = signal.read_steps(0, 500)
```

## pandas

Create a DataFrame from one Signal:

```python
import pandas as pd

frame = pd.DataFrame({signal.name: signal.to_numpy()})
```

For a regular timeline, build offsets from the Signal axis:

```python
offsets_us = [
    signal.time_axis.time_offset_us(index)
    for index in range(signal.n_values)
]
frame.insert(0, "time_offset_us", offsets_us)
```

An ordinal Signal has ordered steps but no time offsets. Do not create timestamps for that axis.

## Polars

Polars accepts the Arrow array directly:

```python
import polars as pl

column = pl.from_arrow(signal.to_arrow())
frame = pl.DataFrame({signal.name: column})
```

## PyTorch

Install the adapter with `uv add 'timenet[torch]'`. Then load a map-style PyTorch dataset:

```python
from timenet.client import TimeNet

torch_dataset = TimeNet(".timenet-registry").load_torch(
    "timenet/hello-world"
)
item = torch_dataset[0]

values = item["series"][0]
present = item["series_masks"][0]
targets = item["tasks"][0].targets
```

The adapter preserves the Signal dtype and per-timestep shape. Each mask has one Boolean value per
timestep.

Tasks and annotations remain Python objects. Signal lengths can also vary. Supply a model-specific
`transform` or `collate_fn` when the default PyTorch batching cannot combine them.

## Train a small classifier

The `timenet/test-mean` connector provides a deterministic classification example. Each Record has
one Signal and one label.

```python
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from timenet.client import TimeNet
import timenet_connectors

registry = Path(".timenet-registry")
timenet_connectors.build("timenet/test-mean", out=registry)
dataset = TimeNet(registry).load("timenet/test-mean")

x, y = dataset.to_features_and_targets(output="numpy")
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    stratify=y,
    random_state=0,
)
model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
print(model.score(x_test, y_test))
```

The helper infers the task type because this dataset has one scalar target type. Pass `task=` when a
dataset contains several eligible task types.

The runnable version is
[`examples/test_mean_classifier.py`](https://github.com/OpenTSLM/TimeNet/blob/main/examples/test_mean_classifier.py).

## Work with downloaded files

`download()` returns a local TimeF version directory:

```python
path = TimeNet().download("org/name@1.0.0")
```

The directory contains `manifest.json`, `control.duckdb`, and a Parquet or Zarr values plane. Use the
TimeNet reader when you need reconstructed Python objects and lazy Signal values.

See [TimeF format](timef-format.md) for direct access to the physical layout.
