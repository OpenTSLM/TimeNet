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

!!! note "Worked examples in progress"
    The per-framework recipes below are placeholders for now. They land once the loader API is pinned
    down.

=== "pandas"

    ```python
    # pandas example — coming soon
    ```

=== "polars"

    ```python
    # polars example — coming soon
    ```

=== "Spark"

    ```python
    # Spark example — coming soon
    ```

=== "PyTorch"

    ```python
    # PyTorch example — coming soon
    ```
