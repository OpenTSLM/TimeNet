---
icon: lucide/shapes
description: "The shape of a TimeNet dataset: how it is ingested, and the samples, annotations, and tasks it holds."
tags:
  - guide
  - concepts
---

# Data model

TimeNet is a standardization layer. It registers, downloads, converts, and serves different
time-series datasets in one on-disk format ([TimeF](../timef-dataset.md)). Then it gives you the data.
TimeNet is not a training toolkit. These pages show the shape of that data:

- **[Datasets](datasets.md)**: versioned collections addressed by an `org/name` id.
- **[Samples](samples.md)**: one recording with its signals, annotations, and tasks.
- **[Time series](time-series.md)**: one channel's values over time. A sample has one or more.
- **[Annotations](annotations.md)**: scoped side-information in three shapes.
- **[Tasks](tasks.md)**: the labeled training targets built from a sample.

The code examples follow one running example: a machine's vibration and temperature channels. But the
same primitives describe any sensor stream, from an ECG to a market series.

## Ingesting a dataset

To onboard a dataset, you write one [`BaseConnector`](../connectors.md). The engine drives it
through a fixed pipeline. `download` fetches raw files (I/O only). `convert` parses them into an
in-memory dataset (CPU only). The engine then derives the schema from the data. It stores the result
as parquet plus a `manifest.json`. The whole surface is frozen dataclasses. So datasets round-trip
deterministically, and reading a compiled version never runs connector code.

```mermaid
flowchart LR
    D["download()<br/><i>I/O · fetch raw refs</i>"]
    C["convert()<br/><i>CPU only</i>"]
    S["derive_schema()<br/><i>types from data</i>"]
    W["store()<br/><i>TimeFWriter</i>"]
    R[("registry<br/>parquet + manifest.json")]
    D --> C --> S --> W --> R
```
