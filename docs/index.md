---
icon: lucide/house
description: "TimeNet registers, queries, downloads, converts, and explores time-series datasets in a shared TimeF format."
tags:
  - overview
---

# TimeNet

TimeNet is a Python library and CLI for registering, fetching, and exploring time-series
datasets in a shared format called TimeF. It gives every dataset one on-disk shape and one way
to load it, so a consumer reads ECGs, accelerometer traces, and market series through the same
API.

!!! info "Scope"
    TimeNet is not a modeling toolkit. Training, inference, model definitions, and evaluation
    metrics are out of scope. It stops at handing you the data.

## How it fits together

![TimeNet architecture diagram](assets/architecture.svg)

A [connector](connectors.md) turns a raw source into a TimeF version and publishes it to a
[registry](registry.md). TimeF keeps its control plane in Parquet and stores series values in either
Parquet or Zarr. The [client](client.md) reads the manifest from the registry and loads the data. The
client never runs connector code.

- [`BaseConnector`](connectors.md) is the only contract a new data source must satisfy.
- [`TimeFDataset`](timef-dataset.md) is the in-memory model a connector populates during `convert()`.
- [`TimeFWriter`](timef-writer.md) serializes a populated `TimeFDataset` to disk.
- [`TimeFReader`](timef-reader.md) reads a TimeF version directory back into a `TimeFDataset`.

## Where next

- [Get started](get-started.md): install TimeNet and load your first dataset.
- [Architecture](architecture.md): how the packages, registries, and curation flow fit together.
- [Datasets](catalog/datasets.md): the datasets already curated into the registry.
