---
icon: lucide/house
description: "TimeNet searches, downloads, and loads time-series datasets in a shared TimeF format."
tags:
  - overview
---

# TimeNet

TimeNet is a shared data format and a set of tools for time-series datasets. It gives
every dataset one on-disk shape, one Python API to read it, and one registry to find it in.

TimeNet keeps the recording and the task separate, so a record can gain a new task later without
a new copy of the dataset.

TimeNet's ambition: a shared foundation every time-series
foundation model trains on.

TimeNet gives you:

- One format across domains. ECGs, accelerometer traces, and market series read through the same
  API.
- A registry to find, version, and load datasets, instead of a folder of scripts per source.
- Loaders that plug into pandas and PyTorch.
- Records kept separate from their tasks and annotations, so a new task never means a new copy
  of the dataset.

TimeNet is the Python library and CLI you use for all of this.

!!! info "Scope"
    TimeNet is not a modeling toolkit. Training, inference, model definitions, and evaluation
    metrics are out of scope. TimeNet works on the data layer

## How it fits together

![TimeNet architecture diagram](assets/architecture.svg)

A [connector](connectors.md) turns a raw source into a TimeF version and publishes it to a
[registry](registry.md). TimeF keeps its control plane in Parquet and stores series values in either
Parquet or Zarr. The [client](client.md) reads the manifest from the registry and loads the data.
Against a remote registry, reading never runs connector code. Against a local registry, `load` can
first build a dataset that the registry does not have from an installed connector (see
[Build & publish](build.md)).

- [`BaseConnector`](connectors.md) is the only contract a new data source must satisfy.
- [`TimeFDataset`](timef-dataset.md) is the in-memory model a connector populates during `convert()`.
- [`TimeFWriter`](timef-writer.md) serializes a populated `TimeFDataset` to disk.
- [`TimeFReader`](timef-reader.md) reads a TimeF version directory back into a `TimeFDataset`.

## Where next

- [Get started](get-started.md): install TimeNet and load your first dataset.
- [Architecture](architecture.md): how the packages, registries, and build flow fit together.
- [Datasets](catalog/datasets.md): the datasets already built into the registry.
