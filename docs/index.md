---
icon: lucide/house
description: "TimeNet registers, queries, downloads, converts, and explores time-series datasets in a shared TimeF format."
tags:
  - overview
---

# TimeNet

TimeNet gives time-series datasets one format and one Python API. You can build a raw source into
TimeF, publish the result to a registry, and load it without dataset-specific code. The same objects
represent ECGs, accelerometer traces, market series, and other sequential data.

!!! info "Scope"
    TimeNet is not a modeling toolkit. Training, inference, model definitions, and evaluation
    metrics are out of scope. TimeNet only gives you the data.

## How it fits together

![TimeNet architecture diagram](assets/architecture.svg)

A [connector](connectors.md) turns a raw source into a TimeF version. A
[registry](registry.md) stores immutable versions. The [client](client.md) searches the registry and
loads datasets with lazy Signal values. Consumer code never runs a connector.

- [`BaseConnector`](connectors.md) is the only contract a new data source must satisfy.
- [`TimeFDataset`](timef-dataset.md) is the in-memory model a connector populates during `convert()`.
- [`TimeFWriter`](timef-writer.md) writes a populated dataset as one immutable version.
- [`TimeFReader`](timef-reader.md) restores the version without loading every Signal value.

## Where next

- [Get started](get-started.md): install TimeNet and load your first dataset.
- [Architecture](architecture.md): how the packages, registries, and build flow fit together.
- [Datasets](catalog/datasets.md): the datasets already built into the registry.
