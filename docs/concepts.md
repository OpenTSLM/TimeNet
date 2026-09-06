---
icon: lucide/book-open
description: "Key TimeNet terminology: TimeF, dataset, record, connector, build, registry, and more."
tags:
  - guide
  - concepts
---

# Concepts

This page collects the vocabulary that appears across these docs. Follow a link for the full detail.

| Term | What it is |
| --- | --- |
| **TimeF** | The one format for every dataset. It has one on-disk shape and one API, so an ECG, an accelerometer trace, and a market series all read the same way. |
| **Dataset** | A named, versioned collection of records in TimeF, addressed as `org/name` (for example `chengsenwang/tsqa`). |
| **Record** | One recording in a dataset (for example, a single patient trace): its [time series](timef-dataset.md), tasks, and annotations. |
| **Time series** | One logical stream within a record, with shape `(n_steps, *value_shape)` and a dtype declared by its spec. You read values on demand with [`to_arrow()`, `to_numpy()`, or `read_steps()`](timef-dataset.md). |
| **Manifest** | The compiled [`manifest.json`](manifest.md) for a dataset version: the card's metadata plus the schema derived from the data. The single source of truth the SDK reads. |
| **Connector** | One [`BaseConnector`](connectors.md) per dataset. `download()` fetches the raw source. `convert()` builds a `TimeFDataset`. It knows nothing about the engine or registry. |
| **Engine** | [`run_pipeline`](build.md): drives any connector through the fixed `download -> convert -> derive_schema -> store` pipeline, and owns caching and idempotency. |
| **Build** | Running a connector through the engine to compile a dataset and publish it to a registry, via the [`timenet-build`](build.md) CLI. |
| **Registry** | A served location of compiled TimeF versions that the [SDK](client.md) reads. It serves a Parquet control plane plus a Parquet or Zarr values plane. It never runs connector code. It can be a [local directory, S3, or a remote host](registry.md). |
| **Client (SDK)** | [`TimeNet`](client.md): the consumer entry point to search, download, and load datasets. |
| **Version** | An immutable snapshot of a dataset, addressed `org/name@version`. Once its `manifest.json` lands, TimeNet commits it atomically. |
