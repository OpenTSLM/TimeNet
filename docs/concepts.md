---
icon: lucide/book-open
description: "Key TimeNet terminology: TimeF, dataset, sample, connector, curation, registry, and more."
tags:
  - guide
  - concepts
---

# Concepts

The vocabulary that shows up across these docs, in one place. Follow a link for the full treatment.

| Term | What it is |
| --- | --- |
| **TimeF** | The shared format every dataset is stored and loaded in: one on-disk shape and one API, so an ECG, an accelerometer trace, and a market series all read the same way. |
| **Dataset** | A named, versioned collection of samples in TimeF, addressed as `org/name` (for example `chengsenwang/tsqa`). |
| **Sample** | One record in a dataset (say, a single patient recording): its [time series](timef-dataset.md) plus tasks, annotations, and metadata. |
| **Time series** | One logical stream within a sample, with shape `(n_steps, *value_shape)` and a dtype declared by its spec. Values are read on demand with [`to_arrow()`, `to_numpy()`, or `read_steps()`](timef-dataset.md). |
| **Manifest** | The compiled [`manifest.json`](manifest.md) for a dataset version: the card's metadata plus the schema derived from the data. The single source of truth the SDK reads. |
| **Connector** | One [`BaseConnector`](connectors.md) per dataset. `download()` fetches the raw source, `convert()` builds a `TimeFDataset`. It knows nothing about the engine or registry. |
| **Engine** | [`run_pipeline`](curation.md): drives any connector through the fixed `download -> convert -> derive_schema -> store` pipeline, and owns caching and idempotency. |
| **Curation** | Running a connector through the engine to compile a dataset and publish it to a registry, via the [`timenet-curate`](curation.md) CLI. |
| **Registry** | A served location of compiled TimeF versions that the [SDK](client.md) reads. It serves a Parquet control plane plus a Parquet or Zarr values plane and never runs connector code: a [local directory, S3, or a remote host](registry.md). |
| **Client (SDK)** | [`TimeNet`](client.md): the consumer entry point to search, download, and load datasets. |
| **Version** | An immutable snapshot of a dataset, addressed `org/name@version`. It is committed atomically once its `manifest.json` lands. |
