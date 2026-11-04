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
| **[TimeF](timef-format.md)** | The one format for every dataset. It has one on-disk shape and one API, so an ECG, an accelerometer trace, and a market series all read the same way. |
| **[Dataset](data-model/datasets.md)** | A named, versioned collection of records in TimeF, addressed as `org/name` (for example `chengsenwang/tsqa`). |
| **[Record](data-model/records.md)** | One recording session containing a recursive hierarchy of Sources and Signals. |
| **[Source](data-model/sources.md)** | A device, sensor, or subsystem. A Source can contain child Sources and Signals. |
| **[Signal](data-model/time-series.md)** | One named value stream with its own TimeAxis and TimeSeriesSpec. Values remain lazy until read. |
| **[Manifest](manifest.md)** | The `manifest.json` entry point for a version. It contains metadata, schema, counts, checksums, and file paths. Its presence marks a committed version. |
| **[Connector](connectors.md)** | One `BaseConnector` per dataset. `download()` fetches the raw source. `convert()` builds a `TimeFDataset`. It knows nothing about the engine or registry. |
| **[Engine](build.md#pipeline)** | `run_pipeline`: drives any connector through the fixed `download -> convert -> derive_schema -> store` pipeline, and owns caching and idempotency. |
| **[Build](build.md)** | Running a connector through the engine to compile a dataset and publish it with the [`timenet-build`](cli/build.md) CLI. |
| **[Registry](registry.md)** | A served location of compiled TimeF versions that the [SDK](client.md) reads. It serves a DuckDB control plane plus a Parquet or Zarr values plane. It never runs connector code. It can be a [local directory, S3, or a remote host](registry.md#choosing-a-registry). |
| **[Client (SDK)](client.md)** | `TimeNet`: the consumer entry point to search, download, and load datasets. |
| **[Version](data-model/datasets.md#versions-are-immutable)** | An immutable snapshot of a dataset, addressed `org/name@version`. Once its `manifest.json` lands, TimeNet commits it atomically. |
