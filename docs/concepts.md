---
icon: lucide/book-open
description: "Key TimeNet terminology: TimeF, dataset, record, source, signal, registry, and more."
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
| **Version** | An immutable snapshot of a dataset, addressed `org/name@version`. Once its `manifest.json` lands, the version is committed and never changes. |
| **Control plane** | The structure of a version: records, sources, signals, tasks, annotations, and chunk locators, in one embedded DuckDB database ([format](timef-format.md)). |
| **Values plane** | The sample values of every signal, in Parquet shards or a Zarr store beside the database. |
| **Record** | One recording session: a patient examination, a machine run, a market window. It holds [sources](data-model/records.md) and carries [annotations](data-model/annotations.md). |
| **Source** | A device, a sensor, or an assembly holding other sources. Sources nest, so an IMU is a source holding an accelerometer, a gyroscope, and a magnetometer. |
| **Signal** | One sequence of values with one time axis and one spec: the leaf of the hierarchy ([signals](data-model/signals.md)). |
| **Annotation** | One `(name, value, unit)` statement attached to a dataset, task, record, source, or signal, placed in time as static, point, or interval ([annotations](data-model/annotations.md)). |
| **Task** | A prompt, the inputs it gives a model, and the target it expects ([tasks](data-model/tasks.md)). |
| **Manifest** | The compiled [`manifest.json`](manifest.md): the dataset's identity plus a checksummed descriptor for every file. |
| **Registry** | A served location of committed TimeF versions that the [client](client.md) reads. It can be a [local directory, S3, or a remote host](registry.md). |
| **Client (SDK)** | [`TimeNet`](client.md): the consumer entry point to search, download, and open datasets. |
| **External id** | The id a builder gave an entity. It survives a rebuild, and it is what the reader's `record()` and `task()` take. |
| **Surrogate id** | The dense integer the writer assigns during the walk. Every join runs on it. It is not stable across rebuilds, so nothing outside the file should quote one. |
