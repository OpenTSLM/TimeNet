---
icon: lucide/box
description: "How TimeNet's packages, registries, and curation fit together."
tags:
  - guide
  - architecture
---

# Architecture

How TimeNet's packages, registries, and curation fit together. This page is the map; follow the links
for per-component detail.

---

## The big picture

TimeNet splits into three parts. A **connector** curates a raw source into a manifest plus parquet
artifacts; the **client/SDK** reads the manifest from a **registry** and loads the data. The client
never runs connector code.

| | What it is | Ships | Used by |
| --- | --- | --- | --- |
| **`timenet`** | Python package | TimeF format, reader/writer, registry client, engine, `BaseConnector`, SDK, CLI | everyone (install coming soon) |
| **registry** | a served location | compiled manifests + parquet | the SDK reads it; curation publishes to it |
| **`timenet-connectors`** | a repo | connector recipes + cards + the `timenet-curate` CLI | connector authors (clone it) |

There can be several registries: one public, private internal ones, or a local directory.

---

## The two flows

```
PRODUCE  dataset.yaml + connector ─► engine (download -> convert -> derive_schema -> store) ─► publish ─┐
                                                                                                     ▼
                                                                                                 registry
CONSUME  SDK ─► get_manifest ─► fetch parquet ─► TimeFReader ─► Arrow  ◄──────────────────────────────┘
```

The compiled `manifest.json` (the card's human-authored metadata plus the schema derived from the data)
is the single source of truth the SDK reads. Because the SDK never imports connector code, everything a
consumer needs to interpret the parquet lives in the manifest.

---

## Curation roles: connector, engine, curator

Three producer-side pieces, each with one job:

| Role | What it is | Job |
| --- | --- | --- |
| **Connector** | one `BaseConnector` subclass per dataset ([connectors](connectors.md)) | the dataset-specific recipe: `download()` fetches raw files, `convert()` builds a `TimeFDataset`. Knows nothing about the engine or registry. |
| **Engine** | `run_pipeline` ([curate & publish](curation.md)) | drives any connector through the fixed pipeline and owns caching, idempotency, and `force` / `clean_cache`. Knows no dataset specifics. |
| **Curator** | the `timenet-curate` CLI ([curation](curation.md)) | the entry point: resolves the id to its connector and runs the engine into a registry. |

```
timenet-curate build org/name                          curator
  └─ discovery.resolve("org/name") -> Connector class   (datasets/<org>/<name>/ exposes CONNECTOR)
      └─ run_pipeline(connector, <registry>)            engine
           metadata -> download -> convert -> derive_schema -> store -> <registry>/org/name/<version>/
```

`metadata()` reads the `dataset.yaml` card and `store()` streams through
[`TimeFWriter`](timef-writer.md). The output directory is itself a valid local registry, so the consume
flow reads it straight back.

---

## Where each component lives

| Component | Package | Side |
| --- | --- | --- |
| TimeF format, types, `TimeFDataset`, manifest | `timenet` | shared |
| CLI, SDK, registry client, `TimeFReader` | `timenet` | consumer |
| Engine, `TimeFWriter`, `BaseConnector` | `timenet` | producer |
| Connector recipes + cards, `timenet-curate` | `timenet-connectors` | producer |

---

## Design principles

- The manifest is self-describing. The SDK reads schema, counts, and file pointers from
  `manifest.json`. It never runs connector code or globs the directory.
- Types are plain frozen dataclasses. Specs, data sources, and annotations are frozen
  [descriptors](types.md), so they pickle and round-trip through the reader with no runtime class
  synthesis. That keeps multiprocessing `DataLoader` workers safe.
- Values are Arrow in, Arrow out. A [`TimeSeries`](timef-dataset.md) exposes `to_arrow()` and
  `to_numpy()` over a private lazy loader. The writer stores `float32` values as Parquet with
  `BYTE_STREAM_SPLIT` and zstd.
- Units go through [pint](https://pint.readthedocs.io). One shared registry owns every definition
  and conversion.
- Commits are atomic. The writer stages a version into a temp directory and publishes it with a
  single atomic rename. Once `manifest.json` is present, the version is committed.
- Versions are immutable; edits are copy-on-write. Removing a row writes a new version through the
  same atomic path ([`edit_version`](timef-writer.md#copy-on-write-edits)); stable never-reused ids keep
  references valid, and content-defined chunking keeps the rewrite cheap on a deduplicating backend.

---

## Lifecycle of a dataset

1. **Author** a connector at `datasets/<org>/<name>/` (its `__init__.py` exposes `CONNECTOR`) with its
   `dataset.yaml` card beside it, in `timenet-connectors`.
2. **Curate**: `timenet-curate build <org>/<name>` runs the engine, compiles the manifest, and writes parquet.
3. **Verify** locally by pointing the SDK at the output directory (itself a valid local registry).
4. **Publish** the manifest + parquet to a registry.
5. **Consume**: `timenet download <id>` reads the manifest and fetches the parquet.
