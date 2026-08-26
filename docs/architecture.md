---
icon: lucide/box
description: "How TimeNet's packages, registries, and build fit together."
tags:
  - guide
  - architecture
---

# Architecture

How TimeNet's packages, registries, and build fit together. This page is the map. Follow the links
for per-component detail.

---

## The big picture

TimeNet splits into three parts. A **connector** builds a raw source into a TimeF version. The
**client/SDK** reads its manifest from a **registry** and loads the data. The control plane is Parquet.
The values plane can be Parquet or Zarr. Reading never runs connector code. Against a local registry,
`load` can first build a dataset that the registry does not have from an installed connector.

| | What it is | Ships | Used by |
| --- | --- | --- | --- |
| **`timenet`** | Python package | TimeF format, reader/writer, registry client, engine, `BaseConnector`, SDK, CLI | everyone (`pip install timenet`) |
| **registry** | a served location | compiled TimeF versions | the SDK reads it and build publishes to it |
| **`timenet-connectors`** | a repo | connector recipes + cards + the `timenet-build` CLI | connector authors (clone it) |

There can be several registries: one public, private internal ones, or a local directory.

---

## The two flows

```
PRODUCE  dataset.yaml + connector
             │
             ▼
         engine   download ─► convert ─► derive_schema ─► store
             │
             ▼
         registry
             │
             ▼
CONSUME  SDK ─► open_version ─► TimeFReader ─► Arrow
```

The compiled `manifest.json` (the card's human-authored metadata plus the schema derived from the data)
is the single source of truth the SDK reads. `open_version` returns a handle: the manifest plus a
filesystem-rooted view of the version's files. `TimeFReader` reads through this handle. It loads each
series only on first use, not every file up front. Because the SDK never imports connector code,
everything a consumer needs to interpret either values backend lives in the manifest.

---

## Build roles: connector, engine, builder

Three producer-side pieces, each with one job:

| Role | What it is | Job |
| --- | --- | --- |
| **Connector** | one `BaseConnector` subclass per dataset ([connectors](connectors.md)) | the dataset-specific recipe: `download()` fetches raw files, `convert()` builds a `TimeFDataset`. Knows nothing about the engine or registry. |
| **Engine** | `run_pipeline` ([build & publish](build.md)) | drives any connector through the fixed pipeline and owns caching, idempotency, and `force` / `keep_cache`. Knows no dataset specifics. |
| **Builder** | the `timenet-build` CLI ([build](build.md)) | the entry point: resolves the id to its connector and runs the engine into a registry. |

```
timenet-build build org/name
  │
  ├─ builder  discovery.resolve("org/name") -> Connector class
  │             datasets/<org>/<name>/ exposes CONNECTOR
  │
  └─ engine   run_pipeline(connector, <registry>)
                metadata -> download -> convert -> derive_schema -> store
                  -> <registry>/org/name/<version>/
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
| Connector recipes + cards, `timenet-build` | `timenet-connectors` | producer |

---

## Design principles

- The manifest is self-describing. The SDK reads schema, counts, and file pointers from
  `manifest.json`. It never runs connector code or globs the directory.
- Types are plain frozen dataclasses. Specs, data sources, and annotations are frozen
  [descriptors](types.md), so they pickle and round-trip through the reader with no runtime class
  synthesis. That keeps multiprocessing `DataLoader` workers safe.
- Values are Arrow in, Arrow out. A [`TimeSeries`](timef-dataset.md) exposes `to_arrow()`,
  `to_numpy()`, and `read_steps()` over a private lazy loader. Its spec declares the scalar dtype and
  per-timestep shape. The writer stores scalar `float32` values in Parquet by default or uses Zarr for
  dtype-preserving scalar and multidimensional values.
- Units go through [pint](https://pint.readthedocs.io). One shared registry owns every definition
  and conversion.
- Commits are atomic. The writer stages a version into a temp directory and publishes it with a
  single atomic rename. Once `manifest.json` is present, the writer commits the version.
- Versions are immutable. Edits are copy-on-write. To remove a row, the writer writes a new version
  through the same atomic path ([`edit_version`](timef-writer.md#copy-on-write-edits)). Stable
  never-reused ids keep references valid. Content-defined chunking keeps the rewrite cheap on a
  deduplicating backend.

---

## Lifecycle of a dataset

1. **Author** a connector at `datasets/<org>/<name>/` (its `__init__.py` exposes `CONNECTOR`) with its
   `dataset.yaml` card beside it, in `timenet-connectors`.
2. **Build**: `timenet-build build <org>/<name>` runs the engine, compiles the manifest, and writes a TimeF version.
3. **Verify** locally: point the SDK at the output directory (itself a valid local registry).
4. **Publish** the complete TimeF version to a registry.
5. **Consume**: `timenet download <id>` reads the manifest and fetches every file it lists.
