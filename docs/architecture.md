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
**client/SDK** reads its manifest from a **registry** and loads the data. The relational control
plane is one immutable DuckDB file. The values plane can be Parquet or Zarr. Reading never runs
connector code. Against a local registry, `open` and `load` can first build a dataset that the
registry does not have from an installed connector.

| | What it is | Ships | Used by |
| --- | --- | --- | --- |
| **`timenet`** | Python package | TimeF format, reader/writer, registry client, engine, `BaseConnector`, SDK, CLI | everyone (`pip install timenet`) |
| **registry** | a served location | compiled TimeF versions | the SDK reads it and build publishes to it |
| **`timenet-connectors`** | Python package and source repository | connector recipes, cards, and the `timenet-build` CLI | dataset builders and connector authors |

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

The compiled `manifest.json` is the entry point and commit marker. It contains the card metadata,
derived schema, counts, and file list. `open_version` returns a handle to the manifest and version
files. `TimeFReader` uses the handle to read the DuckDB hierarchy and lazy Signal values. The SDK
never imports connector code.

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

`metadata()` reads the `dataset.yaml` card. The engine streams the converted dataset through
[`TimeFWriter`](timef-writer.md). A local output directory is also a valid registry, so the client
can read it directly.

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

- The manifest describes the version. The SDK reads schema, counts, and file pointers from
  `manifest.json`. The object hierarchy lives in `control.duckdb`.
- Wire-facing types are immutable Pydantic models. Runtime objects use typed dataclasses. Specs and
  annotation content are typed [descriptors](types.md), so they pickle and round-trip through the
  reader with no runtime class synthesis. That keeps multiprocessing `DataLoader` workers safe.
- NumPy and Arrow values share one array handle interface. A [`Signal`](timef-dataset.md#signal)
  reads `to_arrow()`, `to_numpy()`, and `read_steps()` through a private array handle. NumPy
  conversion rejects null values. Arrow reads preserve them. Its spec declares the scalar dtype and
  per-timestep shape. The writer stores typed scalar values in Parquet by default and uses Zarr for
  dtype-preserving multidimensional values.
- Units go through [pint](https://pint.readthedocs.io). One shared registry owns every definition
  and conversion.
- Commits are atomic. The writer stages a version into a temp directory and publishes it with a
  single atomic rename. Once `manifest.json` is present, the writer commits the version.
- Versions are immutable. A change produces a new version instead of modifying a committed one.

---

## Lifecycle of a dataset

1. **Author** a connector under `timenet_connectors/datasets/<org>/<name>/`. Its `__init__.py`
   exposes `CONNECTOR`, and its `dataset.yaml` card sits beside it.
2. **Build**: `timenet-build build <org>/<name>` runs the engine, compiles the manifest, and writes a TimeF version.
3. **Verify** locally: point the SDK at the output directory (itself a valid local registry).
4. **Publish** the complete TimeF version to a registry.
5. **Consume**: use `TimeNet.open()` for lazy access, `TimeNet.load()` for a dataset with every
   value in memory, or `timenet download` from the CLI for a local copy.

## Dataset inheritance

A dataset can derive from one or more parent datasets. It builds on their data and does not store
their values a second time. For example, `physionet/ecg-qa-cot` adds questions to the recordings of
`physionet/ptb-xl`. The child records its parents and their lineage. So it credits them, and their
access requirements also apply to the child.

Each parent is an exact release, `org/name@version`, that the card of the child declares under
`parents`. While the connector builds the child, it copies Records, Signals, annotations, and tasks
from the parents. The child keeps the copies in its own DuckDB catalog, under IDs of its own. It
reads their values from the release that stores them, its payload source. See
[Importing from a parent](timef-dataset.md#importing-from-a-parent) and
[Compose a child dataset](connectors.md#compose-a-child-dataset).

A build needs the parent releases, and a read needs only the payload sources. Before the connector
runs, the build prepares each parent release, as
[Prepare parent releases](build.md#prepare-parent-releases) describes. `TimeNet.open()`,
`TimeNet.load()`, `TimeNet.open_torch()`, and `TimeFReader.open()` resolve the payload sources
before the first read. A read never opens the catalog of a parent. Against a local registry, the
client reuses a complete copy of each release, copies it from `fetch_from`, or builds it.
`auto_build` is on by default. A remote client never runs connector code. See
[Open, load, or download](client.md#open-load-or-download).

The child keeps an internal map from the ID of each parent object to the ID of its copy. Through
this map, `get(..., parent=)` finds a copy, and a later import reuses it. The child also stores a
read order of its Records, so that a full read follows the order of the stored values. Its manifest
pins each file that it reads from a payload source by path, size, and checksum, as
[Shared payloads](timef-format.md#shared-payloads) describes.
