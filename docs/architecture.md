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
| **`timenet`** | Python package | TimeF format, reader/writer, registry client, engine, `BaseConnector`, SDK, CLI | everyone (`pip install timenet`) |
| **registry** | a served location | compiled manifests + parquet | the SDK reads it; curation publishes to it |
| **`timenet-connectors`** | a repo | connector recipes + cards + the `timenet-curate` CLI | connector authors (clone it) |

There can be several registries: one public, private internal ones, or a local directory.

---

## The two flows

```
PRODUCE  card.yaml + connector ─► engine (download → convert → derive_schema → store) ─► publish ─┐
                                                                                                  ▼
                                                                                              registry
CONSUME  SDK ─► get_manifest ─► fetch parquet ─► TimeFReader ─► Arrow  ◄───────────────────────────┘
```

The compiled `manifest.json` (the card's human-authored metadata plus the schema derived from the data)
is the single source of truth the SDK reads. Because the SDK never imports connector code, everything a
consumer needs to interpret the parquet lives in the manifest.

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

- **The manifest is self-describing.** The SDK reads schema, counts, and file pointers from
  `manifest.json`; it never runs connector code or globs the directory.
- **Types are plain frozen dataclasses.** Specs, data sources, and annotations are frozen
  [descriptors](types.md), so they pickle and round-trip through the reader with no runtime class
  synthesis (safe for multiprocessing `DataLoader` workers).
- **Values are Arrow in, Arrow out.** A [`TimeSeries`](timef-dataset.md) exposes `to_arrow()` /
  `to_numpy()` over a private lazy loader; the writer stores `float32` values as Parquet with
  `BYTE_STREAM_SPLIT` + zstd.
- **Units go through [pint](https://pint.readthedocs.io).** One shared registry owns every definition
  and conversion.
- **Commits are atomic.** The writer stages a version into a temp directory and publishes it with a
  single atomic rename; `manifest.json` present means committed.
- **Versions are immutable; edits are copy-on-write.** Removing a row writes a new version through the
  same atomic path ([`edit_version`](timef-writer.md#copy-on-write-edits)); stable never-reused ids keep
  references valid, and content-defined chunking keeps the rewrite cheap on a deduplicating backend.

---

## Lifecycle of a dataset

1. **Author** a connector at `datasets/<org>/<name>.py` (exposing `CONNECTOR`) with its dataset card
   beside it, in `timenet-connectors`.
2. **Curate**: `timenet-curate build <org>/<name>` runs the engine, compiles the manifest, and writes parquet.
3. **Verify** locally by pointing the SDK at the output directory (itself a valid local registry).
4. **Publish** the manifest + parquet to a registry.
5. **Consume**: `timenet download <id>` reads the manifest and fetches the parquet.
