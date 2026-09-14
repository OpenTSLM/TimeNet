---
icon: lucide/box
description: "How TimeNet's two storage planes, registries, and consumer views fit together."
tags:
  - guide
  - architecture
---

# Architecture

How TimeNet's two storage planes, registries, and consumer views fit together. This page is the map.
Follow the links for per-component detail.

---

## The big picture

TimeNet ships one package, `timenet`. A dataset version splits into two planes that are stored
differently because they are read differently.

| | What it holds | How it is stored | Why |
| --- | --- | --- | --- |
| **control plane** | records, source trees, signals, tasks, annotations, chunk locators | one embedded DuckDB database, `control.duckdb` | small, deeply cross-referenced, read by point lookups, joins, and tree walks |
| **values plane** | the sample values of every signal | Parquet shards by default, a Zarr store with the `zarr` extra | bulk numeric data read by random access at a known offset |

Both planes ship inside one version directory, described by one `manifest.json`, served by a
[registry](registry.md). There can be several registries: one public, private internal ones, or a
local directory.

---

## The two flows

```text
PRODUCE  DeclarativeDataset
             │
             ▼
         TimeFWriter   walk ─► values plane ─► control.duckdb ─► manifest.json
             │
             ▼
         registry
             │
             ▼
CONSUME  TimeNet client ─► open ─► TimeFReader ─► pandas / torch
```

The writer walks `Task -> Record -> Source -> Signal`, streams each signal's values into the values
plane, and inserts the structure into the database. `manifest.json` is written last, so its presence
marks a committed version. The client reads that manifest, hands the reader a rooted view of the
version's files, and the reader answers from there. Nothing on the read path runs producer code.

---

## Why a database for the control plane

Reconstructing one task means walking `Task -> Records -> recursive Sources -> Signals -> axis and
spec -> annotations at every level`. Against normalized Parquet tables that is a recursive traversal
plus several joins, hand-rolled in Python with its own row-group pruning. In SQL it is one statement
the engine plans.

The database ships with no primary keys, no foreign keys, and no indexes. A version is written once
and never changes, so every invariant those constraints enforced is checked once, as a bulk
anti-join, before the writer publishes. Measured on ECG-QA (1.35M control rows): 19.4 MB of plain
tables against 189.0 MB with primary keys, foreign keys, and ten indexes. Point lookups run in
0.5-5 ms either way, and the one query that joins annotations to tasks across the whole dataset is
ten times faster without indexes (41 ms indexed, 4 ms not), because the planner hash-joins columns
instead of walking index nodes.

The values plane stays on pyarrow because DuckDB cannot express what it already does. DuckDB picks
Parquet encodings per column automatically and offers no per-column override, and its choice for
floats costs 1.8x on quantized clinical data. TimeNet keeps its own measured per-modality rule
instead (see [encodings](timef-format.md#encodings)).

---

## Batching is the read contract

Without indexes, every lookup scans its table, so a per-record read degrades linearly with corpus
size. Hydrating a batch does not: `records()` and `tasks()` answer a whole batch in a fixed number
of queries however large the batch is. Measured on a 618,508-record corpus:

| access pattern | throughput | per record |
| --- | --- | --- |
| `reader.record()`, one at a time | 29 rec/s | 34.49 ms |
| `iter_records(batch_size=512)` | 3,708 rec/s | 0.27 ms |
| `iter_records(batch_size=2048)` | 15,540 rec/s | 0.064 ms |

The same holds in the values plane: `values_for()` reads a batch of signals in one pass, while
reading one signal at a time re-opens the shard and re-decodes the row group for each. Every
consumer view is built on the batched calls for this reason.

`iter_records` and `iter_tasks` take `worker_index` and `num_workers`, so a `DataLoader`'s workers
split the corpus into disjoint slices without coordinating. Surrogate ids run 0, 1, 2, and so on
with no gap, which the writer checks before it publishes, so a worker's slice is a modulo of the id.

---

## Where each component lives

| Component | Module | Side |
| --- | --- | --- |
| Declarative hierarchy, `TimeFWriter` | `timenet.control_plane` | producer |
| `TimeFReader`, SQL queries, value reads | `timenet.control_plane` | consumer |
| Values backends | `timenet.control_plane.values`, `timenet.parquet` | shared |
| Manifest, format constants, checksums | `timenet.manifest`, `timenet.format` | shared |
| Registry client and backends | `timenet.registry`, `timenet.client` | consumer |
| Frame and tensor views | `timenet.pandas`, `timenet.torch` | consumer |
| Specs, units, metadata, enums | `timenet.types` | shared |

---

## Design principles

- The manifest is the contract. It lists every file with a `sha256:` checksum and a size, so a
  consumer verifies integrity and plans a download without opening the data.
- Joins run on dense integer ids the writer hands out during the walk. The caller's own id survives
  as `external_id` on the entity that owns it, so a record stays addressable by the name it has
  outside the dataset. Surrogate ids are not stable across rebuilds; nothing outside the file should
  quote one.
- An annotation's payload is stored once and attached many times. `patient_sex=male` on 100,000
  records costs one payload row and 100,000 short attachment rows.
- A signal can belong to several sources. Real corpora reuse one series across many questions, so
  the link is a table rather than a column, and shared values are written once.
- Units go through [pint](https://pint.readthedocs.io). One shared registry owns every definition
  and conversion.
- Commits are atomic. The writer stages a version into a temp directory and publishes it with a
  single rename. A reader either sees a complete version or sees nothing.
- Versions are immutable. That is what lets the control plane drop its constraints, and it is what
  makes a pinned `@version` mean exactly one set of bytes.

---

## Lifecycle of a dataset

1. **Build** a [`DeclarativeDataset`](timef-writer.md) from the raw source.
2. **Write** it with [`TimeFWriter`](timef-writer.md), which compiles both planes and the manifest.
3. **Verify** locally: the output directory is itself a valid local registry.
4. **Publish** the complete version to a registry.
5. **Consume**: `TimeNet().open(id)` returns a [`TimeFReader`](timef-reader.md) over it.
