---
icon: lucide/database
description: "Why TimeF stores waveforms as Parquet, measured against HDF5, Zarr, Lance, Arrow IPC, and raw binary."
tags:
  - reference
  - decision
  - writer
---

# Storage format

TimeF stores waveform values as Parquet shards of `list<float32>` (see [TimeFWriter](timef-writer.md)).
This page records why, measured against the alternatives, and what we would change. It answers a set of
questions raised in review: does Parquet handle ragged and irregularly sampled series, is its list
encoding efficient, do byte-range and parallel reads work, and does ID-addressed random access scale.

**Decision: keep Parquet.** Across a format bake-off it is the smallest layout and the fastest full read
of the compressed options, and it is the only one that keeps two things TimeNet is built on: a directory
any tool reads directly (pandas, polars, DuckDB, Spark) and the HuggingFace-Xet content-defined-chunking
dedup path. The one real weakness is scattered single-series random access, and that is reader tuning
inside Parquet, not a reason to change format.

## How this was measured

Two datasets, real code, warm-cache latencies (the macOS page cache could not be dropped, so trust the
byte-counting amplification ratios over the latency multipliers):

- A synthetic ragged fixture: 10,000 univariate series, 96M float32 values (366 MiB raw), lengths 64 to
  200k, quantized biosignal-like. Used for the full six-format bake-off.
- **Real `chengsenwang/tsqa`**: all 48,000 series, 11.52M values (43.9 MiB raw), lengths 64/128/256/512,
  z-normalized (mean 0, std 1), **91% distinct float32 values**. The high-entropy case.
- **Real PTB-XL ECG** (a `physionet/ecg-qa-cot` subset): ~1,150 records, 13,848 leads, 69M values (264 MiB
  raw), 12 leads at 500 Hz in integer-microvolt millivolts, **0.02% distinct values**. The quantized
  biosignal case, and the primary modality.

The synthetic fixture and the two real datasets disagree on the encoding (below), and the real data wins.

## The questions, answered

**Does Parquet work for time-series-heavy data at scale?** Yes for the dominant loads (bulk train/eval
scans and archival size). On the fixture it was the smallest tested layout and the fastest compressed
full scan (up to ~2.6 GB/s), and it releases the GIL on decode, so parallel scan scales ~2.3-3.6x on 8
threads.

**Can 10,000 ragged, unequal-length series live in `list<float32>` efficiently?** Yes, and it is not
penalized versus the "obvious" flat float column. On real tsqa the list layout (37.62 MB) came out 1.4%
*smaller* than a flat column plus an offset sidecar (38.16 MB); on the synthetic fixture it was 11-12%
smaller. The only hard limit is the 2³¹ 32-bit-offset cap per row group, already guarded by
`MAX_ELEMENTS_PER_ROW_GROUP`. `fixed_size_list` padded to the max length is not viable (a 20.8x blow-up
at 95% padding on the fixture).

**Is Parquet's list encoding actually inefficient?** No. That warning is about deeply nested schemas,
many tiny or null-heavy lists, and string lists. For a single-level, non-null numeric column the
repetition/definition levels RLE-collapse into measurement noise, and the list column came out smaller
than flat on both datasets.

**Irregular / unequal sampling?** Feasible, and dropping it for v1 was the right call. Adding an explicit
per-sample timestamp column as `int64` microseconds with DELTA_BINARY_PACKED costs about 1.7x total size
on the fixture; nanoseconds is the 2.25x trap and `float64` seconds is 4x. If a real irregular dataset
lands it is a clean additive change: an optional timestamp column that uniform readers skip via column
projection at near-zero read cost. No layout change is needed now.

**Parallel and byte-range reads?** They work. A projected single-row-group read fetches one contiguous
byte range plus a fixed footer tail (two GETs), with real column projection. Two caveats for the future
S3 path: keep row-groups-per-shard under ~70 so a shard's footer stays inside one open GET for the
8-column schema, and decode releases the GIL so thread-parallel reads scale.

**Scattered-ID random access, and does the index scale?** The index scales: it loads into RAM and footers
stay linear. The cost of scattered single-series reads is tunable inside Parquet, driven by two things:

- `read_row_group` decodes a whole row group to return one chunk, so a small series amplifies badly.
  On real tsqa, reading 100 random series amplifies 391x at the 4 MiB default row-group size and 144x at
  256 KiB. Row-group size is the free knob (see recommendations).
- The reader keeps a fixed 16-entry LRU of decoded row groups (`_ROW_GROUP_CACHE_SIZE`). On a large
  dataset that thrashes: on the 366 MiB fixture a 1000-series scattered read issued 833 physical
  `read_row_group` calls for only 88 distinct row groups; covering them cut wall time ~9x. tsqa is too
  small to show this (its whole store is 11 row groups, already inside the cache).

Honest floor: reading a scattered ~10% of a dataset touches nearly every row group, so it costs about a
full scan no matter how you tune Parquet. A row-addressed format (Lance `take()`, raw-CSR mmap) still
touches far fewer bytes for genuinely sparse point lookups.

## Bake-off

Synthetic fixture (10k series, 96M values, 366 MiB raw). Sizes and bytes-touched are deterministic;
latencies are warm-cache.

| Format (best config) | Size | vs raw | Scattered single-series | Full scan | Interop / dedup |
| --- | --- | --- | --- | --- | --- |
| **Parquet BSS+zstd (TimeF)** | 220 MB | 1.67x | whole ~4 MiB row group decoded; ~7x amp at 256 KiB | **fastest compressed** | **directory-native + Xet CDC** |
| Parquet plain+zstd | 171 MB | 2.14x | same | same | same |
| HDF5 CSR gzip+shuffle | 228 MB | 1.61x | 2.6x amp, fast local | slow | worst cloud/append; vlen compression a no-op |
| Zarr sharded CSR | 250 MB | 1.46x | ~39x fewer bytes than Parquet | slowest to scan | many files; no Hub dedup |
| Lance zstd + BTREE | 274 MB | 1.34x | `take()` 8-266x fewer bytes; by-id 88x; zero-copy COW | ~4.5x slower | library-gated, no Xet |
| Arrow IPC uncompressed | 366 MB | 1.00x | zero-copy but worse amplification than Parquet | fastest (zero-copy) | larger than raw |
| Raw CSR mmap | 366 MB | 1.00x | **~2 µs/series, ~1x amp (the floor)** | fastest | no interop, no dedup, no compression |

Parquet owns size and full-scan. Lance and raw-CSR own scattered bytes-touched. Nobody else has a
like-for-like size edge, and everyone else forfeits Xet dedup, a stated design goal.

## Encoding is data-dependent, and BSS is the wrong default for biosignals

Measured on two real datasets, the value-column encoding should be chosen per dataset. BYTE_STREAM_SPLIT
wins on high-entropy continuous floats and loses badly on quantized signals:

| `values` (`list<float32>`, zstd L3) | tsqa (z-normalized, 91% distinct) | PTB-XL ECG (quantized µV, 0.02% distinct) |
| --- | --- | --- |
| BYTE_STREAM_SPLIT + zstd (current default) | **37.6 MB** (-11% vs plain) | 127.8 MB (**+50% vs plain**) |
| plain + zstd | 42.3 MB | 85.1 MB |
| dictionary + zstd | 42.9 MB (+1%) | **71.1 MB** (-16% vs plain) |

BSS transposes each float into byte planes, which helps when consecutive values are smooth (tsqa) and
hurts when they are quantized to a grid, because the low mantissa byte becomes noise (real PTB-XL ECG, the
headline modality, is integer-microvolt). On archival ECG the current BSS default is a 50% size regression
and dictionary is best. So BSS is the wrong hardcoded default: pick the encoding per dataset by measuring.
Detailed biosignal benchmarks live in the internal engineering notes, not here.

## Recommended changes

Priority reflects evidence strength and cost, not urgency. None require a format or interop change.

**P0, reader row-group cache.** Replace the fixed `_ROW_GROUP_CACHE_SIZE = 16` in
`timenet.reader.reader` with a memory-bounded LRU (cap by decoded bytes, default well above 16 entries).
Highest leverage on large datasets: ~9x on a 1000-series scattered read on the fixture, and it removes the
833-reads-for-88-groups thrash. RAM-only cost; size it to a budget. Does not affect small datasets like
tsqa.

**P1, drop the default row-group size to 512 KiB.** Lower `DEFAULT_ROW_GROUP_TARGET_BYTES` from 4 MiB to
512 KiB, kept per-dataset overridable (the writer already threads `row_group_target_bytes`). Random reads
decode a whole row group per series, so this is ~7x faster at random single-series reads on ECG (795 vs
5690 µs) and cuts read amplification from ~25x to ~6x, for a small on-disk cost (+5.9% with dictionary
encoding, +0.5% with BSS). The tradeoff: a 128 MiB shard then holds ~256 row groups, so its footer grows
to ~240 KB, past pyarrow's 64 KB open read, meaning a shard open costs a second GET on S3 (free locally).
Keep single-GET opens by staying at 2 MiB+ row groups or using smaller shards; keep 1-4 MiB for long
PhysioNet recordings read mostly by scan.

**P1, choose the value encoding per dataset.** The hardcoded BSS default is a 50% size regression on real
quantized ECG, where dictionary is best; it only wins on high-entropy data like tsqa. Replace the fixed
BSS self-check with a per-dataset pass that writes the smallest of `{dictionary, plain, BSS}`. Correct the
`encodings.py` comment and the storage-decisions memory note that claim BSS wins on quantized biosignals.

**P2, build only when a real requirement appears.**

- Optional `int64`-microsecond timestamp column for irregular sampling (~1.7x total, projection-skippable).
- Real S3 range-read registry plus `pyarrow.dataset` for shard discovery and projection (the `s3://` and
  remote registries are currently stubs). Greenfield, so no migration; bound shard count so cold opens
  don't fan out into thousands of footer GETs.
- A per-dataset random-access sidecar (raw-CSR mmap first, Lance only if copy-on-write versioning is also
  wanted), built once from the Parquet archive, never as the distributed format. Only if scattered by-id
  access is *proven* the dominant cost on real S3.

## What we will not do

- **Not switch the interchange format to Lance.** It is the strongest alternative and its random-access
  win is real, but switching trades away smaller files, ~4.5x faster full scans, Xet dedup, and universal
  directory interop, for a workload not yet proven dominant, and ~9x of the gap closes for free via P0.
  Keep Lance in reserve as an optional local cache, not the format.
- **Not switch to HDF5, Zarr, Arrow IPC, or raw binary.** On real biosignal data Zarr with zstd ties a
  dictionary-encoded Parquet on size and has lower random-access amplification, but a Parquet tuned per the
  changes above matches it while keeping directory interop and Xet dedup that Zarr forfeits (and Zarr fans
  out into thousands of chunk objects). HDF5's vlen compression is a no-op with the worst cloud/append
  story; Arrow IPC is larger with worse amplification; raw binary has no interop or dedup. The lever on
  this data is the encoding, not the container.
- **Not pursue page-level single-row addressing.** `write_page_index` is already on, but pyarrow exposes
  no public per-page read for nested `list<float32>`. Smaller row groups are the only lever until it does.
- **Not hardcode one value encoding for every dataset, and not adopt `fixed_size_list`.**

## Open questions

- All latencies are warm-cache; cold scattered reads and real S3 round-trips are unmeasured. Trust the
  amplification ratios, treat "Nx faster" as indicative.
- The S3 path is a stub, so byte-range and parallel-decode wins are validated locally only. Lance's
  ~1-IOP-per-row advantage is untested on object storage.
- No fixture series exceeded one chunk (< 262,144 values), so multi-chunk long-series behavior is
  inferred. PhysioNet multi-GB recordings will exercise it.
- Whether scattered by-id access at training scale is actually TimeNet's dominant cost is unproven, and it
  drives the entire P2 sidecar decision. Measure the real loader access pattern before building it.
