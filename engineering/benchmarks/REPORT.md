# TimeF storage-format evaluation

A reproducible comparison of on-disk formats for TimeF's ragged float32 time series, on two real
datasets. Everything here is produced by [`format_eval.py`](format_eval.py); the numbers and plots below
come straight from one run. Sizes are MiB (2²⁰ bytes), labelled MB.

## TL;DR

- **The value-column encoding matters more than the container format, and the right encoding depends on
  the data.** BYTE_STREAM_SPLIT (TimeF's current hardcoded default) is an 11% win on continuous data
  (tsqa) but a **50% size regression on quantized ECG**, where dictionary encoding wins.
- **No alternative container beats a well-tuned Parquet.** On ECG, dictionary-Parquet (68 MB) is the
  smallest of everything tested; Zarr (77 MB) is closest. HDF5, Lance, Arrow IPC, and raw binary are all
  larger, and all forfeit directory interop and HuggingFace-Xet dedup.
- **Scattered single-series reads are a tuning problem, not a format flaw.** A memory-bounded reader
  row-group cache is an 11.7x win on ECG, and a 512 KiB row-group default (down from 4 MiB) is ~7x faster
  at random reads. Matched to the same chunk size, Parquet is competitive with or faster than Zarr, so
  there is no random-access case for switching formats.

## The two experiments

| | tsqa | ECG |
|---|---|---|
| Source | `ChengsenWang/TSQA` (HF Hub) | PTB-XL 500 Hz, `physionet/ecg-qa-cot` source |
| Series | 48,000 | 13,848 leads (1,154 records × 12) |
| Values | 11.5M | 69.2M |
| Raw float32 | 43.9 MB | 264.1 MB |
| Lengths | 64 / 128 / 256 / 512 | 5000 (10 s @ 500 Hz) |
| Value character | z-normalized, continuous | integer-microvolt, quantized |
| Distinct values | 10.5M (91%) | 11,149 (0.02%) |

These bracket the interesting range: high-entropy continuous floats vs low-cardinality quantized
biosignals. ECG is TimeNet's headline modality.

## How to run

```bash
# fetch both datasets, benchmark, and make plots (needs a normal network connection)
./engineering/benchmarks/format_eval.py all --ecg-records 1200 --data-dir ./data --out ./results

# or step by step
./format_eval.py prepare   # download + cache tsqa and ECG as .npy
./format_eval.py run       # size/encoding/amplification -> results.json + fig1..fig3
./format_eval.py plot      # re-render fig1..fig3 from results.json
./format_eval.py timing    # read timing -> timing.json + fig4..fig5
```

The script is a single file with an inline `uv` dependency header, so `uv run` (or the shebang) resolves
pyarrow, zarr, h5py, pylance, matplotlib, and wfdb on its own. Re-runs reuse the `.npy` cache. Fixed
seeds, so results are stable.

## Experiment 1: value-column encoding (BSS vs plain vs dictionary)

Same `list<float32>` values, zstd level 3, three encodings.

![encoding comparison](results/fig1_encoding.png)

| Encoding (zstd L3) | tsqa | vs plain | ECG | vs plain |
|---|---|---|---|---|
| BYTE_STREAM_SPLIT (current default) | **35.9 MB** | **-11.1%** | 121.9 MB | **+50.2%** |
| plain | 40.4 MB | baseline | 81.2 MB | baseline |
| dictionary | 40.9 MB | +1.4% | **67.8 MB** | **-16.5%** |

BSS transposes each float into byte planes so zstd compresses each plane separately. That helps when
consecutive values are smooth (tsqa: the high-order planes are nearly constant) and hurts when values sit
on a coarse grid (ECG is integer microvolts, so the low mantissa byte is noise that BSS isolates into an
incompressible plane). Dictionary wins on ECG because 69M samples take only 11,149 distinct values.

**Takeaway: pick the encoding per dataset. BSS is the wrong hardcoded default for biosignals.**

## Experiment 2: format bake-off (on-disk size)

Best-representative config per format, byte-identical data.

![format sizes](results/fig2_formats.png)

| Format | tsqa MB | ECG MB | Full read (ECG) | Random access | Interop / dedup |
|---|---|---|---|---|---|
| Parquet, best encoding | 35.9 | **67.8** | 0.87 s | row-group sized (Exp 3) | directory-native + Xet |
| Zarr v3 CSR + zstd | 40.4 | 76.5 | 0.16 s | ~12.6x amp (1 MiB chunk) | many chunk objects, no Xet |
| Arrow IPC + zstd | 40.5 | 96.6 | 0.29 s | whole file | single file, no Xet |
| HDF5 CSR gzip+shuffle | 35.8 | 117.9 | 1.23 s | ~19.4x amp | worst cloud/append; shuffle hurts |
| Parquet BSS (current default) | 35.9 | 121.9 | 0.90 s | row-group sized | directory-native + Xet |
| Lance | 45.6 | 225.1 | 0.32 s | `take()` 7-28 ms/1000 rows | library-gated, no Xet |
| raw CSR (float32 + offsets) | 44.3 | 264.2 | 0.11 s | mmap ~1x, ~6 ms/1000 | none |

On ECG the tuned Parquet is the smallest, Zarr is close, and everything else is bigger. Lance is
size-poor here (its float encoding compresses these signals weakly) though it is built for random row
access. HDF5's shuffle filter hurts quantized data the same way BSS does. The format is not the lever on
size; the encoding is.

## Experiment 3: scattered single-series random access (Parquet)

The reader resolves a series to `(shard, row_group, offset)` and calls
`read_row_group(rg, columns=["values"])`, which decodes a whole row group to return one series. Two knobs.

![random access](results/fig3_random_access.png)

**Row-group size.** Read amplification = bytes decoded / bytes needed, for K random series:

| Row-group target | tsqa K=100 | ECG K=100 | ECG row groups |
|---|---|---|---|
| 256 KiB | 143.9x | **6.1x** | 990 |
| 1 MiB | 294.6x | 19.7x | 262 |
| 4 MiB (default) | 391.8x | 46.5x | 66 |
| 16 MiB | 391.8x | 63.9x | 17 |

Smaller row groups cut amplification for sparse reads (ECG 46x → 6x). At K=1000 you touch nearly every
row group, so it converges near the full-scan floor (ECG ~6x, tsqa ~40x). tsqa amplifies far more because
its series are tiny, so a row group packs thousands of them.

**Reader row-group cache.** The reader keeps a fixed 16-entry LRU of decoded row groups. On ECG (66 row
groups at 4 MiB), a 1000-series scattered read thrashes:

| Cache size | Physical `read_row_group` calls | Wall time |
|---|---|---|
| 16 (current default) | 764 | 5167 ms |
| covers the 66 touched groups | 66 | 441 ms |

An 11.7x win from a memory-bounded cache. (tsqa never shows this: its whole store is 11 row groups,
already inside the cache.)

## Experiment 4: read timing (random vs batch vs scan)

Amplification (Experiment 3) is bytes touched; this is wall time. Three access patterns per format, each
using its native API, warm cache, median of 3. Random = 1000 individual random series; batch = gather
16 random batches of 64; scan = full sequential read. Latency is µs per series (lower better),
throughput MB/s (higher better).

![read latency](results/fig4_read_latency.png)

![scan throughput](results/fig5_scan_throughput.png)

ECG (264 MB), the meaningful large case:

| Format | random µs/series | batch µs/series | scan MB/s | size MB |
|---|---|---|---|---|
| raw mmap | **2** | **2** | **20139** | 264 |
| Lance | 157 | **24** | 1689 | 225 |
| Parquet 256 KiB | 358 | 338 | 3303 | 72 |
| Zarr 1 MiB | 1275 | 1274 | 3379 | 77 |
| HDF5 1 MiB | 3696 | 4243 | 265 | 118 |
| Parquet 4 MiB | 4957 | 3190 | 3482 | 65 |

The tradeoff:

- **Random point access** spans four orders of magnitude: raw mmap (~2 µs) >> Lance (~150 µs) >> Parquet
  256 KiB (~360 µs) > Zarr / HDF5 (~1.3–3.7 ms) > Parquet 4 MiB (~5 ms). For Parquet, **row-group size is
  the dominant knob: 256 KiB is ~14x faster than 4 MiB** for random reads, because each read decodes one
  whole row group.
- **Batching helps the row-addressed formats most.** Lance's `take()` coalesces a batch into ~24 µs/series
  (7 µs on tsqa); raw mmap stays at the floor. Parquet 4 MiB improves (fewer redundant decodes within a
  batch) but stays slow. Chunk/array formats (Zarr, HDF5) barely benefit, since each series still
  decompresses its chunk.
- **Sequential scan** is multi-GB/s for every columnar format (2–3.5 GB/s), **except HDF5 with gzip+shuffle
  (~265 MB/s, ~10x slower)**. raw mmap is fastest (no decode).

So: for shuffled random/minibatch training, Lance or small-row-group Parquet; for epoch scans, any columnar
format except HDF5-gzip. Within Parquet, dropping the row-group target to 256 KiB buys a 14x random-access
speedup for a modest scan-throughput cost. This is the timing evidence behind the dataset-adaptive
row-group recommendation. (Warm cache; cold/IO-bound reads would widen the gaps in favour of the
low-amplification formats: raw, Lance, small-row-group Parquet.)

## Recommendations

Detailed in [storage-format-benchmarks.md](storage-format-benchmarks.md) and the public
[storage format page](../../docs/timef-storage-format.md). In short: keep Parquet; make the value
encoding per-dataset (dictionary for quantized biosignals, BSS for continuous); make the reader cache
memory-bounded; and **change the default row-group target from 4 MiB to 512 KiB** (~7x faster random
reads, ~6% size cost on dictionary data, one extra GET per shard open on S3), kept overridable per
dataset.

## Caveats

- Latencies are warm-cache (the OS page cache could not be dropped). Trust the deterministic size and
  amplification ratios; treat "Nx faster" wall times as indicative.
- Cross-format random-access amplification for Zarr/HDF5 is at their fixed 1 MiB chunk; Parquet's depends
  on row-group size (Experiment 3). Lance and raw use their own access paths (`take`, mmap), reported
  separately.
- ECG is a ~1,150-record subset of PTB-XL; all series are one chunk (< 262,144 values), so multi-chunk
  long-series behaviour is not exercised here.
