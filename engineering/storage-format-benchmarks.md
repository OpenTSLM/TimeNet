# TimeF storage format: real-data benchmarks

Internal measurement log behind the [public storage-format decision](../docs/timef-storage-format.md).
This is the raw evidence, kept out of the published docs. Date: 2026-07-13.

The public page states the decision (keep Parquet) and answers the review questions. This page records the
numbers, including the one that overturns a long-standing assumption: **BYTE_STREAM_SPLIT is the wrong
default encoding for quantized biosignals, the primary modality.**

## Datasets

Real data, read exactly as the connectors read it (physical float32).

| Dataset | Series | Values | Raw f32 | Value character | Distinct values |
| --- | --- | --- | --- | --- | --- |
| Synthetic fixture | 10,000 | 96.0M | 366 MiB | quantized biosignal-like | 957 (0.001%) |
| `chengsenwang/tsqa` (real) | 48,000 | 11.5M | 43.9 MiB | z-normalized, continuous | 10.5M (91%) |
| PTB-XL ECG (real subset) | 13,848 leads | 69.2M | 264 MiB | integer-microvolt (500 Hz, mV) | 11,149 (0.02%) |

PTB-XL is `physionet/ecg-qa-cot`'s signal source. The connector's loader returns raw physical millivolts
via `wfdb.rdsamp`, so on disk the values are quantized to integer microvolts (gain 1000 ADC/mV). This is
what TimeF actually stores. The training-time z-normalization the dataset card mentions happens at load
time, not on disk.

## Finding 1: encoding is data-dependent, BSS is wrong for biosignals

`values` as `list<float32>`, zstd level 3, the same three encodings on each dataset:

| Encoding | Synthetic | tsqa (high-entropy) | PTB-XL ECG (quantized) |
| --- | --- | --- | --- |
| BYTE_STREAM_SPLIT + zstd (current default) | 220 MB (+29%) | **37.6 MB (-11%)** | 127.8 MB (**+50%**) |
| plain + zstd | 171 MB (baseline) | 42.3 MB (baseline) | 85.1 MB (baseline) |
| dictionary + zstd | **114.7 MB (-33%)** | 42.9 MB (+1%) | **71.1 MB (-16%)** |

Percentages are versus plain+zstd. BSS was verified applied to the leaf on each run
(`values.list.element` encoding = `('RLE', 'BYTE_STREAM_SPLIT')`).

**Mechanism.** BSS transposes each float into four byte planes so zstd can compress each plane
separately. It wins when consecutive values are smooth, because the sign and high-mantissa planes are
nearly constant (tsqa, continuous z-normalized floats: -11%). It loses when values are quantized to a
grid, because the low mantissa byte is effectively noise that BSS isolates into an incompressible plane
(both quantized datasets: +29% synthetic, +50% real ECG). Dictionary wins on quantized data because there
are only ~11k distinct values in 69M samples.

**Consequence.** On real archival ECG the hardcoded BSS default is a 50% size regression versus plain and
79% versus dictionary (127.8 MB vs 71.1 MB). ECG is the headline TimeNet modality, so this is not an edge
case. The writer's `encodings.py` comment and the `timenet-storage-format-decisions` memory note both
claim "BSS ~30% smaller than plain+zstd on quantized biosignals"; real PTB-XL data contradicts that. The
earlier ~30% figure was likely measured on z-normalized or otherwise continuous float data, not the raw
quantized signal the writer emits.

`list<float32>` versus a flat float column stays fine on ECG: list = 127.8 MB, flat + offset sidecar =
143.7 MB, so the list form is 11% smaller. No list-encoding penalty.

## Finding 2: random access on a realistically large dataset

The 264 MiB ECG store finally exercises the paths tsqa (36 MiB, 11 row groups) was too small to stress.
TimeF-faithful build: one row per lead, byte-based row groups, BSS store ~122 MiB.

**Reader row-group cache (P0).** Scattered read of 1000 random leads, 4 MiB row groups (66 groups total):

| Reader `_ROW_GROUP_CACHE_SIZE` | Physical `read_row_group` calls | Wall time |
| --- | --- | --- |
| 16 (current default) | 764 | 5093.8 ms |
| 67 (covers the touched groups) | 66 | 460.2 ms |

The fixed 16-entry LRU thrashes: 764 decodes for 66 distinct groups, an 11x wall-time penalty. A
memory-bounded cache removes it. (tsqa never showed this: its whole store is 11 groups, already inside
the cache.)

**Row-group size (P1).** Read amplification = row-group bytes decoded / target-series bytes, K random
leads:

| Row-group target | Row groups | K=100 amplification | K=1000 amplification |
| --- | --- | --- | --- |
| 256 KiB | 990 | **6.1x** | 4.5x |
| 1 MiB | 262 | 19.7x | 6.3x |
| 4 MiB (default) | 66 | 46.5x | 6.4x |
| 16 MiB | 17 | 63.9x | 6.4x |

For sparse reads (K=100) smaller row groups cut amplification 46x to 6x. At K=1000 you touch nearly every
group regardless, so it converges near the full-scan floor. Trade against footer cost: keep
row-groups-per-shard under ~70 for a single-GET footer on S3.

## Finding 3: Parquet vs Zarr on the ECG data

You asked to compare Zarr here. Same ragged ECG (264 MiB raw), Zarr v3, warm-cache. Amplification is
approximate (average compressed unit size times units touched).

| Store | Size | vs raw | K=100 amp | Full read | Objects | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Parquet BSS+zstd (current default, 4 MiB rg) | 122.1 MB | 0.46x | 46.5x | fast | shards | current on-disk reality |
| Parquet dictionary+zstd | 71.1 MB | 0.26x | (as rg) | fast | shards | best Parquet, keeps interop + Xet |
| Zarr CSR zstd, 256 KiB chunks | 74.2 MB | 0.28x | **3.6x** | 0.24 s | 1057 | lowest amplification |
| Zarr CSR zstd, 1 MiB chunks | 76.5 MB | 0.29x | 12.6x | 0.14 s | 265 | |
| Zarr CSR zstd, 4 MiB chunks | 73.4 MB | 0.28x | 27.6x | 0.06 s | 67 | |
| Zarr sharded (256 KiB in 8 MiB shards) | 74.2 MB | 0.28x | 34.3x (shard unit) | 0.18 s | 34 | cloud-native, few objects |
| Zarr CSR Blosc zstd + bitshuffle, 1 MiB | 181.5 MB | 0.69x | 29.8x | 0.13 s | 265 | shuffle hurts, like BSS |

Reading:

- **Zarr zstd (73-74 MB) ties dictionary-Parquet (71 MB)** and beats plain-Parquet (85 MB) and the current
  BSS store (122 MB). Zarr's zstd runs on one contiguous stream with no per-row list overhead.
- **Zarr has lower random-access amplification** at equal granularity (256 KiB: 3.6x vs Parquet 6.1x),
  because each chunk is an independent object and the CSR layout has no row structure. Its chunks map
  directly to object-storage GETs.
- **Bitshuffle confirms the mechanism:** the byte-transpose family (Blosc bitshuffle, like Parquet BSS)
  balloons quantized data to 181.5 MB. Do not use shuffle or BSS on quantized biosignals.
- **The format is not the lever here, the encoding is.** Moving Parquet from BSS to dictionary takes it
  122 MB to 71 MB and matches Zarr. Zarr's edge (amplification, chunk-per-object) does not offset losing
  directory interop (pandas/polars/DuckDB/Spark read the Parquet directory directly), the HuggingFace-Xet
  content-defined-chunking dedup path, and single-file-per-shard simplicity, and it fans out into ~1000
  chunk objects (or 34 shards).

## Recommendations (updated by the biosignal data)

1. **P0, memory-bounded reader row-group cache.** Replace fixed `_ROW_GROUP_CACHE_SIZE = 16`
   (`timenet.reader.reader`) with a byte-budgeted LRU. 11x on the ECG scattered read; free.
2. **P0/P1, choose the value encoding per dataset.** The hardcoded BSS default regresses real ECG 50%.
   Replace the "verify BSS was applied" self-check with a pass that writes the smallest of
   `{dictionary, plain, BSS}` per dataset. For quantized biosignals that is dictionary; for high-entropy
   series like tsqa it is BSS. Fix the `encodings.py` comment and the storage-decisions memory note.
3. **P1, change `DEFAULT_ROW_GROUP_TARGET_BYTES` from 4 MiB to 512 KiB**, kept per-dataset overridable
   (already threaded through the writer). On ECG that is ~7x faster at random reads (795 vs 5690 µs) and
   6.1x vs 24.8x amplification, for +5.9% size (dictionary) or +0.5% (BSS). Cost: a 128 MiB shard's footer
   grows to ~240 KB, past pyarrow's 64 KB open read, so a shard open is a second GET on S3 (trivial
   locally). Stay at 1-4 MiB for long recordings read mostly by scan; BSS datasets can go smaller for free.
   See [`benchmarks/REPORT.md`](benchmarks/REPORT.md) "Choosing a row-group size".
4. **P2, unchanged.** Optional timestamp column for irregular sampling; real S3 range reads +
   `pyarrow.dataset`; a random-access sidecar only if scattered by-id access is proven the dominant S3
   cost. Keep Parquet as the format; do not switch to Zarr or Lance.

## Reproduction

Everything here is produced by [`benchmarks/format_eval.py`](benchmarks/format_eval.py), a single
self-contained script (inline `uv` deps). See [`benchmarks/REPORT.md`](benchmarks/REPORT.md) for the
shareable write-up with plots.

```bash
./benchmarks/format_eval.py all --ecg-records 1200 --data-dir ./data --out ./results
```

`prepare` downloads both datasets (tsqa from the HuggingFace datasets-server parquet URL; PTB-XL records
from `physionet.org/files/ptb-xl/1.0.3/records500`), `run` benchmarks them, `plot` renders the figures.
A normal network connection is enough. Committed outputs live in
[`benchmarks/results/`](benchmarks/results/).
