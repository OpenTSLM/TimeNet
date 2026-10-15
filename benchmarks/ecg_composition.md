# ECG-QA composition: conversion, reads, and fidelity

The ECG-QA child preserves the standalone connector's consumer data and shares the stored PTB-XL
waveforms. Its write stage is 8.2 times faster, and its own artifact is 87.2% smaller. This
implementation also increases SDK load time and peak memory.

The comparison uses the existing `physionet/ecg-qa-cot` connector, including its OpenTSLM
rationales. The baseline is commit `34e2b55f0d62a610be8176bb7d3e3f4b9e289f7f`, immediately before
the child change. Both variants use the same TimeF format, writer, reader, Parquet backend, and
content-method API. Thus, the comparison isolates the connector's use of parent payloads.

## Data and environment

The complete cached sources contain 21,799 PTB-XL recordings and 231,543 ECG-QA questions. The
questions reference 16,024 distinct recordings, all present in PTB-XL 1.0.3. Each selected recording
has 12 leads with 5,000 float32 samples at 500 Hz, in millivolts. The comparison includes 192,288
signals and 961,440,000 samples. It does not use a row limit.

The machine was an Apple M5 Max with 64 GiB RAM and macOS 26.6.2. The environment used Python
3.13.15, DuckDB 1.5.5, PyArrow 24.0.0, NumPy 2.5.0, and WFDB 4.3.1. The sources were already local,
so these measurements exclude network downloads and archive extraction. The artifact sizes use
decimal MB and GB.

The public source descriptions are [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and
[ECG-QA](https://github.com/Jwoo5/ecg-qa).

## Conversion and writing

Each row reports one full conversion in a fresh process. The parent and child rows have separate
costs.

| Measurement | Standalone ECG-QA | Complete PTB-XL parent | ECG-QA child |
| --- | ---: | ---: | ---: |
| Record count | 16,024 | 21,799 | 16,024 |
| Convert catalog | 6.83 s | 7.92 s | 14.68 s |
| Derive schema | 0.10 s | 0.35 s | 0.10 s |
| Write artifact | 151.00 s | 183.96 s | 18.39 s |
| Own artifact size | 990.32 MB | 1,197.20 MB | 127.16 MB |
| Own DuckDB catalog | 117.98 MB | 25.18 MB | 127.15 MB |
| Own waveform bytes | 872.33 MB | 1,172.01 MB | **0** |
| Peak process memory | 2.25 GB | 2.02 GB | 3.41 GB |

The child made **zero inherited payload reads** during conversion, schema derivation, and writing.
Its manifest lists one physical owner: `physionet/ptb-xl@1.0.0`, pinned by the parent manifest
checksum. It has no waveform shards. The parent contains all recordings, including 5,775 recordings
absent from ECG-QA.

With an existing parent, the complete child pipeline takes 33.17 seconds versus 157.93 seconds for
the standalone connector. The child catalog is larger because it includes lineage and foreign
payload references. Catalog import also temporarily holds the parent objects and detached child
objects, which increases peak memory.

Without an existing parent, the two write stages total 202.35 seconds, and the two artifacts occupy
1.324 GB. This first build costs more than a standalone ECG-QA build. Later children can reuse the
parent without another waveform copy. The zero-copy benefit applies to storage and write work. It
does not imply faster reads or lower catalog memory.

A second full child conversion with the checked-in benchmark script took 15.39 seconds to convert
and 18.00 seconds to write. It also recorded zero inherited reads and zero owned waveform bytes.

## Consumer reads

The SDK call was `TimeNet(...).load(..., auto_build=False)` against each local registry. It hydrated
all records and tasks while waveform values stayed lazy. Each subsequent scan materialized the
content and computed its fidelity fingerprint.

The table reports medians from three alternating runs per variant, each in a fresh process. The
filesystem cache was warm. No repository checks ran during these six measurements. An earlier read
pair also passed fidelity checks, but its timings are excluded because core tests ran concurrently.

| Measurement | Standalone ECG-QA | ECG-QA child | Change |
| --- | ---: | ---: | ---: |
| SDK load | 11.11 s | 15.19 s | 36.8% slower |
| Read and hash every waveform | 8.61 s | 10.69 s | 24.1% slower |
| Read and hash all task content | 8.82 s | 8.93 s | 1.2% slower |
| Peak process memory | 2.57 GB | 3.19 GB | 23.9% higher |

The waveform scan includes Arrow reads, NumPy conversion, null-mask reads, and SHA-256 computation.
The task scan includes prompts, targets, rationales, annotation values, and canonical JSON encoding.
These are end-to-end scan timings, rather than isolated storage bandwidth measurements.

The child stores more catalog information and reads a subset of the parent's larger waveform layout.
This benchmark does not isolate the cost of each factor. It does not measure cold-cache or remote
reads. The read overhead is a real limitation of this implementation and a useful target for later
profiling.

## Fidelity

All four complete read pairs produced the same three SHA-256 fingerprints: record catalog, waveform
content, and task content. The checked-in [measurements](ecg_composition_results.json) include the
fingerprints and individual timings.

The catalog comparison covers recording and source IDs, hierarchy, order, annotations, metadata,
task memberships, and schema. It compares signal IDs, names, counts, specifications, units, cadence,
offsets, and record start times. It also compares which signals share a time axis. The waveform
comparison includes every float32 byte and every null mask. There are no null samples. The task
comparison covers every ID, prompt, answer, rationale, ordered input reference, annotation, scope,
split, modality declaration, and task metadata.

Generated axis IDs and annotation occurrence IDs differ across independent conversions, so the
comparison excludes those IDs. It checks axis-sharing groups and compares task annotation references
through their attached content. Stable recording, source, signal, annotation content, and task IDs
remain unchanged.

The intended differences are the child release version (`1.0.0` to `1.1.0`), declared build input,
lineage fields, physical payload ownership, manifests, and artifact checksums. All other descriptive
dataset metadata matches. The child replaces parent clinical annotations with ECG-QA's original
split annotations to preserve the consumer view. The parent retains its clinical annotations. Task
splits remain unset, as in the standalone connector.

## Reproduce

Use the cached source layout from the standalone ECG-QA connector:

```text
<cache>/
  answers_for_each_template.csv
  ecg_qa_cot/
    ecg_qa_cot_train.csv
    ecg_qa_cot_val.csv
    ecg_qa_cot_test.csv
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ptbxl_database.csv
    records500/
```

Run the build phases in an empty output directory. Each command starts a separate Python process.
The script reads the baseline connector and card with `sl cat`. It does not change the checkout or
raw cache.

```sh
ECG_BENCH_CACHE="$HOME/.cache/timenet/cache/physionet/ecg-qa-cot"
ECG_BENCH_WORK="$(mktemp -d /tmp/timenet-ecg-comparison.XXXXXX)"

for variant in before parent after; do
  uv run python benchmarks/ecg_composition.py build \
    --variant "$variant" --cache "$ECG_BENCH_CACHE" \
    --work "$ECG_BENCH_WORK" \
    --output "$ECG_BENCH_WORK/$variant-build.json"
done
```

Run the reads after the builds finish. Avoid other CPU-intensive work during the timing runs.

```sh
for run in 1 2 3; do
  for variant in before after; do
    uv run python benchmarks/ecg_composition.py read \
      --variant "$variant" --work "$ECG_BENCH_WORK" \
      --output "$ECG_BENCH_WORK/$variant-read-$run.json"
  done
  uv run python benchmarks/ecg_composition.py compare \
    --work "$ECG_BENCH_WORK" \
    --before "$ECG_BENCH_WORK/before-read-$run.json" \
    --after "$ECG_BENCH_WORK/after-read-$run.json" \
    --output "$ECG_BENCH_WORK/fidelity-$run.json"
done
```

The comparison fails if any logical fingerprint, entity count, or descriptive metadata field
differs. Timing results are observations. CI does not impose performance thresholds or download
these datasets.
