---
icon: lucide/gauge
description: "Benchmarks built on TimeNet datasets."
tags:
  - catalog
  - benchmarks
---

# Benchmarks

Benchmarks will pair TimeNet datasets with standard tasks and metrics. You can then compare models fairly.

!!! planned "Planned"
    This is not built yet. The benchmark suite and results are still in progress.

## Inheritance read performance

The inheritance implementation uses independent child catalogs and shared PTB-XL payloads.
These local measurements compare it with main at `00656c9f51b2`.

| Operation | Main | Inheritance | Change in elapsed time |
| --- | ---: | ---: | ---: |
| Complete materialized dataset | 10.04 s | 8.02 s | −20.1% |
| 64 random signal windows | 0.882 s | 0.887 s | +0.5% |
| 64 training items, batches of 16 | 0.514 s | 0.547 s | +6.4% |

Each number is the median of three interleaved runs in fresh Python processes with a warm filesystem
cache. The machine has an Apple M5 Max, 18 CPU cores, and 64 GiB of memory.
The dataset contains 16,024 ECG records, 192,288 signals, 961,440,000 values, and 231,543 tasks.
The parent contains all 21,799 PTB-XL records. Both versions use the standard Parquet group size.

The full-read measurement retains every signal array and all task text. Main's lazy read includes
explicit array materialization. The new eager reader uses four workers for required Parquet groups.
All waveform and task-text hashes match. Random-window and training-output hashes also match.
These measurements do not measure network latency or remote throughput.

The build also completed against cached source data. The initial full PTB-XL build took 204 seconds.
The child build took 53 seconds and wrote no waveform files. These write timings are single runs.

The scripts and raw measurements are in `benchmarks/inheritance/`. Run each script from its matching
checkout against the corresponding compiled artifact:

```bash
uv run python benchmarks/inheritance/read.py /path/to/version
uv run python benchmarks/inheritance/access.py /path/to/version
```

The writer computes scan order and the required group list. Eager reads consume this plan with
bounded parallel decoding. Lazy handles and range reads do not prefetch unrelated groups.
