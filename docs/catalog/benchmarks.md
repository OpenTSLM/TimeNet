---
icon: lucide/gauge
description: "Benchmarks built on TimeNet datasets."
tags:
  - catalog
  - benchmarks
---

# Benchmarks

A TimeNet benchmark measures two things that a generic task leaderboard does not test.

First, it compares TimeF's storage and read performance to raw formats. Examples are Pandas and
PyTorch, which read raw files directly. The comparison covers storage size, time to first item, full
read time, and throughput.

Second, it uses the shared format to pool datasets from different tasks. You can then train one
model across all of them.

!!! planned "Planned"
    This is not built yet. The benchmark suite and results are still in progress.
