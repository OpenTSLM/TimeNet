# End-to-end regression suite

This directory contains a deterministic offline corpus and two commit-to-commit gates:

- `noop.py` fails if the logical TimeF dataset changes at all. It compares table row counts, tasks,
  records, annotations, source trees, signal contracts, dtypes, shapes, and exact value bytes while
  deliberately ignoring backend-specific physical layouts. `values_artifacts` and `signal_chunks` are
  left out of the counts: how many files a backend wrote and how it split a signal are exactly what a
  backend is free to decide.
- `benchmark.py` fails if the median conversion, write, read, total, or stored-size measurement gets
  worse. Its default permitted regression is zero; use `--max-regression-percent` when a noisy shared
  runner needs an explicit budget.

Both commands create isolated Git worktrees and run each revision with its own locked `uv` environment.
The suite drives the public writer and reader only, so it survives a change of storage layer. It
detects each revision's capabilities first: Parquet is the reference for a newly introduced Zarr
backend, and a `rich` case runs only when both revisions' `TimeSeriesSpec` accepts `dtype` and
`value_shape`.

```bash
uv run python -m benchmarks.end_to_end.noop BASELINE CURRENT
uv run python -m benchmarks.end_to_end.benchmark BASELINE CURRENT
```

The matrix is six cases: the `portable` corpus under Parquet and under Zarr, each at small and large
chunk budgets, plus one `rich` case per backend at a middling budget. The Parquet and Zarr cases
share a fingerprint on purpose. The two backends lay bytes out differently and the digest covers only
the logical content, so a matching digest is what says they are interchangeable.

One corpus feeds every case. It is four records, each holding a bedside monitor source with an ECG
source of three float32 leads and an auxiliary source with an irregular float32 temperature; two of
the four also carry a shared float32 series, which must be stored once and linked twice. On top of
that there are three tasks, one of them with two ordered inputs, and annotations at the dataset,
task, record, source, and signal levels. The `rich` profile adds one float64 series on an ordinal
axis per record. The shapes are chosen for what has broken a writer before: a source tree deeper than
one level, a series shared by two records, an irregular axis, a constant-valued lead, and a task with
two inputs.

Increase `--scale` for more value data. The benchmark defaults to five measured repetitions after one
warm-up and alternates revision order to reduce cache and thermal bias.
