# End-to-end regression suite

This directory contains a deterministic offline corpus and two commit-to-commit gates:

- `noop.py` fails if the logical TimeF dataset changes at all. It compares metadata, schema, tasks,
  samples, annotations, series contracts, dtypes, shapes, and exact value bytes while deliberately
  ignoring backend-specific physical layouts.
- `benchmark.py` fails if the median conversion, write, read, total, or stored-size measurement gets
  worse. Its default permitted regression is zero; use `--max-regression-percent` when a noisy shared
  runner needs an explicit budget.

Both commands create isolated Git worktrees and run each revision with its own locked `uv` environment.
The suite is intentionally placed before the values-backend stack. It detects each revision's
capabilities: legacy Parquet is the reference for a newly introduced Zarr backend, and rich dtype/N-D
cases run only when both revisions can represent that corpus.

```bash
uv run python -m benchmarks.end_to_end.noop BASELINE CURRENT
uv run python -m benchmarks.end_to_end.benchmark BASELINE CURRENT
```

The default matrix includes portable float32 scalar data under Parquet and Zarr with small and large
chunks, plus a rich Zarr run with float64, int16, uint16, and N-D values. The eight domain scenarios
cover vibration, 12-lead ECG, sleep physiology, accelerometry, finance/TSQA, workout telemetry, energy,
and automotive degradation. Their layouts also mimic existing connectors:

- ECG-QA: long 12-lead values reused across multiple reasoning samples;
- TSQA: many independent short one-to-three-channel question-answer samples;
- test-mean: many tiny labeled series;
- hello-world: deterministic closed-form values, annotations, task variety, and stable ids.

Increase `--scale` for more value data. The benchmark defaults to five measured repetitions after one
warm-up and alternates revision order to reduce cache and thermal bias.
