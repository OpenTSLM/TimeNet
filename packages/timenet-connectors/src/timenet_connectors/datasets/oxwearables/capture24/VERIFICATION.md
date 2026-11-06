# CAPTURE-24 verification

On 2026-11-06, the connector built the full cached extracted release and the resulting TimeF
files passed `TimeFReader.verify()`. The run included named-axis loading and the separate windowing module. It preceded the
extraction of shared CSV-table utilities; that migration was checked with the shared utility and
connector regression suites, without repeating the full build.

## Input and method

The input was the existing `~/.cache/timenet/cache/oxwearables/capture24/capture24` extraction,
containing all 151 participant recordings. No source files were modified. The original archive
was not downloaded or checksummed again; this run verifies conversion and storage of the cached
extraction, not its correspondence to a newly downloaded archive.

The run called `Capture24Connector().convert([raw])`, stored the result with
`LocalRegistry(output).store(dataset)`, opened the stored version, and called `reader.verify()`.
The output was a fresh registry at `/private/tmp/capture24-local-refactor-build`, so storage could
not reuse an existing version. Output files and measurement logs are outside the repository.

## Results

| Measurement | Result |
| --- | --- |
| Records | 151 |
| Sources | 159 |
| Signals | 477 |
| Classification tasks | 1,869,524 |
| Windows, with two tasks each | 934,762 |
| Annotation contents / occurrences | 513 / 13,902 |
| Conversion | 90.19 s |
| Storage, including the second source read | 738.68 s |
| File verification | 0.67 s |
| Total | 829.55 s |
| Peak process resident memory | 2,360,197,120 bytes (2.20 GiB) |

Times use `time.monotonic()`. Peak memory uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`, which
reports bytes on this machine (macOS 26.6, arm64). Other repository tests ran concurrently during
part of the build. These are observed timings, not an isolated performance benchmark.

File verification checks stored file sizes and checksums. It does not independently establish
benchmark correctness. Synthetic tests cover window selection, labels, splits, gaps, channel
values, diagnostics, lazy loading, and a verified storage round-trip.

## Repository checks

- `make check` and `make lint-fix`: passed.
- `make test`: 2,607 passed, one skipped, including the 18 shared CSV tests.
- CAPTURE-24: all 60 tests passed.
- `make test-connectors`: all populated suites and type checks passed. The runner still exits
  nonzero for seven unrelated empty test directories: `hayleyzhou1113/veritime`,
  `mikeam/time_series_reasoning`, `pandalin98/enginemt_qa`, `physionet/mitdb`, `time_ra/rats40k`,
  `trace/weather_events`, and `zenodo/code_15`.
