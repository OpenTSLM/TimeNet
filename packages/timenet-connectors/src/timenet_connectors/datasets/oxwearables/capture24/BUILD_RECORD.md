# CAPTURE-24 build record

On 2026-11-06, an earlier version of the connector built all 151 participant recordings from a
cached extracted release. This is a historical conversion and storage measurement, not a
validation of the current revision or an independent comparison with the authors' benchmark.
The exact tested commit was not recorded here.

## Input and method

No source files were modified. The original archive was not downloaded or checksummed again,
so this run does not establish the cached extraction's correspondence to the pinned archive.

The run called `Capture24Connector().convert([raw])`, stored the result with
`LocalRegistry(output).store(dataset)` in a fresh registry, opened the stored version, and
called `reader.verify()`. The fresh registry prevented reuse of an existing stored version.
Output files and measurement logs were kept outside the repository.

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

A full build of the current revision and an independent comparison of every window's participant,
time, scheme, and target against the authors' implementation remain outstanding.
