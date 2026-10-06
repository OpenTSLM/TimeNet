# HEARTS dataset layers

Each original dataset is a standalone connector. Each HEARTS child selects original records,
applies benchmark transformations, and adds the published tasks.

The final HEARTS connector imports the five children. It does not download raw files or copy signal
values.

| Original dataset | Benchmark child |
| --- | --- |
| `physionet/cgmacros` | `yang-ai-lab/hearts-cgmacros` |
| `iiscleap/coswara` | `yang-ai-lab/hearts-coswara` |
| `epfl/coughvid` | `yang-ai-lab/hearts-coughvid` |
| `oca-john/harespod` | `yang-ai-lab/hearts-harespod` |
| `cstr/vctk` | `yang-ai-lab/hearts-vctk` |

All five children are parents of `yang-ai-lab/hearts`. Dataset cards pin each parent to an exact
version. The build records manifest checksums for the complete dependency graph.

## Build

The source downloads total approximately 28 GB. Extracted files and TimeF outputs need additional
disk space. All sources are public. The build installs connector dependencies in isolated
environments.

Use a new registry directory for this layout. Existing HEARTS datasets at version `1.0.0` are not
rebuilt automatically.

Build the complete benchmark:

```bash
uv run timenet-build build yang-ai-lab/hearts --out ./registry
```

Build one child and its original parent:

```bash
uv run timenet-build build yang-ai-lab/hearts-cgmacros --out ./registry
```

Build an original dataset without HEARTS tasks:

```bash
uv run timenet-build build physionet/cgmacros --out ./registry
```

## Transformations and limits

- CGMacros windows select exact timestamps. Imputation inputs mask glucose values. Targets use the
  original unmasked values. HEARTS meal photographs remain benchmark assets because they differ
  from the original photographs.
- Coswara keeps native audio in the parent. The child makes mono copies and checks symptoms against
  participant metadata. Empty source recordings retain metadata without invented samples.
- COUGHVID keeps native decoded audio in the parent. The child converts WebM and Ogg samples to
  PCM16 before mono conversion. Its comparison permits one PCM16 step for decoder rounding.
- HARESPOD retains all five channels from the complete continuous release. Ranking cases select
  timestamps. Pairing cases require one matching window because the benchmark resets their clocks.
  Missing or ambiguous matches fail the build. The separate incomplete release is not included.
- VCTK retains both microphones. HEARTS omits microphone identifiers. A verified mapping selects
  the original microphone for each pinned case. The child resamples with soxr HQ and reverses the
  selected cases. Comparisons allow float32 rounding differences.

Children compare derived values with the pinned HEARTS release. A failed comparison stops the
build. The connector does not replace a failed derivation with the frozen waveform.

Original dataset cards state their source licenses. HEARTS does not declare a benchmark-wide
license, so the children and aggregate use `other` with a link to the release.
