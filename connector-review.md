# Connector review and implementation stack

This review covers the proposed SLIP, HEARTS, ARFBench, and VerbalTS connectors.
The source revisions below are pinned. A newer upstream release needs a deliberate
connector update; it must not silently change an existing TimeF dataset.

## What each dataset provides

| Dataset | What it provides | How it is published |
| --- | --- | --- |
| [SLIP](https://huggingface.co/datasets/LeoChen085/SlipDataset) | Sensor windows with four natural-language captions for pretraining; eleven separate labelled evaluation collections. | Hugging Face Parquet shards. The training shards have nested `time_series` values and `meta.csv` for source rates. Evaluation shards have nested `X`, labels, prompts, and split-specific files. Revision `e5e4871a9f376ff5377ed16c598e7a84a160664a`. |
| [HEARTS](https://huggingface.co/datasets/yang-ai-lab/HEARTS) | 1,455 frozen health-reasoning cases in 30 task directories: glucose, respiration, heart rate, audio, meal photographs, forecasting, imputation, and symptoms-only questions. | One Python pickle per case under five source-corpus folders. Some cases contain pandas frames, audio arrays, or JPEG bytes. Revision `7c18df521ae36cbc6b61e17782f1ac08dc378ea1`. |
| [ARFBench](https://huggingface.co/datasets/Datadog/ARFBench) | 750 incident questions about monitoring metrics, with numerical series at several actual resolutions and 414 published charts. | A question CSV, metric Parquet files, and chart PNGs on Hugging Face. Revision `1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a`. |
| [VerbalTS](https://github.com/seqml/VerbalTS) | Six collections of time-series windows paired with captions or instructions. They cover training, validation, and test splits. | A Google Drive release with 54 NumPy arrays and six metadata tables. The connector pins each of the 60 file IDs, sizes, and checksums. |

## What conversion must do

SLIP should be two registry datasets: `leochen085/slip-train` and
`leochen085/slip-eval`. They share the source pin, Hub fetcher, and nested Parquet
reader. Training produces one sensor record per row and a separate caption task
for each caption. Evaluation produces one sensor record and classification task
per labelled window, retaining component, split, participant, and prompt data.
The source often gives no reliable clock for evaluation windows, so their axes
are ordinal rather than invented sampling rates. Sensor units are unknown unless
the source states them.

HEARTS needs all 30 directories, not the 21 handled by the proposal. A small
declarative table states the task type, prompt, visible payload fields, and
answer handling for each directory. A restricted pickle reader loads only the
globals used by the pinned release. Input series retain real or ordinal axes;
photographs become image signals in Zarr. Forecast and imputation answers live
in separate target-only records so they cannot leak into the input. A
symptoms-only task uses a prompt and dataset annotations without a record.
Timestamped frames share one source clock and keep their gaps and relative
positions. Imputation targets use `mask_indices` to select input timestamps.
The pinned [imputation case][hearts-imputation-case] confirms that these are
original frame row labels, not zero-based positions. The connector resolves
them against `window_df.index` before it selects the timestamps.
Forecast targets lack individual timestamps. They use the one-minute cadence
described by the source harness, starting at the supplied meal time.
The pinned imputation and forecasting `0.pkl` cases both pass a direct
conversion with the release-compatible pandas 2 environment.
The release mixes upstream licences and does not provide a single reusable
licence for every corpus. Check source terms before redistribution.

ARFBench must keep the metric values at *every published resolution*. Values
at different resolutions are not guaranteed to be simple resamples of the
finest series. Store each metric/resolution once, then make a question task for
each resolution common to all metrics in that question. The task points to
the existing records; it does not duplicate their values. Make a separate
image-input task for each question with a chart. Use Zarr for the decoded PNG
signals. This produces 3,848 numerical tasks and 750 chart tasks from the
pinned release, while reusing the metric and chart records. Metric timestamps
retain their exact UTC positions against a shared fixed origin. Their physical
units are unknown.

VerbalTS should download the pinned files with `gdown`, check their sizes and
SHA-256 digests, and memory-map the NumPy arrays. It makes a record for each
window and a generation task whose text prompt describes the desired output.
The BlindWays collection does not publish a defensible sampling rate, so it
uses an ordinal time axis. A generation task that has only a prompt declares
`TEXT` and `NO_INPUT`. The released arrays and metadata do not map channel
positions or `var_id` codes to the physical names in the paper. Signals keep
positional names and preserve the original `var_id` annotations.

## Shared code and format choices

The proposals repeat pinned Hub downloads, image decoding, file verification,
stable IDs, lazy value loaders, split/provenance annotations, and task creation.
Shared helpers are useful only where the semantics match. The implementation
adds one pinned-Hub helper, one image-to-signal helper, and a SLIP-specific
nested-Parquet reader. VerbalTS's Drive manifest and checksum rules remain in
its connector: making a generic download framework for one dataset would add
more configuration than it removes. Dataset-specific prompt and column maps
stay as data tables, rather than a hierarchy of adapter classes.
Core span checks also live in one independent module. Source, Record, and task
validation import it directly, without a lazy import or circular dependency.

Images as binary annotations would need a new annotation value type, a storage
layout, and consumer changes. Zarr already supports tensor signals, so decoded
RGB and RGBA images with an ordinal axis are the smaller complete implementation. For
tasks that mix a question, chart, and series, `input_modalities` states what
the model actually receives. It is a required set of enum values, including
`TEXT`, `TIME_SERIES`, `IMAGE`, `AUDIO`, and `NO_INPUT`. Prompt-only tasks use
`NO_INPUT`; `None` and empty declarations are invalid. The control table stores
the set, and both SQL and in-memory task readers can filter it before loading
large record sets. This is a breaking TimeF control-schema change; old task
tables need rebuilding, not an inferred modality.

Each Source has a `TimeOrigin`. Sources share a timeline only when they refer to
the same object. DuckDB stores one `BIGINT` clock row per object and each Source
stores its clock ID. The reader restores shared references. `Record.start_time`
is now a derived summary of the earliest timed sample, not the origin for all
signals. One span cannot cover independent clocks. A timezone-less date remains
local provenance; the connector does not assign UTC to it. The revised format
uses manifest version 3 and control schema version 4.

`TimeSeriesSpec.unit_value=None` now means that the source does not establish a
unit. It differs from `ureg.dimensionless`, which means that the values have a
known dimensionless interpretation. The field remains required. SLIP and
ARFBench use unknown units where the source gives no per-channel or per-metric
unit.

## Remaining source assumptions

- HEARTS forecasts provide thirty values and a meal time, but no per-value
  timestamps. The connector uses the source harness's one-minute forecast
  cadence. This spacing cannot be checked against supplied target timestamps.
- Some HEARTS frames contain local dates without a timezone. Their shared
  relative clock preserves differences between those dates. Their absolute UTC
  origin remains unknown.
- ARFBench uses a fixed UTC day boundary as a reversible coordinate zero. The
  metric Parquet timestamps, not that boundary, determine sample positions.
- The VerbalTS paper names physical variables, but the released NPY arrays and
  the [authors' loader](https://github.com/seqml/VerbalTS/blob/main/data/data.py)
  give no channel-order map or `var_id` codebook. Positional names remain until
  a source mapping can be verified.

## Commit-by-commit stack

Each numbered item is one functional change and one pull request. The order
keeps shared format work below the dataset connectors.

1. **Required input kinds.** Add the modality enum and `NO_INPUT`. Require a
   non-empty declaration on every task. Store it in control-schema version
   three, with no legacy inference. Update existing task producers.
2. **Task filtering.** Filter in-memory tasks, stored task rows, and target
   rows by required and supported input kinds before record hydration.
3. **Shared span checks.** Move common span validation out of Record so Source,
   Record, and task validation can import it without a cycle.
4. **Shared source clocks.** Give Sources reference-based origins, store clock
   IDs in DuckDB, derive Record summaries, and reject mixed-clock spans.
5. **Unknown units.** Store a required unit as null when its meaning is unknown.
   Preserve the distinction from dimensionless values through every reader.
6. **Pinned Hub sources.** Share helpers that list and download files from a
   fixed Hugging Face dataset revision.
7. **Image signals.** Convert image files or bytes to RGB or RGBA tensor
   signals with an ordinal axis.
8. **SLIP training.** Map the training Parquet shards to sensor records and
   four caption tasks per row. Keep source metadata and lazy signal loaders.
9. **SLIP evaluation.** Add its own registry card and connector. Reuse the
   source pin and nested-Parquet reader, retaining all eleven components and
   their native splits and labels.
10. **ARFBench.** Add shared metric/resolution records, tasks at every common
   resolution, and chart-input tasks with Zarr images.
11. **VerbalTS.** Pin all Drive files, verify them, and convert the six
   components with memory-mapped arrays and explicit prompt-only modalities.
12. **HEARTS source handling.** Add its restricted pickle reader and input
   signal adapters, including source clocks and decoded photographs.
13. **HEARTS task map.** Declare all 30 task directories, their visible inputs,
   prompts, and answer types in one table.
14. **HEARTS integration.** Download the pinned release, create held-out
   sequence targets with source timing, and support recordless symptoms.
15. **Integration and documentation.** Run core and isolated connector checks;
    document source terms, schema breakage, and the conversion decisions.

The source manifests, task prompts, and integrity data account for many lines
of the final diff. Those are declarative facts about the releases, not
independent conversion engines. Removing them would lose coverage or
reproducibility, not simplify the model.

[hearts-imputation-case]: https://huggingface.co/datasets/yang-ai-lab/HEARTS/blob/7c18df521ae36cbc6b61e17782f1ac08dc378ea1/cgmacros/non_meal_imputation_cgm_only/0.pkl
