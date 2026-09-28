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
Frames with source-backed alignment share one Record timeline and keep their gaps and relative
positions. Unaligned recordings become separate task inputs, each with an unknown absolute origin.
Imputation targets use `mask_indices` to select input timestamps.
The pinned [imputation case][hearts-imputation-case] confirms that these are
original frame row labels, not zero-based positions. The connector resolves
them against `window_df.index` before it selects the timestamps.
Forecast targets lack individual timestamps. They use the one-minute cadence
described by the source harness, starting at the supplied meal time.
One pinned `0.pkl` case from each of the 30 directories passes conversion and a Zarr/DuckDB
round trip with pandas 2. The result has 47 Records and 55 Signals. Every value and Signal span
survives the round trip. Forecast and imputation targets remain separate, with shared Record origins.
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
adds pinned-Hub, image-to-signal, and exact relative-axis helpers, plus a SLIP-specific
nested-Parquet reader. Timestamp subtraction requires an explicit origin. Regular-axis detection
preserves gaps in a published cadence. These helpers reuse NumPy and Fraction, with no new clock library.
VerbalTS's Drive manifest and checksum rules remain in
its connector: making a generic download framework for one dataset would add
more configuration than it removes. Dataset-specific prompt and column maps
stay as data tables, rather than a hierarchy of adapter classes.
Core span checks also live in one independent module. Signal, Source, Record, and task
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

Each Record has a `TimeOrigin`. All timed Signals and annotations within that Record use one
relative timeline. Connectors establish alignment before they create the Record. The origin's
timestamp is optional: known alignment does not require a known absolute date.

Related Records can share the same origin object, including `TimeOrigin(None)`. Separate unknown
origins do not imply alignment. DuckDB stores one BIGINT clock row per origin and each Record stores
its clock ID. The reader restores shared Python references. Sources only group Signals.

Regular axes store `offset_us`, an exact period, and `start_index`. Slicing preserves fractional
sampling phase. Irregular axes retain explicit offsets. Ordinal axes do not invent elapsed time.
The format uses manifest version 3 and control schema version 8.

Annotation ownership defines scope: Signal, Source subtree, or Record. A selection creates one
occurrence per selected Signal with shared content. Annotation spans have no Signal selectors or
clock references. Task time spans must identify one input Record, optionally through Signal IDs.
The former source-clock selector proposal, PR #392, is superseded.

`TimeSeriesSpec.unit_value=None` now means that the source does not establish a
unit. It differs from `ureg.dimensionless`, which means that the values have a
known dimensionless interpretation. The field remains required. SLIP and
ARFBench use unknown units where the source gives no per-channel or per-metric
unit.

## Remaining source assumptions

- HEARTS forecasts provide thirty values and a meal time, but no per-value
  timestamps. The connector uses the source harness's one-minute forecast
  cadence. This spacing cannot be checked against supplied target timestamps.
- Related CGMacros reference and window frames contain local dates without a timezone. Their
  Record preserves differences between those dates. Their absolute UTC origin remains unknown.
  Independent frames and audio recordings are not synchronized by matching lengths or cadences.
- ARFBench uses a fixed UTC day boundary as a reversible coordinate zero. The
  metric Parquet timestamps, not that boundary, determine sample positions.
- The VerbalTS paper names physical variables, but the released NPY arrays and
  the [authors' loader](https://github.com/seqml/VerbalTS/blob/main/data/data.py)
  give no channel-order map or `var_id` codebook. Positional names remain until
  a source mapping can be verified.

## Commit-by-commit stack

Each numbered item is one functional change and one pull request. The order
keeps shared format work below the dataset connectors.

| Order | PR | Change |
| --- | --- | --- |
| 1 | #377 | Require input modalities, including explicit `NO_INPUT`. No legacy inference. |
| 2 | #386 | Filter tasks and targets before Record hydration. |
| 3 | #391 | Share span validation without import cycles. |
| 4 | #393 | Add regular-axis offsets and preserve exact slice phase. |
| 5 | #389, rewritten | Store Record origins and shared references. Keep unknown absolute time as `None`. |
| 6 | #394 | Make annotation ownership define scope. Migrate existing connectors and schema discovery. |
| 7 | #390 | Distinguish unknown units from dimensionless values. |
| 8 | #387 | Share pinned Hugging Face helpers in `huggingface_hub.py`. |
| 9 | #388 | Store RGB and RGBA images as ordinal tensor Signals. |
| 10 | #395 | Share explicit origin subtraction and exact axis detection. |
| 11 | #378 | Convert SLIP training with shared Parquet readers and lazy values. |
| 12 | #379 | Convert SLIP evaluation as a separate registry dataset. |
| 13 | #380, updated | Preserve every ARFBench resolution, task reuse, and chart. Move anchors to Records. |
| 14 | #381 | Convert all six VerbalTS components with pinned files and source-backed cadence. |
| 15 | #382, rewritten | Build one HEARTS Record per established timeline. Preserve independent inputs. |
| 16 | #383, updated | Map all 30 task families, including tasks with several independent input Records. |
| 17 | #384, updated | Preserve target alignment, image inputs, and recordless symptoms. |
| 18 | #385, updated | Document the final model, assumptions, stack, and verification. |

PR #392 is removed from the stack and closed as superseded. All remaining PRs stay Draft.
Each PR contains one commit. Connector changes stay in the PR that owns them.

Verification covers shifted grids, fractional-rate slicing, shared unknown origins, independent
origins, constructor annotation checks, nested schema discovery, and task scope rejection.
The source-level HEARTS run covers one case per directory, not all 1,455 released cases.

The source manifests, task prompts, and integrity data account for many lines
of the final diff. Those are declarative facts about the releases, not
independent conversion engines. Removing them would lose coverage or
reproducibility, not simplify the model.

[hearts-imputation-case]: https://huggingface.co/datasets/yang-ai-lab/HEARTS/blob/7c18df521ae36cbc6b61e17782f1ac08dc378ea1/cgmacros/non_meal_imputation_cgm_only/0.pkl
