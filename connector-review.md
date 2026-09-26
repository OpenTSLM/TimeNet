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
are ordinal rather than invented sampling rates.

HEARTS needs all 30 directories, not the 21 handled by the proposal. A small
declarative table states the task type, prompt, visible payload fields, and
answer handling for each directory. A restricted pickle reader loads only the
globals used by the pinned release. Input series retain real or ordinal axes;
photographs become image signals in Zarr. Forecast and imputation answers live
in separate target-only records so they cannot leak into the input. A
symptoms-only task uses a prompt and dataset annotations without a record.
The release mixes upstream licences and does not provide a single reusable
licence for every corpus. Check source terms before redistribution.

ARFBench must keep the metric values at *every published resolution*. Values
at different resolutions are not guaranteed to be simple resamples of the
finest series. Store each metric/resolution once, then make a question task for
each resolution common to all metrics in that question. The task points to
the existing records; it does not duplicate their values. Make a separate
image-input task for each question with a chart. Use Zarr for the decoded PNG
signals. This produces 3,848 numerical tasks and 750 chart tasks from the
pinned release, while reusing the metric and chart records.

VerbalTS should download the pinned files with `gdown`, check their sizes and
SHA-256 digests, and memory-map the NumPy arrays. It makes a record for each
window and a generation task whose text prompt describes the desired output.
The BlindWays collection does not publish a defensible sampling rate, so it
uses an ordinal time axis. A generation task that has only a prompt declares
`TEXT` and `NO_INPUT`.

## Shared code and format choices

The proposals repeat pinned Hub downloads, image decoding, file verification,
stable IDs, lazy value loaders, split/provenance annotations, and task creation.
Shared helpers are useful only where the semantics match. The implementation
adds one pinned-Hub helper, one image-to-signal helper, and a SLIP-specific
nested-Parquet reader. VerbalTS's Drive manifest and checksum rules remain in
its connector: making a generic download framework for one dataset would add
more configuration than it removes. Dataset-specific prompt and column maps
stay as data tables, rather than a hierarchy of adapter classes.

Images as binary annotations would need a new annotation value type, a storage
layout, and consumer changes. Zarr already supports tensor signals, so decoded
RGB images with an ordinal axis are the smaller complete implementation. For
tasks that mix a question, chart, and series, `input_modalities` states what
the model actually receives. It is a required set of enum values, including
`TEXT`, `TIME_SERIES`, `IMAGE`, `AUDIO`, and `NO_INPUT`. Prompt-only tasks use
`NO_INPUT`; `None` and empty declarations are invalid. The control table stores
the set, and both SQL and in-memory task readers can filter it before loading
large record sets. This is a breaking TimeF control-schema change; old task
tables need rebuilding, not an inferred modality.

## Commit-by-commit stack

Each numbered item is one functional change and one pull request. The order
keeps shared format work below the dataset connectors.

1. **Input foundation.** Add the modality enum and `NO_INPUT`. Require a
   non-empty declaration on every task. Persist and filter it in control-schema
   version three, with no legacy inference. Add the small Hub and image helpers
   needed by the connectors. These changes are one commit because the earlier
   partial commits did not pass checks on their own.
2. **SLIP training.** Map the training Parquet shards to sensor records and
   four caption tasks per row. Keep source metadata and lazy signal loaders.
3. **SLIP evaluation.** Add its own registry card and connector. Reuse the
   source pin and nested-Parquet reader, retaining all eleven components and
   their native splits and labels.
4. **ARFBench.** Add shared metric/resolution records, tasks at every common
   resolution, and chart-input tasks with Zarr images.
5. **VerbalTS.** Pin all Drive files, verify them, and convert the six
   components with memory-mapped arrays and explicit prompt-only modalities.
6. **HEARTS source handling.** Add its restricted pickle reader and input
    signal adapters, including decoded photographs.
7. **HEARTS task map.** Declare all 30 task directories, their visible inputs,
    prompts, and answer types in one table.
8. **HEARTS integration.** Download the pinned release, create held-out
    sequence targets, and support recordless symptoms.
9. **Integration and documentation.** Run core and isolated connector checks;
    document source terms, schema breakage, and the conversion decisions.

The source manifests, task prompts, and integrity data account for many lines
of the final diff. Those are declarative facts about the releases, not
independent conversion engines. Removing them would lose coverage or
reproducibility, not simplify the model.
