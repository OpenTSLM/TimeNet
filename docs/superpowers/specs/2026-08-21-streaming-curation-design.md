# Streaming curation: recording-as-sample + task-referenced annotations

Status: approved design, implementing
Owner: connectors/engine/writer

## Problem

`physionet/ecg-qa-cot` is ~231k chain-of-thought QA rows over ~16k PTB-XL
recordings. (The CoT CSVs carry multi-line quoted fields, so a `wc -l` count of
~5.3M lines badly overstates the row count.) The connector builds **one sample
per QA row**, so the pipeline materializes ~231k samples + ~1M annotations +
~231k tasks (each with a long CoT rationale) + a ~2.8M-row (sample × series)
index, and grinds for 20+ min building those objects before the write starts.
The deeper problem is the model, not the count: a question is a task on a
recording, not a recording of its own. The perf fixes already landed (direct
WFDB reads, O(N) validation, source-grouped series sort) cut CPU/write time but
not the object explosion.

## Model change

- **Sample = one ECG recording** (~16k), carrying its 12 leads. Samples, series,
  index, and recording annotations are all bounded and built normally.
- **Task = one QA** (`AnswerTask`: prompt=question, target=answer, rationale=CoT),
  referencing its recording's `sample_id`. ~231k tasks, streamed.
- Samples 231k→16k, index 2.8M→~192k. Measured: full ingest 20+ min heading to
  OOM → 63 s at 581 MB peak.

## QA metadata → deduped, task-referenced annotations

Tasks are fixed typed targets by design (the type resolves against a fixed
registry, not the manifest), so metadata does not belong as task fields.
Annotations are the format's metadata mechanism, and a task already references
them via `input_annotation_ids`.

- `question_type`, `template_id`, `answer_options`, `clinical_context` become
  **value-deduped annotations**: one annotation per distinct value (a handful of
  question types, ~70 templates, ~70 option sets). A few hundred rows total,
  independent of QA count.
- Each `AnswerTask` references the relevant ones by id.
- **Writer change (general):** persist annotations referenced by a task's
  `input_annotation_ids`/`target_annotation_ids` even when no sample carries
  them. This removes the sample-attachment coupling (no ballooning `sample_ids`
  back-references) and keeps tasks pure typed targets.

## Streaming pipeline

The ~231k tasks and CSV rows never all live in memory:

- `download` fetches sources, returns lightweight handles (paths), not a ref per row.
- `convert` builds the bounded recording dataset (samples, series, recording +
  metadata annotations); no tasks.
- a streaming task method yields `AnswerTask`s from a streaming CSV read.
- the writer streams task shards (`write_sharded_table` already consumes rows
  lazily), peeking the first task for its type + id storage, tallying counts and
  file parts for the manifest. Tasks need no global sort, so this is clean.
- All three registry backends (local/remote/s3) wrap the same
  `TimeFWriter(...).write()`, so the streaming writer serves every publish path.

## Out of scope (separate, later)

The general **external-sort writer** for datasets with huge *sample* counts
(the in-memory sorts of the samples/index/annotation tables are the real limit
there). With recording-as-sample every sorted table is bounded, so it is not
needed here. Parallelizing convert (Ray/ProcessPool, the earlier spec) is also
independent.

## Increments (each verifiable; `make check` + `make test` green at each)

1. Dataset: task-referenced annotation registry (sample-free, deduped by id).
2. Writer: persist task-referenced annotations; validate task annotation refs.
3. Writer + dataset: streaming tasks (re-iterable source; peek-first for type/id).
4. `BaseConnector`: opt-in streaming contract; engine drives it.
5. ecg-qa-cot remodel: recording-as-sample, QA-as-task, deduped metadata annots.
6. End-to-end ingest to dev under bounded memory; load from a fresh install.
