---
name: add-dataset-connector
description: Use when adding a new TimeNet dataset connector, i.e. converting an external dataset (from a HuggingFace repo, PhysioNet, or another source given by a link or reference) into the TimeF format. Guides exploring the source, choosing the task type and example-row shape, confirming the plan, then implementing the connector folder that reuses the HuggingFace or PhysioNet base connectors.
---

# Adding a dataset connector

A connector fetches a dataset's raw source and converts it into a `TimeFDataset`. It implements the
`BaseConnector` contract (in the `timenet` package) and lives in `timenet-connectors`. The engine drives
it `download -> convert -> derive_schema -> store`; `timenet-build build <id>` runs that pipeline and
writes the result into a registry.

Reuse a base connector wherever the source allows: `BaseHuggingFaceConnector` for Hub datasets,
`BasePhysioNetConnector` for PhysioNet/WFDB records. Write a raw `BaseConnector` subclass only when
neither fits.

Read `references/connector-anatomy.md` for the contract, the base-connector APIs, the task types, the
folder layout, and a fully worked example before implementing.

## Workflow

Create one task per step. Do the exploration and design first, get sign-off at the gate, then build.

### 1. Take the input and derive the id
Start from the link or reference the user gives (a HuggingFace dataset URL, a PhysioNet page, etc.).
Derive an `org/name` id: lowercase, hyphens allowed in the leaf. On disk hyphens become underscores, so
`physionet/ecg-qa-cot` maps to `datasets/physionet/ecg_qa_cot/`. The id shape is validated as
`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`, and no segment may start with `.`.

### 2. Explore the source structure
Understand the raw data before designing anything:
- Fields/columns and their types; which one holds the series values and how they are encoded (JSON
  list, list-of-lists for multivariate, WFDB record, etc.).
- Series shape: univariate vs multivariate, number of channels, length.
- Sampling rate and physical units (Hz, seconds, mV, dimensionless).
- Labels, questions/answers, rationales, splits, and any per-row metadata.

For HuggingFace, the Hub auto-converts public datasets to parquet on the `refs/convert/parquet` branch;
inspect columns via the dataset viewer or the datasets-server API. For PhysioNet, inspect the archive
layout and a WFDB header (`fs`, `sig_len`, `sig_name`).

### 3. Pick the base connector
- Hub dataset: subclass `BaseHuggingFaceConnector`, set `HF_REPO`, inherit `download`, implement only
  `convert`.
- PhysioNet/WFDB: subclass `BasePhysioNetConnector`, implement `download` (use `ensure_archive` /
  `fetch_files` from `timenet_connectors.download`) and `convert` (use `_read_header` / `_lead_loader`).
- Neither: subclass `BaseConnector[TRaw]` and implement `download` + `convert` yourself.

### 4. Determine the task and sketch an example row
- Choose one built-in `Task` by the *kind of answer* the dataset supervises: `ClassificationTask` (a
  category, whole-sample or over a `scope`), `AnswerTask` (free text; a caption without a `prompt`),
  `ScalarPredictionTask` (a number with a unit), `TemporalLocalizationTask` (regions to find),
  `ForecastingTask`, `TSEditingTask`, `TSGenerationTask`, or `TSCorrespondenceTask`. Any of them can carry
  a `rationale`, so a chain-of-thought dataset is not a separate type.
- Sketch one `Sample`: its `TimeSeries` channel(s) with their `spec` and `time_axis`;
  the annotations you'll attach; and the task payload. Concrete values, not placeholders.

### 5. HARD GATE: confirm with the user

**STOP. Do not write any connector code yet.** Present, and wait for explicit approval:
- the proposed `org/name` id,
- the source and the base connector you'll reuse,
- the task type, and
- a concrete example-row sketch (channels + units + annotations + task payload).

Only continue once the user confirms. If they change the task or shape, revise the sketch and re-confirm.

### 6. Implement the connector folder
Create `packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/` with:
- `dataset.yaml`, the dataset card: `yaml_schema_version`, `dataset_id`, `dataset_version`, `name`,
  `description`, `license`, `domains`, `tags`.
- `connector.py`, the `BaseConnector` subclass, ending with a module-level `CONNECTOR = <YourClass>`.
- `__init__.py`, re-exporting `CONNECTOR` (and the class) from `connector.py`.
- `requirements.txt`, when the connector needs a library outside `timenet-connectors`' core
  dependencies. Import it lazily inside the connector. If it is missing, raise a clear error. Curation
  installs it into the environment that the build runs in. The lazy import keeps `--no-isolation`
  usable while you write the connector.

Add the org namespace `__init__.py` if the org is new. Keep `download` I/O-only and `convert` CPU-only
with lazy value loaders (never materialize arrays in `convert`). See the worked example in
`references/connector-anatomy.md`.

### 7. Verify
- Add a fixture-based test in `<org>/<name>/tests/`, mirroring the one at
  `datasets/chengsenwang/tsqa/tests/test_connector.py`: check in a tiny sample of the raw shape and call
  `convert()` on it directly (no network). The `TIMENET_TESTING` / `TIMENET_ROW_LIMIT` env vars mentioned
  in some docs are **not implemented**, so don't rely on them.
- If you added or changed a `requirements.txt`, re-run `make sync` so the new library lands in your own
  environment. Otherwise `ty` reports your lazy import as unresolved and the connector test can't run.
- Round-trip end to end: `uv run timenet-curate build <id> --out <tmp-dir>`, then
  `TimeNet(registry="<tmp-dir>").load("<id>").describe()`.
- Run `make check` and `make test`, and state which checks you ran (per AGENTS.md).

## Further reading

`references/connector-anatomy.md` in this skill. Repo docs: `docs/connectors.md` and `docs/build.md`
(design proposal; where they describe a flat-file `datasets/<org>/<name>.py` layout, the real code uses
the folder-package layout above, so follow the code).
