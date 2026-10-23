# Verification: `chengsenwang/tsqa`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py` unless stated otherwise. The ☑️ Done once
checks are one-off passes run on 2026-10-23 over the Hub files listed under References and two
full builds of dataset version 1.2.0, read back through `TimeNet.load`. Their scripts aren't in the
repo yet (see Plan).

The connector has no README. The only prose about it is the card's description and the TSQA
paragraphs in `docs/connectors.md`, so Step 10 reviews those. The file `chengsenwang/tsqa.yaml`
beside the connector folder is a stale copy of `dataset.yaml` from the old flat layout. Discovery
only reads `*/dataset.yaml`, so no build uses it, but it lacks `source_url` and its comment
describes a layout that no longer exists.

## References

The connector reads `default/train/0000.parquet` from the Hub's `refs/convert/parquet` ref of
`ChengsenWang/TSQA`, which pointed at commit `22518209` on 2026-10-23. That file is 182,056,290
bytes with sha256 `3cb8a0db…e136`, equal to the Hub's LFS id and to the copy the build cached.
Independent copies and references, which the connector doesn't read:

- **The source CSV** on the Hub's `main` branch at commit `5f9565d4` (2024-12-17): `TSQA.csv`,
  266,444,635 bytes, sha256 `49759d77…fdce`. It was uploaded in commit `75de511e` on 2024-07-08,
  and every later commit only edits `README.md`. The parquet file is the Hub's automatic
  conversion of this CSV.
- **The Hub dataset card** at `5f9565d4`: the four features, three categories each, 12,000 rows per
  feature, lengths 64, 128, 256 and 512, generated with KernelSynth, and `license: apache-2.0`.
- **The authors' code**, ForestsKing/ChatTime at commit `8c2d4c20`: `dataset/TSQA.csv` (a one-row
  sample), `demo.ipynb`, which feeds `Question` and `Series` to `model.analyze`, and
  `utils/prompt.py`, which wraps the question in `ANA_SYS_PROMPT` and `ANA_INST_PROMPT_TEXT`.
- **The paper**, ChatTime, arXiv:2412.11376 (AAAI 2025): Section 3.4 and Appendix B.3 for the
  generation, Table 9 for the counts, Table 6 for accuracy. It fine-tunes on 25K randomly chosen
  rows and tests on the rest, without publishing the selection or a seed.

TSQA is synthetic. There is no upstream recording to compare with, and the KernelSynth parameters
that produced each series and label aren't published.

## Fields

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| Row position | Parquet row order | Record id `row-<i>`, task id `qa-<i>`, signal, source and annotation ids | The source has no id column. The rows are sorted by label, which no input carries (#52) |
| `Task` | Parquet | `task` record annotation, and the stratification key of the split | Four values: `Trend`, `Volatility`, `Seasonality`, `Outliers`. The Hub card calls the last two "Season" and "Outlier" |
| `Size` | Parquet | Dropped | Equals the series length in all 48,000 rows (#5). Nothing notes the drop |
| `Question` | Parquet | `AnswerTask.prompt`, verbatim | One fixed multiple-choice text per `Task` (#49) |
| `Answer` | Parquet | The task's only target, verbatim | `(a)`, `(b)` or `(c)` (#54) |
| `Label` | Parquet | `label` record annotation on the task's input Record | The answer as text. It maps one to one to `Answer` (#51) |
| `Series` | Parquet, a JSON list of floats | One float32 Signal `c0` per Record, on an `OrdinalAxis` | Every series is univariate and z-scored (#10, #19). float32 rounds every value (#14) |
| Prompt wrapper | `utils/prompt.py` in ChatTime | Not used | `ANA_SYS_PROMPT` and "Please answer the following question carefully after analyzing the sequence: {}". Nothing notes the drop |
| 25K fine-tuning selection | Paper | Not used | Random, and not published, so it can't be rebuilt (#43) |
| `.gitattributes`, `README.md` | Hub repo | Not read | Not data |
| Spec `tsqa_series`, "TSQA Series", dimensionless | Added | Signal spec | Dimensionless fits z-scored values (#19) |
| `OrdinalAxis` | Added | Time axis | The source has no timestamps or sample rate |
| Channel name `c0`, source name "TSQA series" | Added | Names | Formatting |
| Train, validation and test split | Added | `AnswerTask.split` | TimeNet's own 80/10/10 split, stated in the card's description (#43, #45) |
| Record, source, signal, annotation and task ids | Added | Ids | Built from the row position, so they follow the label order (#52) |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | There is one release. The paper and the ChatTime README point to `ChengsenWang/TSQA` on the Hub, whose data has not changed since commit `75de511e`. The parquet file the connector reads equals the CSV at `main` (`5f9565d4`) in all 48,000 rows and all six columns, with 0 mismatches. The one row in the authors' GitHub sample is Hub row 47,998. |
| 2 | Unpinned input | ⚠️ Open | The connector reads the `refs/convert/parquet` ref, which the Hub rewrites whenever it reconverts. Nothing pins a commit or a sha256. `test_real_download_reads_auto_converted_parquet` checks only that the ref name reaches `list_repo_files` and `hf_hub_download`. Fix: read `TSQA.csv` at commit `5f9565d4`, or pin the parquet commit, and check the sha256 above. |
| 3 | Partial, corrupt or changed download | ⚠️ Open | The connector checks no checksum, so a changed conversion would build without notice. One-off: the cached parquet file hashes to the Hub's LFS id `3cb8a0db…e136`. Closes with the pin of #2. |
| 4 | Wrong files from the archive | ☑️ Done once / ⚠️ Open | Done once: on 2026-10-23 the ref holds one parquet file, `default/train/0000.parquet`, beside `.gitattributes`. Open: the base reads every `*.parquet` file on the ref, so a new config or split would be added to the build silently. Needs a pinned inventory. |
| 5 | Source structure misread | ☑️ Done once | Over all 48,000 rows: 0 nulls in any column, `Size` equals the series length in every row, and every `Series` is a flat list of finite numbers. The connector's docstring describes multivariate series as lists of lists, but the source has none, so that branch never runs on real data. The Hub card's feature names differ from the `Task` values ("Season" for `Seasonality`, "Outlier" for `Outliers`). |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ☑️ Done once / ⚠️ Open | Done once: the build holds 48,000 Records and 48,000 tasks. Each `Task` has 12,000 rows, as paper Table 9 and the Hub card give. Each of the 12 labels has 4,000, and each `Task` has 3,000 rows of each length 64, 128, 256 and 512. Open: no test pins these counts (plan D). |
| 7 | Rows dropped silently | ✅ Done / ☑️ Done once | Done: `convert` makes one Record and one task per row with no filter, and a row that fails to parse raises instead of being skipped (`test_convert_builds_one_record_per_row`). Done once: the full build, read back, has Records `row-0` to `row-47999`, one for each source row. |
| 8 | Wrong filter or threshold | ☑️ Done once | The connector applies no filter, and neither the authors' code nor the paper describes one. 48,000 rows in, 48,000 Records out. |
| 9 | Duplicate records | ☑️ Done once | 48,000 distinct record ids and 48,000 distinct task ids. All 48,000 `Series` strings are distinct, and so are their parsed values. No two series share any 16-value window, and no two match on their first 32 values after z-scoring. |
| 10 | Missing or extra channels | ☑️ Done once | Every one of the 48,000 rows is univariate, and every Record read back has exactly one Signal, `c0`. The Hub card and the paper describe univariate series. |
| 11 | Source information left unused without a note | ☑️ Done once / ⚠️ Open | Done once: the Fields table lists every column and file of the Hub repo and the inputs of the authors' code. Open: with no README, nothing notes that `Size` is dropped, that the authors wrap the question in their own prompt, or that the paper's 25K split can't be rebuilt. |
| 12 | Information the connector made up | ☑️ Done once | The Fields table lists every added value. The dimensionless unit fits the z-scored values (#19), the ordinal axis fits a source with no time base, and the split is stated in the card's description. The rest are names and ids. |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ☑️ Done once | The full build, read back through `TimeNet.load`, against `json.loads` of the source parquet. All 48,000 Records and all 11,520,000 values equal the float32 of the source value, and every length matches. |
| 14 | Precision lost on a dtype change | ☑️ Done once / ⚠️ Open | **The check found a loss.** The source holds float64 values with 13 to 21 decimal digits, and the spec stores float32. None of the 11,520,000 values reads back equal at float64. The largest error is 4.75e-7 absolute and 5.96e-8 relative, the float32 rounding. ChatTime itself rounds to 4 decimals, so the loss is far below what the paper uses, but nothing documents it. Needs a decision: document it, or store float64. |
| 15 | Overflow or wraparound in integer conversion | ⚪ N/A | The series are floats, from −10.07 to 10.30. The only integer column, `Size`, is dropped. |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: `test_splits_survive_storage_round_trip` writes and reads a fixture build and compares every split. Done once: the full-build read-back compared values (#13), both annotations, and every task's prompt, target, input Record, split, input annotations and rationale. 0 mismatches over 48,000 Records and 48,000 tasks. |
| 17 | Build isn't deterministic | ✅ Done / ☑️ Done once | Done: split assignment repeats on fixtures (`test_split_assignments_are_repeatable_and_preserve_source_ids`). Done once: two full builds from the same cached file. The value file is byte-identical (sha256 `634d5355…a2cb`). All 16 `control.duckdb` tables hold the same rows, apart from the writer's generated UUIDs in `axes.axis_id` (48,000) and `annotation_occurrences.occurrence_id` (96,000). The manifests differ only in that file's checksum. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ⚪ N/A | The series are synthetic floats from KernelSynth. There is no instrument. |
| 19 | Wrong units or scale | ☑️ Done once | Every one of the 48,000 series is z-scored: the largest absolute mean is 3.9e-16, and the population standard deviation is 1 within 3e-16. So the values have no unit, and dimensionless is right. Nothing says the series are normalized (#59). |
| 20 | Swapped or mislabeled channels | ⚪ N/A | Every series has one channel (#10). |
| 21 | Flipped sign convention | ☑️ Done once | Direction is the one sign that matters here, and it holds. The least-squares slope over the index is positive in 4,000 of 4,000 "upward trend" series, negative in 4,000 of 4,000 "downward trend" series, and positive in 50.7% of "constant trend". The second half's detrended step size exceeds the first half's in 85.5% of "increased volatility" series, 14.7% of "decreased volatility" and 50.3% of "constant volatility". So the series are in time order, not reversed. |
| 22 | Missing-value markers stored as values | ☑️ Done once | 0 non-finite values in 11,520,000, and no series repeats any value, so no marker value exists. |
| 23 | Wrong categorical codebook | ☑️ Done once | `Task` and `Label` take the 4 features and 12 categories that the Hub card lists, with the naming differences of #5. Each label maps to one answer letter in all 48,000 rows, and that letter's option text in the question names the label: for example `upward trend` is `(b)` "This time series has an upward trend" in all 4,000 rows. |
| 24 | Implausible upstream values passed through without a note | ☑️ Done once | Z-scored values span −10.07 to 10.30. The largest come from spikes and level shifts, which the labels describe. No value is out of place for a z-scored series. |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ⚪ N/A | The source has no sample rate. The build uses an ordinal axis. |
| 26 | Wrong start time or epoch | ⚪ N/A | The source has no timestamps. |
| 27 | Wrong time zone | ⚪ N/A | The source has no timestamps. |
| 28 | Daylight-saving errors | ⚪ N/A | The source has no timestamps. |
| 29 | Gaps or missing values filled or dropped silently | ☑️ Done once | The source has no missing values: 0 non-finite values, and every series has exactly `Size` values. The read-back has the same lengths, so nothing was filled or dropped. |
| 30 | Artifacts from resampling or interpolation | ⚪ N/A | The source has no time base, and the connector doesn't resample. |
| 31 | Off-by-one sample | ☑️ Done once | The full-build read-back matches every value at its index, with every length equal (#13). |
| 32 | Sources or channels out of sync | ⚪ N/A | One channel per Record (#10). |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ⚪ N/A | The source is one table, and each row carries its own series, question, answer and label. There is nothing to join. |
| 34 | Off-by-one id | ✅ Done / ☑️ Done once | Done: each task's input is the Record of its own row (`test_split_assignments_are_repeatable_and_preserve_source_ids`). Done once: in the full build, task `qa-<i>` has Record `row-<i>` as its input, and row i's question and answer, for all 48,000 tasks. |
| 35 | Ids or join keys not unique | ☑️ Done once | 48,000 distinct record ids and 48,000 distinct task ids in the full build. |
| 36 | Subject metadata on the wrong recording | ⚪ N/A | The source has no subjects or metadata. |
| 37 | Join silently drops rows | ⚪ N/A | There is no join (#33). |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ☑️ Done once | There are no class codes. The target is the source's answer letter, copied, and #23 found it consistent with the label and the option text in all 48,000 rows. The authors' code has no mapping to compare with. |
| 39 | Label distribution differs from the paper | ☑️ Done once / ⚠️ Open | Done once: 4,000 rows per label, which fits Table 9's 12,000 per feature over 3 categories. Per split, the 12 labels hold 3,150–3,233 tasks in train, 376–428 in validation and 385–423 in test. Each `Task` holds exactly 9,600, 1,200 and 1,200. Open: the paper's own split can't be compared (#43), and no snapshot keeps these counts (plan D). |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | One label per series, as the paper and the Hub card describe each generated series. |
| 41 | Derived target computed wrongly | ☑️ Done once | The connector derives nothing: every target equals the row's `Answer`, for all 48,000 tasks. The authors' generator isn't published, so their labels can't be regenerated. #42 checks them against the series instead. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once | A gradient-boosted classifier per `Task` on 17 summary features of each series, trained on a seeded 80% and scored on the other 20% (2,400 per task). Trend 100.0%, Volatility 99.8%, Seasonality 97.2%, Outliers 100.0%. With shuffled train labels, it scores 30.1–35.5%, near the 33.3% of chance. Most Seasonality errors confuse "shifting" with "fixed". |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ⚠️ Open | The authors publish no split. The paper fine-tunes on 25K randomly chosen rows and tests on the rest, with no seed or list. The connector generates its own 80/10/10 split, 38,400 / 4,800 / 4,800 in the full build, and the card's description says it isn't the authors'. Results on it aren't comparable with paper Table 6. Needs a decision on whether to offer a split of the paper's sizes. |
| 44 | Leakage across splits | ✅ Done / ☑️ Done once | Done: tasks that share a series stay in one split (`test_generated_splits_balance_categories_and_keep_signals_together`, `test_equivalent_series_encodings_share_a_split`). Done once: all 48,000 series are distinct, so every group has one task and no series spans splits. No pair of series shares a 16-value window or matches after z-scoring (#9), so there are no cropped copies to leak either. |
| 45 | Splits regenerated without a fixed seed | ✅ Done | Seed 42 and ten folds are fixed in the connector, and scikit-learn is pinned at 1.9.1 in the package's dependencies, the version the build reports. Assignments repeat (`test_split_assignments_are_repeatable_and_preserve_source_ids`, and `test_seed_is_repeatable_and_does_not_change_global_random_state` in the package tests). The two full builds have identical `tasks` tables (#17). |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | The paper poses TSQA as question answering scored by accuracy, and the source gives a question and a letter answer per series. An `AnswerTask` with the question as prompt and the letter as target fits that. Reviewed against the paper and `demo.ipynb`. |
| 47 | Task can't be solved from its inputs | ☑️ Done once | The model check (#42) answers 97.2–100% from the series alone, so the series supports the answer. The question names the options, and the input Record holds the series. |
| 48 | A field in the wrong role | ⚠️ Open | **The check found a problem.** `Label` is the answer in words, yet it is a `label` annotation on the Record that is the task's input. The torch dataset returns a Record's annotations with every sample. It belongs with the target, or out of the Record. The other roles hold in the full build: the prompt is the question, the target is the answer, there are no input annotations and no rationale. |
| 49 | Wrong prompt, or one without a source | ✅ Done / ⚠️ Open | Done: the prompt is the row's `Question` and the target its `Answer` (`test_qa_tasks_match_fixture`). In the full build, all 48,000 prompts equal their source question. Open: the authors wrap the question in `ANA_SYS_PROMPT` and `ANA_INST_PROMPT_TEXT` (`utils/prompt.py`), which the build doesn't record. No golden rendered task is committed. |
| 50 | Series placed at the wrong point in the prompt | ☑️ Done once / ⚠️ Open | Done once: read four seeded tasks from a full build (`qa-29098`, `qa-18896`, `qa-33285`, `qa-5191`). Each has one input Record with one series, a prompt with no placeholder, and a letter target. The renderer places the series. Open: ChatTime puts the series under `### Input:` after the instruction, and nothing documents that. No review sheet with plots yet (plan G). |
| 51 | An input reveals the target | ⚠️ Open | **The check found a leak.** The prompt can't reveal it: each `Task` has one fixed question for all its 12,000 rows. But the input Record's `label` annotation names the answer, and it maps one to one to the answer letter in all 48,000 rows (#23). A model that reads the Record's annotations answers 100%. Fix: drop the annotation from the Record, or move it to the task's target annotations. |
| 52 | The target leaks through an id, metadata or another field | ☑️ Done once / ⚠️ Open | Done once, ids: the source rows are sorted by `Task`, then length, then label, in 48 runs of exactly 1,000 rows, and every id carries the row position (`row-<i>`, `qa-<i>`). That isn't a leak here: no input carries an id, evaluation shuffles samples, and the split is a seeded stratified split, not a contiguous range of ids (#45). Open: the torch dataset returns each input Record's `label` annotation with the sample, and it names the answer (#51). |
| 53 | An input carries information from after the target | ⚪ N/A | The source has no time and no events. Each answer describes the whole of its own series. |
| 54 | Answer format doesn't match what the evaluation expects | ⚠️ Open | Targets are the strings `(a)`, `(b)` and `(c)`, and the question asks "Only answer (a), (b), or (c)." The paper reports accuracy but doesn't say how a response is matched. TimeNet has no evaluation for this task. Each label always has the same letter, and the options never change order. Needs a dummy evaluation that parses the letter (plan G). |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ☑️ Done once | The card declares Apache-2.0, as the Hub card's `license: apache-2.0` metadata does. That metadata is the source's only statement: the Hub repo holds no license file, and the GitHub API reports no license for the ChatTime repository. |
| 56 | Wrong domain, tags or description | ✅ Done / ⚠️ Open | Done: the card validates (`test_every_connector_ships_a_valid_dataset_card` in the package tests). Open: neither the description nor the tags say the series are synthetic, from KernelSynth, or z-scored. The `huggingface` tag names a host. The stale `chengsenwang/tsqa.yaml` copy should go. |
| 57 | A README number that doesn't match the build | ☑️ Done once / ⚠️ Open | There is no README. Done once: the card's numbers hold. The split is 80/10/10 exactly (38,400 / 4,800 / 4,800), with ten folds, seed 42 and scikit-learn 1.9.1, the version in the build's environment. Open: no README gives the counts of #6, and no test ties prose numbers to the build (plan D). |
| 58 | A README claim the data doesn't support | ⚠️ Open | There is no README. The card and `docs/connectors.md` claim the split keeps identical series together, which holds but does nothing here, since all 48,000 series are distinct (#44). The connector's docstring describes multivariate series, which the source doesn't have (#5). |
| 59 | Known limitations left undocumented | ⚠️ Open | Nothing documents: the label annotation leak (#51), the ids sorted by label (#52), the float32 rounding (#14), the z-scoring (#19), that the data is synthetic, the fixed option order (#54), the dropped `Size` and prompt wrapper (#11), or that the paper's split can't be rebuilt (#43). |

## Open items

Most important first.

1. **The answer leaks through the input Record (#48, #51, #52).** The `label` annotation on the
   input Record states the answer in words, one to one with the letter, and the torch dataset
   returns it with every sample. Fix: drop it, or attach it as a target annotation of the task.
2. **Unpinned input (#2, #3, #4).** The connector reads a ref the Hub rewrites, checks no
   checksum, and reads every parquet file it finds. Fix: read `TSQA.csv` at `main` commit
   `5f9565d4` with its sha256 `49759d77…fdce`, or pin the conversion commit `22518209` and the
   parquet sha256, with a one-file inventory.
3. **A README (#11, #57, #58, #59).** Write one that says the data is synthetic and z-scored,
   gives the counts, the dropped `Size`, the authors' prompt wrapper and where ChatTime places the
   series, the float32 rounding, the ids sorted by label, and the annotation leak until it is
   fixed. Fix the card's description
   and tags (#56), drop the multivariate wording from the docstring (#5), and delete the stale
   `chengsenwang/tsqa.yaml`.
4. **float32 storage (#14).** Every value rounds by up to 6e-8 relative. Decision: document it, or
   store float64.
5. **The paper's split (#43).** It can't be rebuilt. Decision: keep TimeNet's 80/10/10 split as the
   only one, or also offer a seeded 25K / 23K split of the paper's sizes, labeled as not the
   authors' selection.
6. **Answer format and option order (#54).** No evaluation parses the letter yet, and each label
   always has the same letter. Plan G.
7. **Numbers that can drift (#6, #39).** Plan D.
8. **Repeatable checks on the real data.** Plans A1, A2, B1, C1 and E ran once. Commit them so they
   run again.

## Plan

Checks on the real data run in a `real_data` pytest marker, skipped unless an environment variable
points at the downloaded Hub files. Every check below except D, F and G ran once for this record.

- **A1. CSV against parquet.** Read `TSQA.csv` at `5f9565d4` and the parquet conversion, and
  compare every row and column. Check both sha256s against the pins of open item 2.
- **A2. Read-back.** Read a full build through `TimeNet.load`, and compare every value with the
  float32 of the source, every annotation, and every task's prompt, target, input and split.
- **B1. Source invariants.** `Size` equals the length, every series is univariate, finite and
  z-scored, every series is distinct, and each label maps to one letter whose option text names it.
- **C1. Leakage outside the inputs.** Check that no input Record annotation determines the
  target, and that no input carries an id. After open item 1 is fixed, a model given the Record's
  annotations must score at chance.
- **D. Stats snapshot.** Commit a `verification.json` with the counts per `Task`, label and length,
  per split. A test regenerates it from the build and compares.
- **E. Model check.** Rerun the per-`Task` classifier against shuffled labels, and the trend and
  volatility direction checks.
- **F. Corruption tests.** Reverse each series, and move every label one record over. Each must
  fail E.
- **G. Review sheet and evaluation.** Render 12 seeded tasks, three per `Task`, with the series
  plotted, and once in the authors' prompt template. Run a dummy evaluation that parses the letter
  from a response and scores accuracy.
