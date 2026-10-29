# Verification: `hayleyzhou1113/veritime`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py`. The ☑️ Done once checks are one-off passes run
on 2026-10-29. They ran over the four pinned files and two full builds of dataset version 1.0.0,
which were read back through `TimeNet.load`. Each pass parsed the source with the standard `json`
module and its own code, which shares nothing with the connector's reader. The pass scripts aren't
in the repository yet (see Plan).

## References

The connector reads four files from `HayleyZhou1113/VeriTime` at commit
`fc88cb1305a96a208093e13b69d766b19c84c99b`:

| File | Rows | sha256 |
| --- | ---: | --- |
| `Dataset/Scenario/scenario_shuffled_train.jsonl` | 2,059 | `3234d8a6…68da` |
| `Dataset/Scenario/scenario_shuffled_test.jsonl` | 461 | `40cc9c96…478b` |
| `Dataset/Knowledge/knowldege_shuffled_train.jsonl` | 1,142 | `d1a638bb…6394` |
| `Dataset/Knowledge/knowledge_test.jsonl` | 694 | `5503c1d7…e2f` |

Independent copies and references, which the connector doesn't read:

- **The authors' repository**, `anonymous.4open.science/r/VeriTime-E017`, which the paper cites.
  Its archive downloaded on 2026-10-29 has sha256 `233e6710…5c69d9`.
  - `Dataset/` holds the same four files with four more columns: `id`, `cot_deepseekr1`, the
    unrounded or undecimated `timeseries`, and, on 400 knowledge test rows, `timeseries3`.
  - `TSRBench/` holds the curation scripts.
  - `VeriTime/` holds `VeriTime_sft.py`, `VeriTime_grpo.py` and `VeriTime_evaluate.py`, which
    build the model input and score the answers.
- **ChatTS**, `ChatTSRepo/ChatTS-Training-Dataset` at `1e2ae56a`. A scenario row's `id` is its
  line in `sft/train.jsonl`.
- **The paper**, arXiv:2602.07830. Table 9 gives the counts and Appendix C.1 the splits.

The knowledge sources upstream of the authors' "TimeBed" data aren't read: PhysioNet/CinC 2017
for ECG, and the CTU, EMG and right-whale datasets. No knowledge value is compared with them.

## Fields

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| `task` | Every row | The task kind, and `source_task` in task metadata | Checked against the file's family (#5) |
| `question` | Every row | `prompt`, verbatim, with its `<ts><ts/>` marker | The authors' user message, without the series (#49) |
| `output` | Scenario rows | `source_answer` task annotation | ChatTS's answer, which is supervision and not input (#48) |
| `label` | Every row | Target: a class name, an option text, or a number | As published (#41) |
| `cot` | Every row | `rationale`, verbatim | The authors' SFT target (#54) |
| `step1_label`, `step2_label`, `step4_label`, `step6_label` | Every row | Task annotations of the same names, when not null | The process labels the rewards score (#48) |
| `timeseries2` | Every row | The values of the Record's one signal | What the authors' scripts paste into the question (#13) |
| `id` | Repository copy only | Not in the build | The Hugging Face release removed it, and the connector reads that release (#11) |
| `cot_deepseekr1` | Repository copy, knowledge test only | Not in the build | The raw model response, from before `generate_cot.py` (#11) |
| `timeseries`, `timeseries3` | Repository copy | Not in the build | The originals the authors decimated, which the model never sees; used once to find the rates (#25) |
| `SYSTEM_PROMPT` | `VeriTime_sft.py` and the other two scripts | `request.SYSTEM_PROMPT`, not stored | The authors' system message (#49) |
| Class vocabularies | Each task's question options; `extract_label.py` for Yes/No | `<task>_vocabulary` dataset annotations, each its tasks' `target_schema` | Option order a), b), … (#38) |
| Specs, signal names, source names | Added | Signal spec and names | Assumption, named in the README |
| EMG unit `millivolt` | The EMG question | Spec unit | The question states "amplitude (mV)" (#19) |
| ECG 150 Hz, RCW 250 Hz axes | Added | Time axis | From the decimation on the test rows; assumed for train (#25) |
| CTU 120 s axis | The CTU question | Time axis | "sampled every 2 minutes" (#25) |
| Scenario and EMG ordinal axes | Added | Time axis | No rate stated |
| `veritime` record metadata | Added | `kind` and the `rows` that publish the series | Provenance |
| `part`, `row` task metadata | Added | The row a task comes from | Provenance |
| `source_repository`, `source_revision` | Added | Dataset annotations | The pinned release |
| Record and task ids | Added | Ids | Numbered by TimeNet when the dataset is built. The connector sets none |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | The dataset card cites the paper. The four pinned files equal the repository's `Dataset/` on every shared field of all 4,356 rows, in the same order, so both copies are one release. The Hugging Face copy is the later one: its history has "fix invalid json" and "remove extra columns". |
| 2 | Unpinned input | ✅ Done | `SOURCE_REVISION` is a 40-character commit, which `BaseHuggingFaceConnector` enforces when the class is defined, and `test_declares_pinned_source_files` pins it. |
| 3 | Partial, corrupt or changed download | ✅ Done / ⚠️ Open | Done: every build parses every row, and checks the rows per file and task and the distinct series per family (`test_pinned_inventory_mismatch_fails`, `test_a_family_series_count_is_checked_only_with_both_files`). Open: no file checksum is checked, so a change that keeps the counts would pass. The sha256 of each file is in References. |
| 4 | Wrong files from the archive | ✅ Done | `SOURCE_FILES` names the four files, and `test_declares_pinned_source_files` pins them. The base rejects an undeclared part. Besides them, the revision holds only the README, a figure, `.gitattributes` and `.DS_Store` files. |
| 5 | Source structure misread | ✅ Done / ☑️ Done once | Done: a strict row model validates every row on every build: its fields, both series formats, one marker, and the task's family (`test_malformed_rows_have_source_context`). Done once: the release matches the repository copy (#1). The repository copy's one corrupt line, a stray `c` before a knowledge train row, is fixed in the release. A file that isn't JSON values fails with its row (`test_a_file_that_is_not_json_values_fails_with_its_row`). |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ☑️ Done once | The release has 4,356 rows (3,201 train, 1,155 test). Paper Table 9 has 4,340, and its per-task counts differ from the release's. The paper's train total of 3,201 matches. The README lists both. |
| 7 | Rows dropped silently | ✅ Done / ☑️ Done once | Done: every row becomes one task. The build checks the rows per task (`test_pinned_inventory_mismatch_fails`), and the round trip has one task per fixture row (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: the full build has 4,356 tasks. |
| 8 | Wrong filter or threshold | ✅ Done | The connector filters nothing. The authors filtered before the release, and the README describes their scripts. `test_storage_round_trip_keeps_the_whole_conversion` maps every row to a task. |
| 9 | Duplicate records | ✅ Done / ☑️ Done once | Done: rows group into Records by a hash of their float64 values. The build checks the distinct series per family, and the round trip shares one Record between rows of both splits (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: the full build has 3,095 Records. Repeated rows are kept as tasks: 3 ChatTS lines twice in train, and 18 repeated (series, question) pairs. The README documents them. |
| 10 | Missing or extra channels | ✅ Done | A row has one series. A scenario row with two lists fails (`test_malformed_rows_have_source_context`). |
| 11 | Source information left unused without a note | ☑️ Done once | The Fields table lists every field of both copies, and every input the authors' scripts read. All release fields are used. The four repository-only columns are listed with the reason they are dropped. |
| 12 | Information the connector made up | ☑️ Done once / ⚠️ Open | Done once: the Fields table lists every added value with its reference. Open: the train rows' ECG and RCW rates are assumed from the test rows (#25). |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once | Done: a reread checks its values against the converted row's hash (`test_a_reread_that_no_longer_matches_its_row_fails`). The round trip reads through CRLF line ends, UTF-8 text and 17-byte reads (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: over the full build, every task's Record values equal an independent parse of its row, 4,356 of 4,356. |
| 14 | Precision lost on a dtype change | ☑️ Done once | Values are JSON decimals with at most four decimals, stored as float64. Read back, all 4,356 series equal the source exactly. The authors' text formatting of them rebuilds their message exactly (#50). |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | The only integer conversion is the inferential calculation count, and its largest value is 726. Signal values stay floats. |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: `test_storage_round_trip_keeps_the_whole_conversion` compares every field before and after storage. Done once: the full build read back matched every value, prompt, rationale, target, step label and split of all 4,356 tasks. |
| 17 | Build isn't deterministic | ☑️ Done once | Two builds: the `time_series/` and `payloads/` Parquet files are byte-identical. All 18 tables of `control.duckdb` hold equal rows. The DuckDB file's bytes, and so the manifest's checksum of it, differ. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ⚪ N/A | The release publishes decimals that were already scaled or synthetic, not ADC counts. |
| 19 | Wrong units or scale | ☑️ Done once / ⚠️ Open | Done once, by range: EMG −7.3 to 3.25 (mV per its question). ECG −6.45 to 5.90. CTU z-scores: every series has mean 0 and standard deviation 0.9993. RCW −10 to 9.9997. Open: the ECG and RCW units are unknown, and are left unknown until checked against the upstream data. |
| 20 | Swapped or mislabeled channels | ✅ Done | One channel per Record. Its kind follows the row's task, and each knowledge kind's fixed length is checked on every build (`test_a_knowledge_series_of_another_length_fails`). |
| 21 | Flipped sign convention | ⚪ N/A / ⚠️ Open | N/A for the synthetic scenario series. Open for ECG, EMG and RCW: not compared with the upstream recordings. |
| 22 | Missing-value markers stored as values | ✅ Done / ☑️ Done once | Done: the parser accepts only finite numbers (`test_malformed_rows_have_source_context`). Done once: no null or NaN in any row. 227 scenario rows publish a constant series, as ChatTS made them. 4 RCW series reach exactly −10.0 (16 values), likely clipping. The README documents both. |
| 23 | Wrong categorical codebook | ✅ Done / ☑️ Done once | Done: a label outside its task's vocabulary fails (`test_malformed_rows_have_source_context`). Done once: every ECG, CTU and RCW question is identical within its task. EMG has two, which differ by one space. The vocabularies are their options in order. |
| 24 | Implausible upstream values passed through without a note | ☑️ Done once | Ranges per kind are as in #19. Scenario values reach −3.2e8 and 7.3e8 in synthetic series. The README notes the constant series, the RCW clipping and the z-scored CTU. |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ☑️ Done once / ⚠️ Open | Done once: on all 200 ECG test rows, `timeseries2` is every second sample of the 1,000-sample original (150 Hz from the stated 300 Hz). On all 200 RCW test rows, it is every eighth of 4,000 (250 Hz from 2 kHz). The 144 CTU and 150 EMG test rows equal their originals. `test_series_kinds_state_the_published_rate_and_length` pins the axes. Open: the train files keep no original, so their rates are assumed from equal lengths. |
| 26 | Wrong start time or epoch | ⚪ N/A | The release publishes no timestamps. Some ChatTS questions name a start date, which stays in the prompt. |
| 27 | Wrong time zone | ⚪ N/A | No timestamps. |
| 28 | Daylight-saving errors | ⚪ N/A | No timestamps. |
| 29 | Gaps or missing values filled or dropped silently | ✅ Done | The connector fills nothing. A null or non-finite value fails the row (`test_malformed_rows_have_source_context`). |
| 30 | Artifacts from resampling or interpolation | ☑️ Done once | The connector doesn't resample. The authors' decimation keeps original samples unfiltered (#25), so RCW content above 125 Hz aliases. The README notes it. |
| 31 | Off-by-one sample | ☑️ Done once | The decimated test series start at the original's first sample, offset 0, in all 400 ECG and RCW test rows. |
| 32 | Sources or channels out of sync | ⚪ N/A | One signal per Record. |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ✅ Done | Tasks reach their Record by the row's byte range, not by order, and Records group rows by value hash. `test_storage_round_trip_keeps_the_whole_conversion` checks each Record's listed rows against the tasks. |
| 34 | Off-by-one id | ☑️ Done once | Over the full build, each task's `part` and `row` name the raw row whose question, label, `cot`, step labels and series equal the task's: 4,356 of 4,356. |
| 35 | Ids or join keys not unique | ✅ Done | A row's key is its file and position, which are unique. Series group on (kind, 128-bit hash), and the build checks the distinct series count (`test_pinned_inventory_mismatch_fails`). |
| 36 | Subject metadata on the wrong recording | ⚪ N/A | The release has no subjects. |
| 37 | Join silently drops rows | ✅ Done / ☑️ Done once | Done: every row is listed in its Record's `rows` (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: 4,356 of 4,356 in the full build. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ☑️ Done once | There are no codes: the authors compare label strings. The vocabularies are each question's option order (#23), and `Yes`/`No` as `extract_label.py` extracts them. |
| 39 | Label distribution differs from the paper | ⚠️ Open | The paper gives no class balance, and its task counts differ (#6). Per split, train / test: Anomaly detection Yes 706/137, No 310/43. ECG AF 152/47, NSR 149/47, noise 142/58, other 141/48. EMG neuropathy 155/76, myopathy 113/63, healthy 32/11. CTU laptop 81/92, desktop 50/52. RCW absent 65/132, present 62/68. |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | One question per row, as the paper and the authors' scripts use it. |
| 41 | Derived target computed wrongly | ✅ Done / ☑️ Done once | Targets are copied, except that an inferential calculation label is parsed as a number, and a non-number fails (`test_an_inferential_calculation_label_must_be_a_number`). Done once: every target equals its label in all 4,356 tasks. The one decimal label, train row 1125, is the amplitude of a two-part question. |
| 42 | Label misaligned with, or contradicted by, its signal | ⚠️ Open | Not checked. The labels come from ChatTS answers and the knowledge datasets, and the CoTs come from DeepSeek-R1. One CoT, train row 1782, says it saw no series. |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ☑️ Done once | Each file is one split, as the dataset card's configs define it. Train has 3,201 rows, as in the paper. |
| 44 | Leakage across splits | ☑️ Done once / ⚠️ Open | Done once: 241 scenario series are in both splits. 326 of the 461 scenario test rows share a series with train rows, and 3 (series, question) pairs are in both. The README documents this. No knowledge series is in both splits. Open: the knowledge rows name no patient or recording, so leakage between subjects can't be checked. |
| 45 | Splits regenerated without a fixed seed | ✅ Done | Nothing is regenerated: a task's split is its file's (`test_storage_round_trip_keeps_the_whole_conversion`). |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | Mapped from the paper's answer formats and agreed with the maintainer. True/false and fixed classes are classification, per-question options are answer, and counting is scalar prediction. |
| 47 | Task can't be solved from its inputs | ⚠️ Open | Each target comes from the question and the series. No baseline was run. The paper's ECG accuracies near 25% for four classes, and RCW's aliasing (#30), suggest that some knowledge tasks are barely solvable from the published series. |
| 48 | A field in the wrong role | ✅ Done | `test_storage_round_trip_keeps_the_whole_conversion` pins every field's role. `input_annotations` are empty, and the step labels and `source_answer` are task annotations. |
| 49 | Wrong prompt, or one without a source | ✅ Done | The prompt is the published question, which the round trip compares verbatim. `test_request_rebuilds_the_authors_text` pins the system prompt as golden text. |
| 50 | Series placed at the wrong point in the prompt | ✅ Done / ☑️ Done once | Done: `authors_request` equals a line-for-line port of the authors' `process_timeseries` (`test_request_rebuilds_the_authors_text`). Done once: equal on all 4,356 tasks of the full build. |
| 51 | An input reveals the target | ☑️ Done once | By design, multiple-choice prompts name their options: all knowledge prompts, and 1,020 of 1,031 scenario attribution prompts, contain the target. Also, 333 anomaly detection prompts contain their label's word, "yes" or "no", and 99 inferential calculation prompts contain their count as a word. The questions state other numbers too, such as the series length. Which option is right is not stated. |
| 52 | The target leaks through an id, metadata or another field | ✅ Done | Record metadata holds the kind and rows. Task metadata holds `source_task`, `part` and `row`. Neither holds a label, and `test_storage_round_trip_keeps_the_whole_conversion` pins both. |
| 53 | An input carries information from after the target | ⚪ N/A | No task predicts future values. The series is the whole context of its question. |
| 54 | Answer format doesn't match what the evaluation expects | ✅ Done / ☑️ Done once | Done: the rationale earns the authors' final-answer reward against the label (`test_rationale_earns_the_authors_final_answer_reward`). Done once: 4,354 of 4,356 published `cot` values earn it. The 2 that don't are scenario attribution rows whose `step6_label` lacks the label. |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ☑️ Done once / ⚠️ Open | Done once: the release's card states Apache-2.0. Open: the ChatTS parent's card says Apache-2.0 but its `LICENSE` file is MIT. The knowledge datasets' upstream terms weren't read. |
| 56 | Wrong domain, tags or description | ☑️ Done once | Reviewed: general for ChatTS, cardiology, health for EMG, energy, and audio. |
| 57 | A README number that doesn't match the build | ☑️ Done once | Every README number comes from the passes in this record. They aren't generated from a stats snapshot. |
| 58 | A README claim the data doesn't support | ☑️ Done once | Each claim cites the authors' code or a pass here. Paper claims are named as the paper's. |
| 59 | Known limitations left undocumented | ☑️ Done once | The README's "Source issues" section lists the count mismatch, the split overlap, the duplicates, the decimation and aliasing, the decimal label, the counts of every point, the scenario attribution labels, and the CoT disagreements. |

## Open items

1. **Train-row sampling rates (#25, #12).** The ECG and RCW train series are assumed to be
   decimated like the test series. The repository copy keeps no train original.
2. **Upstream knowledge data (#19, #21, #44, #55).** The ECG, EMG, CTU and RCW series weren't
   compared with their upstream datasets. So their units, sign, subject leakage and license terms
   are unchecked.
3. **Label quality (#42, #47).** Labels and CoTs come from ChatTS and DeepSeek-R1, and no model or
   rule check has been run.
4. **ChatTS license (#55).** For maintainers: the parent's card and its `LICENSE` file disagree.
5. **File checksums (#3).** A changed file that keeps the counts would pass.
6. **Class balance (#39).** The paper publishes none to compare.

## Plan

- Check the decimation on train: map knowledge train rows to their TimeBed or upstream originals
  by value, and confirm stride 2 and 8.
- Fetch PhysioNet/CinC 2017, the EMG and right-whale sources, and compare matched series' units
  and sign. Record patient or recording ids where they can be matched, to count leaks.
- Pin each file's sha256 in `release.py` and check it in `convert`.
- Add the one-off pass scripts as a `scripts/` check. They are: the comparison with the repository
  copy and ChatTS, the decimation, the full read-back and the reward pass.
- Run the authors' rule-based answer check on the anomaly detection thresholds as a label sanity
  check.
