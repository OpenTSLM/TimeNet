# Verification: `physionet/ecg-qa-cot`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py` unless stated otherwise. They run on
hand-written fixtures, so they show the code's rules, not what the real data means. The ☑️ Done
once checks are one-off passes run on 2026-10-23 over the three inputs as downloaded that day and
two full builds of this connector, both read back through `TimeNet().load`. Their scripts aren't in
the repo yet (see Plan). The connector has no README.

## References

The connector reads three inputs, none of them pinned (#2):

- **PTB-XL 1.0.3**, `s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip` (1,839,504,686 bytes, S3 ETag
  `d824d1c0…-220`, last modified 2024-07-15). Only `records500/` is read. The archive's own
  `SHA256SUMS.txt` is byte-identical to the copy PhysioNet publishes for 1.0.3, and lists every
  file in the archive.
- **ECG-QA's `answers_for_each_template.csv`**, from the `master` branch of the ECG-QA GitHub
  repository (sha256 `e7d28502…db10`). `master` was at commit `73386bb` (2025-05-13).
- **The OpenTSLM ECG-QA-CoT archive** on an ETH polybox share (`ecg_qa_cot_final.zip`, 45,608,362
  bytes, sha256 `f397786c…164e`). It holds three CSVs, one per split.

Independent copies used by the checks below, none of which the connector reads:

- **`wfdb.rdsamp`** on each PTB-XL record, as the reference reader for values, lead names, units,
  sample rate and length.
- **`ptbxl_database.csv`** from the same archive: patient ids, age, sex, height, weight, the noise
  and pacemaker columns, and `strat_fold`.
- **The ECG-QA release** at `73386bb`: `template/{train,valid,test}/*.json` (every question with
  its full answer list, ECG ids and attribute type) and `{train,valid,test}_ecgs.tsv`.
- **The OpenTSLM code** at commit `2968f4b`: `ecgqa_cot_loader.py` (reads the CSVs),
  `ECGQACoTQADataset.py` (training prompt, preprocessing, answer options), `ecgqa_loader.py`
  (`create_clinical_context`, which wrote the CSVs' context column), and
  `evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py` (answer extraction and scoring).
- **The OpenTSLM paper**, arXiv:2510.02410: Table 4 for split sizes, Section 4.1 for how the
  rationales were generated (GPT-4o, given a plot and the ground-truth answer).

## Fields

What every field of the three inputs and of the authors' code becomes, then every value the
connector adds. "CoT CSV" is `ecg_qa_cot_{train,val,test}.csv`.

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| `records500/<bucket>/<id>_hr.{hea,dat}` | PTB-XL | 12 float32 lead Signals per Record, in mV at 500 Hz | Decoded with the header's gain and baseline (#18) |
| Lead names (`I` … `V6`, with `AVR`, `AVL`, `AVF`) | PTB-XL headers | Signal names | As published. OpenTSLM writes `aVR`, `aVL`, `aVF`. TimeF returns signals sorted by name, so the header's lead order isn't kept |
| `records100/` | PTB-XL | Not read | The 100 Hz copy. The authors decimate the 500 Hz records instead (#30) |
| `ptbxl_database.csv`, `scp_statements.csv` | PTB-XL | Not read | Patient id, demographics, SCP codes and folds stay out of the build. Not noted anywhere (#11) |
| `recording_date` | `ptbxl_database.csv` | Not read | No Record has a start time (#26) |
| `sample_id` | CoT CSV | Dropped | OpenTSLM's own row number, not ECG-QA's `sample_id` (#5). 7 values repeat in train |
| `question` | CoT CSV | Task `prompt` | ECG-QA's template question, unchanged (#5) |
| `answer` | CoT CSV | Task target | The first item of ECG-QA's answer list (#41) |
| `ecg_id` | CoT CSV, as `[n]` | Join key, Record id `ptbxl-<n>` | Every value holds one id (#33) |
| `clinical_context` | CoT CSV | `clinical_context` input annotation | Written by OpenTSLM's `create_clinical_context`, which misreads PTB-XL (#23, #24, #51) |
| `question_type` | CoT CSV | `question_type` input annotation | ECG-QA's type. The authors' prompt doesn't show it to the model (#48) |
| `template_id` | CoT CSV | `template_id` input annotation, and the key for `answer_options` | ECG-QA's id. The authors' prompt doesn't show it either (#48) |
| `prompt` | CoT CSV | Dropped | The GPT-4o prompt that wrote the rationale. It contains `Answer: <target>` in all 231,543 rows. Supervision, so right to drop, but not noted |
| `rationale` | CoT CSV | Task `rationale` | Supervision only (#48). The authors train on it as the answer |
| `template_id`, `classes` | `answers_for_each_template.csv` | `answer_options` input annotation | ECG-QA's options per template, all 70 parsed (#38) |
| Comparison questions, `paraphrased/` | ECG-QA only | Not in the source | The CoT release keeps the 42 single-ECG templates, as the paper says |
| Full answer list | ECG-QA `template/*.json` only | Not in the source | The CoT CSV keeps only the first answer (#41) |
| Training prompt | `ECGQACoTQADataset._get_pre_prompt`, `_get_post_prompt` | Not reproduced | The build stores the bare question (#49) |
| Downsample to 100 Hz (`[::5]`), per-lead z-score, mean and std in the lead label | `ECGQACoTQADataset._process_ecg_lead` | Not applied | `dataset.yaml` asks users to reproduce it at load time (#30) |
| `split` annotation | Added | One per Record: `train`, `validation` or `test` | From the CSV the ECG's rows come from. No task carries a split (#43) |
| Default context `12-lead ECG recording.` | Added | Used when `clinical_context` is empty | OpenTSLM's default. No row has an empty context, so it is never used |
| Spec `ecg` / `12-lead ECG`, unit millivolt | Added | Signal spec | From the headers, which say mV (#19) |
| Record, source, signal, annotation and task ids | Added | Ids | `ptbxl-<ecg_id>`, `ecg-<ecg_id>-<lead>`, `ecgqa-<split>-<row>`, and value-derived ids for the annotations (#52) |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | The three URLs are the ones OpenTSLM's loaders read at `2968f4b`: PTB-XL 1.0.3 (`PTBXL_ZIP_URL`), the ECG-QA repo's `master`, and the same polybox link (`ECG_QA_COT_URL`). The CoT split sizes, 159,313 / 31,137 / 41,093, equal paper Table 4. The connector's cached answers CSV equals the file at ECG-QA `73386bb`, byte for byte. |
| 2 | Unpinned input | ⚠️ Open | Nothing is pinned. `ensure_archive` and `download_files` get no `sha256`, the answers CSV comes from a branch, and polybox has no immutable revision. Plan: pin the sha256 values in References. ECG-QA can be pinned to commit `73386bb` instead of `master`. Polybox needs a decision on a mirror. |
| 3 | Partial, corrupt or changed download | ☑️ Done once / ⚠️ Open | Done once: all 87,203 files listed in the archive's `SHA256SUMS.txt` hash correctly, and that file equals PhysioNet's published copy. The three CoT CSVs hash the same in the connector's cache and in a separate download. Open: no build checks a hash. The hash check exists (`test_download_http_rejects_a_sha256_mismatch` in the package tests), but the connector doesn't use it, and an S3 download gets no hash check at all. |
| 4 | Wrong files from the archive | ✅ Done / ⚠️ Open | Done: each CoT CSV is found by name, and a missing one raises (`test_find_dir_containing_raises_when_nothing_matches` in the package tests). A missing PTB-XL record raises when its header is read. Open: there is no pinned inventory with row counts. A CoT CSV that is short, or has extra files beside it, would pass. One-off: the CoT zip holds exactly the three CSVs, and PTB-XL has 21,799 `.hea` and 21,799 `.dat` files under `records500/`. |
| 5 | Source structure misread | ☑️ Done once | Every CoT row was joined to ECG-QA's template release on (`ecg_id`, `question`), per split. All 231,543 rows join, each to exactly one ECG-QA question. `template_id` and `question_type` agree for all of them, and the CoT answer is ECG-QA's first answer for all of them. The CoT `sample_id` is not ECG-QA's: matched by `sample_id`, 0 train rows share both ECG and question. Each split's ECGs are a subset of ECG-QA's `*_ecgs.tsv` for that split. |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ☑️ Done once / ⚠️ Open | Done once: the full build holds 16,024 Records, 192,288 Signals and 231,543 tasks. That is 159,313 / 31,137 / 41,093 tasks, as in paper Table 4, and 11,543 / 1,274 / 3,207 ECGs. Open: the paper also says 159,306 train samples (Table 7), "3,138 unique questions" (the CSVs hold 3,409) and "240k samples". The source disagrees with itself, and nothing in the repo records the counts (plan D). |
| 7 | Rows dropped silently | ☑️ Done once | Every CSV row becomes a task, and every task id `ecgqa-<split>-<row>` maps back to its row: 231,543 rows, 231,543 tasks, 0 missing. Every ECG that a row names becomes a Record, and every Record has a task. The CoT release itself drops 30 ECGs that ECG-QA's splits list, 23 train and 7 test (#37). |
| 8 | Wrong filter or threshold | ☑️ Done once | The connector filters nothing. The authors removed ECG-QA's comparison questions, which leaves the 42 single-ECG templates the paper names. The CSVs hold exactly those 42 templates and the three `single-*` types. |
| 9 | Duplicate records | ✅ Done / ☑️ Done once / ⚠️ Open | Done: one Record per ECG, not per question (`test_one_record_per_ecg_not_per_question`). Done once: 16,024 Record ids and 231,543 task ids, all distinct. Open: the CoT CSVs repeat a question on the same ECG 29,254 times, in 28,581 groups, and leave out 29,247 of ECG-QA's single-ECG questions. Repeats always share the answer and the context, and differ in rationale in all but 36 groups. They look like a resample, and nothing documents them. |
| 10 | Missing or extra channels | ☑️ Done once | All 16,024 Records have the same 12 lead names as their header, which `wfdb.rdsamp` confirms: `I`, `II`, `III`, `AVR`, `AVL`, `AVF`, `V1`–`V6`. The fixture test `test_twelve_leads_per_record` only shows the fixture has 12. Nothing enforces 12 on real data. |
| 11 | Source information left unused without a note | ⚠️ Open | The Fields table lists every field. Dropped without a note: the CSVs' `sample_id` and generator `prompt`, ECG-QA's full answer lists, and all of `ptbxl_database.csv`, including the patient id that the split leakage check needs (#44). The connector has no README to note them in. |
| 12 | Information the connector made up | ☑️ Done once | The Fields table lists every added value. The split comes from the source file. The default context is OpenTSLM's, and no row uses it. The unit comes from the headers. The spec name and the ids are labels, not data. |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once | Done: the direct format-16 reader matches `wfdb.rdsamp` on a fixture (`test_series_values_match_fixture_record`), and records it can't read go to `wfdb` (`test_read_header_falls_back_when_the_record_is_not_plain_16`). Done once: every lead of all 16,024 Records, read back through `TimeNet().load`, against `wfdb.rdsamp`. 192,288 leads, all 5,000 samples each, maximum difference 9.5e-7 mV. |
| 14 | Precision lost on a dtype change | ☑️ Done once | PTB-XL stores int16 counts at 1000 per mV. Every float32 value read back, times 1000, is within 0.00095 of an integer, so it restores the count exactly. The largest difference from `wfdb`'s float64 is 9.5e-7 mV. |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | Counts are read as `<i2` and converted to float32, which holds every int16. Read back, values run from −31.896 to 30.669 mV, inside the int16 range of ±32.767 mV. They match `wfdb` everywhere (#13). |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: `test_convert_round_trips_through_the_writer`. Done once: the full build, read back, against the CSVs for every task. Record, prompt, target, rationale, `question_type`, `template_id` and `clinical_context` all agree, 0 mismatches over 231,543 tasks. Every task has the same four input annotation keys. Signals as in #13. |
| 17 | Build isn't deterministic | ☑️ Done once | Two full builds from the same inputs. The 29 value files are byte-identical. All 16 `control.duckdb` tables hold the same rows, apart from the writer's generated ids in `annotation_occurrences.occurrence_id` and `axes.axis_id`. The manifests differ only in that file's checksum and size. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ✅ Done / ☑️ Done once | Done: the fixture test decodes with the header's gain and baseline, and compares with `wfdb` (`test_series_values_match_fixture_record`). Done once: every real value matches `wfdb`'s physical signal (#13). |
| 19 | Wrong units or scale | ☑️ Done once | Every header says mV for all 12 leads, and the spec says millivolt. Values match `wfdb`'s physical units. Across all Records, 24 exceed ±10 mV, the most extreme ±31.9 mV. That points to upstream artifacts in a few records, not a factor of 1000. |
| 20 | Swapped or mislabeled channels | ☑️ Done once | Over all 16,024 Records, each lead correlates best with its own `wfdb` lead, never another. Einthoven's law holds: the median of \|I + III − II\| / \|II\| is 3.7e-8, at most 2.5e-6, so the limb leads aren't swapped. The read-back order differs from the header's, because TimeF sorts signals by name, but each name stays on its lead. |
| 21 | Flipped sign convention | ☑️ Done once | Einthoven's law (#20) holds with its signs, and so does Goldberger's: aVR + aVL + aVF ≈ 0. Its median residual is 0.015 of \|II\|, which fits rounding to whole µV. 12 Records exceed 0.05. The signs agree with `wfdb` everywhere. |
| 22 | Missing-value markers stored as values | ☑️ Done once | No NaN, and no value at WFDB's int16 marker of −32.768 mV. One Record has one flat lead, as in the source. |
| 23 | Wrong categorical codebook | ☑️ Done once / ⚠️ Open | Done once: `question_type`, `template_id` and the answer options are copied, and every target is one of its template's options in all 231,543 tasks. Open: **the context's sex is reversed in every ECG.** OpenTSLM writes "male" for PTB-XL sex code 1, but code 0 is male. In PTB-XL, code 0 averages 173.5 cm and 78.5 kg (3,187 with height), and code 1 averages 161.0 cm and 64.6 kg (3,787). So all 16,024 contexts give the wrong sex. The connector copies the CSV, which is upstream's error, but passes it to the model as input. |
| 24 | Implausible upstream values passed through without a note | ⚠️ Open | The check found problems, and nothing documents them. **The context claims every artifact for nearly every ECG.** `create_clinical_context` treats a missing value as present, because NaN is truthy. All 16,024 contexts claim baseline drift, static noise, electrode artifacts, extra beats and a pacemaker, and 16,023 claim burst noise. PTB-XL records these for 1,491, 1,904, 29, 1,208, 99 and 279 of the ECGs. 290 contexts give PTB-XL's over-89 marker as age: "300-year-old". The signals hold 24 Records over ±10 mV and one flat lead (#19, #22). |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ☑️ Done once | Every header gives 500 Hz and 5,000 samples, which is PTB-XL's 10 s. Every Signal read back has a 2,000 µs period and 5,000 values. The paper's "100 Hz" is the authors' downsampling (#30), not the source. |
| 26 | Wrong start time or epoch | ☑️ Done once | No Record has a start time (`TimeOrigin(timestamp=None)`), and every axis starts at offset 0. `recording_date` isn't read (#11). No task depends on absolute time. |
| 27 | Wrong time zone | ☑️ Done once | Nothing in the build holds a clock time (#26), so no time zone applies. |
| 28 | Daylight-saving errors | ☑️ Done once | As for #27: no clock times are stored. |
| 29 | Gaps or missing values filled or dropped silently | ☑️ Done once | Every lead is a regular 5,000-sample axis without NaN, equal to `wfdb` (#13). Nothing is filled. |
| 30 | Artifacts from resampling or interpolation | ☑️ Done once / ⚠️ Open | Done once: the connector doesn't resample. All samples match the 500 Hz source. Open: the authors train on `[::5]` decimation with no anti-alias filter, plus a per-lead z-score. `dataset.yaml` says to downsample and normalize, but not that the decimation is plain, nor that PTB-XL ships its own `records100/`. A user who picks one or the other gets different inputs from the paper's. |
| 31 | Off-by-one sample | ☑️ Done once | All 5,000 samples of every lead match `wfdb` sample for sample (#13). |
| 32 | Sources or channels out of sync | ☑️ Done once | The 12 leads share one rate and length per Record. Einthoven's law holds sample by sample (#20), which a shift in any limb lead would break. |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ☑️ Done once | Tasks join Records by `ecg_id`, not by order. Every task's Record is `ptbxl-<ecg_id>` of its row (231,543, 0 mismatches). An independent join with ECG-QA on (`ecg_id`, `question`) matches every row and every answer (#5). |
| 34 | Off-by-one id | ☑️ Done once | Checked against the signal. For "Does this ECG show symptoms of sinus tachycardia?", lead II's heart rate exceeds 100 bpm in 96.0% of 99 "yes" tasks and 6.7% of 209 "no" tasks. A threshold rule scores 94.2%, against 55.0% with shuffled answers and 67.9% for the majority. For sinus bradycardia, under 60 bpm in 94.4% of 108 "yes" and 14.2% of 212 "no", 88.7% against 52.8% and 66.2%. Age and sex in each context come from its own ECG's PTB-XL row (#36). |
| 35 | Ids or join keys not unique | ✅ Done / ☑️ Done once | Done: Records are keyed by ECG id, one per ECG (`test_one_record_per_ecg_not_per_question`), and `add_record` rejects a repeated id. Done once: every `ecg_id` holds one integer. 16,024 Record ids and 231,543 task ids are all distinct. In ECG-QA, (`ecg_id`, `question`) is unique. |
| 36 | Subject metadata on the wrong recording | ☑️ Done once | The context is the only subject field. Regenerated with the authors' `create_clinical_context`, executed from source, it equals the CSV for all 16,024 ECGs. Each ECG has one context. Age equals PTB-XL's for all 16,024. Across the 1,239 patients with more than one ECG, the context's sex never changes. It is reversed throughout (#23). |
| 37 | Join silently drops rows | ☑️ Done once | Every row joins (#7). Upstream, the CoT release covers 16,024 of ECG-QA's 16,054 split ECGs and 202,289 of its 231,536 single-ECG questions (#9). Nothing documents that. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ✅ Done / ☑️ Done once | Done: options come from `answers_for_each_template.csv` by template id (`test_answer_options_come_from_template`). Done once: the 42 options annotations, read back, equal ECG-QA's lists in their order, and every target is in its list. OpenTSLM keys the same file by the same `template_id`. |
| 39 | Label distribution differs from the paper | ☑️ Done once / ⚠️ Open | Done once: split sizes equal Table 4 (#6). Open: the paper gives no answer balance. Verify answers are no / yes / not sure: 40,578 / 17,375 / 67 in train, 6,538 / 2,693 / 4 in validation, 7,660 / 3,431 / 10 in test. "none" is 24.8%, 33.3% and 25.8% of the other answers. Nothing records these counts (plan D). |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | One task per question on one 10 s recording, as in ECG-QA's single-ECG templates. The connector keeps one Record per ECG, which the questions share. |
| 41 | Derived target computed wrongly | ⚠️ Open | **The check found a problem upstream.** For 37,140 tasks, ECG-QA's answer has 2 to 12 items, and the CoT CSV kept only the first: 25,260 in train, 3,894 in validation, 7,986 in test. All are `single-query` or `single-choose`, for example which leads show a finding. A model that names a correct second lead counts as wrong. Also, 3,540 rationales end with an "Answer:" different from the target, and 1,052 have no "Answer:". Some of them refuse to answer. Needs a decision: restore the full lists from ECG-QA, or document it. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once | The heart-rate check (#34) finds the rate labels agree with the signal, far above shuffled labels. Only the two rate questions were checked against the signal. Other labels rest on ECG-QA, matched for every row (#5). |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ☑️ Done once / ⚠️ Open | Done once: each Record's `split` annotation comes from its CSV. The three splits share no ECG, and each is a subset of ECG-QA's split of that name. They aren't PTB-XL's `strat_fold` split: each draws from all ten folds. Open: **no task has a split.** All 231,543 have `split=None`, so `get_train()` raises "assigns no split to any task". The split exists only as a Record annotation, which the SDK doesn't read. `test_recording_carries_its_split` checks the annotation, not the task. Fix: set `split` on each `AnswerTask`. The paper's "80/10/10" is really 68.8 / 13.4 / 17.7%. |
| 44 | Leakage across splits | ☑️ Done once | Grouped by PTB-XL `patient_id`, the 16,024 ECGs come from 14,423 patients, and 0 patients span two splits. No ECG is in two splits. The same template questions appear in every split, by design. |
| 45 | Splits regenerated without a fixed seed | ⚪ N/A | Nothing regenerates a split. The connector uses the CSVs' splits as published. |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | Question answering with a chain-of-thought rationale, as in paper Section 4.1 and `ECGQACoTQADataset`, which trains on the rationale and scores the final answer. Each is an `AnswerTask` with a short target and the rationale. |
| 47 | Task can't be solved from its inputs | ☑️ Done once / ⚠️ Open | Done once: the rate labels follow from the signal (#34). Options and question come from the inputs. Text alone, the majority answer per question in train, scores 43.9% on test. The paper's model scores 46.25%. Open: the context contradicts its target in 10,778 `single-verify` tasks. They ask about baseline drift, static noise, burst noise, electrodes or a pacemaker, the answer is "no", and the context says it is present (#24). 210,813 of 231,543 rationales mention a pacemaker. No ECG baseline has run (plan G). |
| 48 | A field in the wrong role | ✅ Done / ☑️ Done once / ⚠️ Open | Done: the rationale is only a rationale, and differs from the target (`test_each_question_is_an_answer_task_on_its_recording`). Done once: in the full build, every task's input annotations are exactly `answer_options`, `clinical_context`, `question_type` and `template_id`. The generator prompt, which holds the answer, isn't stored. Open: `question_type` and `template_id` are metadata that the authors' prompt never shows the model, yet they are task inputs here. |
| 49 | Wrong prompt, or one without a source | ⚠️ Open | The prompt is the bare ECG-QA question, which is its source. The authors' training prompt adds a role, the context, instructions, the options and "Answer: ", in `_get_pre_prompt` and `_get_post_prompt` at `2968f4b`. The build neither stores nor cites it, and no golden text pins it. Needs a decision and a snapshot (plan G). |
| 50 | Series placed at the wrong point in the prompt | ⚠️ Open | The authors place the 12 leads between the question block and the options, each labelled "This is ECG Lead X, it has mean … and std …". The build's prompt has no placeholder, so a renderer decides. No rendered task has been read (plan G). |
| 51 | An input reveals the target | ☑️ Done once / ⚠️ Open | Done once: no input carries the answer beyond what the task design gives. The options list it, and `single-choose` questions name both choices, as in ECG-QA. Question plus context scores 42.7% on test, no better than the question alone (43.9%). Open: the context, though not a leak, is a wrong input. It asserts every artifact and a reversed sex for every ECG (#23, #24). Needs a decision: rebuild the context from `ptbxl_database.csv`, or drop it from the inputs. |
| 52 | The target leaks through an id, metadata or another field | ☑️ Done once | Besides the inputs, a loader passes only the record annotations, and in the full build each of the 16,024 Records has just one, `split`. Record and source metadata are empty. The split comes from the source file, not from the answer. No input holds an id: the prompt is the question, and the input annotations are the four of #48. No split is cut from a range of ids. Each split's ECG ids run from 1–3 to about 21,836, and each split draws from all ten PTB-XL folds (#43). Ids hold the ECG id, the split and the row index. Neither predicts the answer: per question, the majority answer by ECG-id decile scores 41.9% on test, and by row-index decile 43.6%, against 43.9% for the question alone. |
| 53 | An input carries information from after the target | ☑️ Done once | The inputs are the recording, the question, the options and a context built from PTB-XL metadata of the recording. The rationale was written later, with the answer, but is supervision only (#48). |
| 54 | Answer format doesn't match what the evaluation expects | ☑️ Done once / ⚠️ Open | Done once: a dummy evaluation with the authors' `extract_answer` and `calculate_f1_score`, executed from source. Every target is among its options, so a model that outputs `Answer: <target>` matches the build's target exactly. Open: the authors take their gold label from the rationale, not from `answer`. Scored that way, the same perfect model gets 97.98% / 98.06% / 98.13% on train / validation / test, and 708 / 165 / 223 gold labels aren't among the options (#41). TimeNet has no evaluation for this task yet. |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ⚠️ Open | The card declares CC-BY-4.0. PTB-XL's `LICENSE.txt` and ECG-QA's `LICENSE` are CC BY 4.0. The CoT archive states no license. The OpenTSLM repository is MIT, which covers its code. The rationales were generated with GPT-4o. Needs the authors to confirm the license of the CoT release. |
| 56 | Wrong domain, tags or description | ✅ Done / ⚠️ Open | Done: the card validates (`test_every_connector_ships_a_valid_dataset_card` in the package tests). Domain and tags were reviewed. Open: the description promises "clinical questions, short answers", but for 37,140 tasks the short answer is incomplete (#41), and the "clinical context" misstates sex and artifacts (#23, #24). |
| 57 | A README number that doesn't match the build | ⚠️ Open | The connector has no README. The connector's docstring says "~230k" tasks, which fits 231,543. Nothing ties a number to the build (plan D). |
| 58 | A README claim the data doesn't support | ⚠️ Open | No README. Two claims elsewhere fail. The docstring and card say each recording "carries its dataset split", but the SDK finds no split (#43). The card says a model reproduces OpenTSLM's preprocessing by "downsample to 100 Hz", which leaves out that the authors decimate with no filter (#30). |
| 59 | Known limitations left undocumented | ⚠️ Open | Nothing documents the limitations. They are: the wrong context (#23, #24), the truncated multi-answers (#41), rationales that contradict their target (#41), repeated and missing questions (#9, #37), split only on Records (#43), the lost lead order (#20), and the unclear license (#55). |

## Open items

Most important first.

1. **No task has a split (#43).** `get_train()`, `get_validation()` and `get_test()` raise on this
   dataset, because the split is a Record annotation only. Fix in the connector: set
   `split=` on each `AnswerTask`. A test should then check one task of each split.
2. **The clinical context is wrong for every ECG (#23, #24, #47, #51).** Upstream,
   `create_clinical_context` reads NaN as present, so every context claims every artifact and a
   pacemaker. It also reverses PTB-XL's sex code, and writes 290 ages as 300. The context is a task
   input, it contradicts the target in 10,778 noise and pacemaker questions, and most rationales
   repeat it. Decision for maintainers: rebuild the context from `ptbxl_database.csv`, drop it from
   the inputs, or keep it as published and document it. Rebuilding it would change what the
   rationales refer to.
3. **Multi-answer questions keep only the first answer (#41, #54, #56).** That's 37,140 tasks.
   Decision: restore ECG-QA's full answer lists as the target, joined on (`ecg_id`, `question`),
   with the CoT answer kept for the rationale, or document the truncation. Document the 3,540
   rationales whose final answer differs from the target, and the 1,052 that give none.
4. **Unpinned inputs (#2, #3).** Pin the three sha256 values in References. Pin ECG-QA to commit
   `73386bb`. Hash the S3 archive after download, and the extracted CSVs before reuse.
5. **No README (#11, #57, #58, #59).** Write one: the sources and the release used, the fields
   dropped, the authors' prompt and preprocessing, the repeated and missing questions, and every
   limitation above.
6. **Prompt and placement (#49, #50).** Decide whether to store the authors' training prompt, or
   cite it and snapshot a rendered task. Read rendered tasks.
7. **Field roles (#48).** `question_type` and `template_id` should be task metadata, not inputs.
8. **License of the CoT release (#55).** Ask the authors.
9. **Numbers that can drift (#6, #39).** Plan D.
10. **Lead order (#20).** TimeF returns leads sorted by name. Decide whether to record the header
    order, as other connectors do with a `channel_order` metadata field.

## Plan

Checks on the real data run in a `real_data` pytest marker, skipped unless an environment variable
points at the extracted inputs. Every check below except D, F, G and H ran once for this record.

- **A1. Reference reader.** Read every lead of every Record through the SDK and through
  `wfdb.rdsamp`. Compare values, names, units, rate and length, and check Einthoven's and
  Goldberger's laws.
- **A2. ECG-QA join.** Join every CoT row to ECG-QA's template JSON at a pinned commit on
  (`ecg_id`, `question`). Check the template, the type, and the answer against the full list.
  Count repeated and missing questions, and check each split's ECGs against `*_ecgs.tsv`.
- **A3. The authors' code.** Execute `create_clinical_context` and `extract_answer` from source at
  `2968f4b`. Compare every context, and run the dummy evaluation.
- **B1. PTB-XL facts.** Check the context against `ptbxl_database.csv`: sex from body size, the
  noise and pacemaker columns, and age. Count patients across splits.
- **B2. Rate check.** The heart rate from lead II against the sinus tachycardia and bradycardia
  answers, with shuffled answers as the control.
- **C1. Text-only baselines.** Answer from the question alone, then with the ECG-id decile, the
  row-index decile and the context. None may beat the question alone.
- **D. Stats snapshot.** Commit a `verification.json` with the counts per split, answer balance,
  repeated and truncated questions, context claims, and signal range and flat-lead counts. A test
  regenerates it and compares.
- **E. Determinism.** Build twice and compare value files and control tables.
- **F. Corruption tests.** Swap leads II and III, shift every task to the next ECG id, and scale a
  lead by 1000. A1 and B2 must each fail at least one.
- **G. Review sheet and evaluation.** Render 12 seeded tasks across the three question types,
  using the authors' prompt, with the leads plotted over `wfdb`. Add an evaluation that scores the
  final "Answer:" against the target and the options.
- **H. Split on tasks.** After the fix to item 1, check that every task's split equals its
  Record's, and that `get_train()` returns 159,313 tasks.
