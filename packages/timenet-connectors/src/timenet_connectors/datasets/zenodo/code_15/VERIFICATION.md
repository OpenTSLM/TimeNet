# Verification: `zenodo/code-15`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py` unless stated otherwise. The ☑️ Done once
checks are one-off passes run on 2026-10-29 over the pinned release, whose 19 files have the md5s
Zenodo publishes: an independent scan of every part, and part 17's 5,771 Records converted, stored
in a local registry and read back with `TimeFReader`. The source was read for them with h5py and
Python's `csv` alone, sharing no code with the connector. Their scripts aren't in the repo yet (see
Plan). The read-back of a full build is still open.

## References

The connector reads 19 files of Zenodo record 4916206, version 1.0.0, the record's only version:
`exams.csv` (35,500,900 bytes) and `exams_part0.zip` to `exams_part17.zip` (2.67 GB each, 0.78 GB
for part 17), each holding one HDF5 file. Each is pinned by the sha256 computed on download.

Independent copies and references, which the connector doesn't read:

- **The Zenodo description**: columns, HDF5 layout, lead order, 400 Hz, the 7 s and 10 s lengths and
  their padding, and the md5 of every file.
- **The authors' code**: `antonior92/ecg-age-prediction` at `f192e49` (`train.py`, `evaluate.py`,
  `dataloader.py`, `formulate_problem.py`), `antonior92/automatic-ecg-diagnosis` at `4039348`
  (README output order, `datasets.py`, `train.py`), and `antonior92/ecg-preprocessing` at
  `439bc1b` (`ecgprep/preprocess.py`, the README's CODE-15% command). Issues 1, 2 and 5 of the
  first and 3, 10, 32 and 34 of the second.
- **The papers**: Lima et al., *Nat Commun* 12, 5117 (2021), and Ribeiro et al., *Nat Commun* 11,
  1760 (2020).
- **The published model output**: `nn_predicted_age`, the age network's prediction for each exam,
  is in `exams.csv`.

The recordings before the authors' preprocessing are not public, so no value can be compared with
the device's output. No other copy of CODE-15% exists.

## Fields

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| `exam_id` | `exams.csv` | `exam_id` in the Record metadata; the join key | Unique in the CSV and the HDF5 files (#35) |
| `age` | `exams.csv` | The age task's target, as a float in years | Not on the Record (#51) |
| `is_male` | `exams.csv` | `sex` Record annotation, `male` or `female` | Decoded (#23) |
| `nn_predicted_age` | `exams.csv` | `nn_predicted_age` annotation of the age task, in years | The authors' prediction; never input (#48) |
| `1dAVb`, `RBBB`, `LBBB`, `SB`, `ST`, `AF` | `exams.csv` | The abnormality task's targets, in the authors' output order | Not on the Record (#51) |
| `patient_id` | `exams.csv` | Core `subject_id` of the Record metadata | |
| `death`, `timey` | `exams.csv` | `death` and `follow_up` (years) Record annotations, on the exam that states them | No task: neither paper trains on them (#46). Lost for 419 patients whose only statement is on an all-zero exam (#11) |
| `normal_ecg` | `exams.csv` | `normal_ecg` annotation of the abnormality task | Never input (#48) |
| `trace_file` | `exams.csv` | `trace_file` in the Record metadata; checked against the part that holds the exam | |
| `exam_id` dataset | Each HDF5 file | The join key of each row | The last row, id 0, is a filler and skipped (#7) |
| `tracings` dataset | Each HDF5 file | 12 lazy float32 signals per Record, halved, without the centred zero padding | The padding is `pad_start`, `pad_end` in the Record metadata (#29) |
| `--scale 2` | `ecg-preprocessing` README, author in issue 5 | Values halved to millivolts | Exact in float32 (#14, #19) |
| Padding rule | `preprocess.py` `preprocess_ecg` | `preprocessing.centred_padding` | Ours: approved by the maintainer; zeros the rule can't explain stay (#8) |
| Lead names and order | Zenodo, `preprocess.py` `all_leads` | Signal names | #20 |
| Abnormality order | `automatic-ecg-diagnosis` README | Target order and `abnormality_vocabulary` | `exams.csv` writes ST before AF (#38) |
| Splits | The papers | `test` for age, `train` for abnormalities | Ours: the release has none; each task takes its paper's role for CODE-15%, approved by the maintainer (#43) |
| `n_valid`, `val_split` | `train.py` of both repositories | Not stored | Positional splits of the file order, not fixed splits (#43) |
| Age weights | `ecg-age-prediction` `compute_weights` | Not stored | Computed from the training ages at training time |
| Spec `ecg`, `millivolt`, float32 | Added | Signal spec | #19 |
| 400 Hz `RegularAxis`, offset 0 | Added, from Zenodo | Time axis | #25; no exam time (#26) |
| Source name `12-lead ECG` | Added | Source name | |
| `source_url`, `source_doi`, `abnormality_vocabulary` | Added | Dataset annotations | The record, and the authors' output order |
| Record and task ids | Added | Ids | Numbered by TimeNet when the dataset is built. The connector sets none |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | The record has one version, 1.0.0, which both repositories' READMEs link by DOI. All 19 downloaded files have the md5 Zenodo publishes. |
| 2 | Unpinned input | ✅ Done | Every file's sha256 is passed to `download_files` or `ensure_archive`, and a download whose bytes differ fails (`test_download_http_rejects_a_sha256_mismatch` in the package tests). The connector's own download of `exams.csv` and part 17 passed its pins once, and a rerun reused the extraction. |
| 3 | Partial, corrupt or changed download | ✅ Done / ⚠️ Open | Done: downloads are hashed (`test_download_http_rejects_a_sha256_mismatch`); every build scans every part, checks each part's datasets and shapes (`test_a_part_with_other_datasets_fails`), and fails unless it holds the pinned counts and all-zero exams (`test_pinned_counts_mismatch_fails`); a reread trace must have the zeros the scan saw (`test_a_trace_changed_after_the_scan_fails_when_read`). Open: `ensure_archive` reuses an extraction without hashing it again, so a changed value that keeps the counts and zero runs would pass. |
| 4 | Wrong files from the archive | ✅ Done / ☑️ Done once | Done: each part must exist (`test_a_missing_part_fails`), and hold only exams the CSV lists in that part (`test_parts_that_dont_match_exams_csv_fail`). Done once: each zip holds exactly one member, `exams_part<n>.hdf5`. |
| 5 | Source structure misread | ✅ Done / ☑️ Done once | Done: the header, field count, `True`/`False` spellings, `death` and `timey` together, and the `trace_file` pattern are enforced (`test_malformed_exams_csv_fails_with_its_path`). Done once: all 345,779 rows parse; every HDF5 file holds exactly `exam_id` (N,) int64 and `tracings` (N, 4096, 12) float32, N = 20,001, 5,780 for part 17. The Zenodo snippet's `id_exam` and `signal` names are wrong; the prose and the files say `exam_id` and `tracings`. |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ✅ Done / ☑️ Done once / ⚠️ Open | Done: the build fails unless it holds 345,779 exams, 233,770 patients and the 670 pinned all-zero exams (`test_pinned_counts_mismatch_fails`). Done once: the independent scan finds 345,779 traces for the CSV's 345,779 exams, 670 of them all zero, so 345,109 Records and 690,218 tasks. Open: counting them in a full build read back; and Lima et al. report 218,169 patients and ages 16 to 85, where the release has 233,770 patients and ages 17 to 100, which the README says. |
| 7 | Rows dropped silently | ✅ Done / ☑️ Done once | Done: every CSV exam must have a trace (`test_parts_that_dont_match_exams_csv_fail`), the filler row must be there (`test_a_part_without_its_filler_fails`), and the dropped all-zero exams are pinned (`test_pinned_counts_mismatch_fails`). Done once: the independent scan finds all 18 filler rows, 345,779 traces for 345,779 CSV exams, and 670 all-zero; the parse of the whole release builds one Record for each of the other 345,109. |
| 8 | Wrong filter or threshold | ☑️ Done once / ⚠️ Open | Done once: the only filter is a trace that is zero everywhere. The padding rule is `preprocess_ecg`'s: on part 17, stored and read back, every one of the 5,771 Records' 12 leads times 2, re-padded with its `pad_start` and `pad_end`, equals its published trace bit for bit. Open: the read-back of a full build. |
| 9 | Duplicate records | ✅ Done / ☑️ Done once | Done: an exam listed twice in the CSV or with two traces fails (`test_malformed_exams_csv_fails_with_its_path`, `test_parts_that_dont_match_exams_csv_fail`). Done once: 345,779 distinct exam ids in the CSV and in the HDF5 files, by the independent scan. |
| 10 | Missing or extra channels | ☑️ Done once | Every part's `tracings` has 12 leads, which the build checks on every part, and every Record of part 17, read back, has exactly the 12 named leads. |
| 11 | Source information left unused without a note | ☑️ Done once | The Fields table lists every column, dataset and input of the authors' code. Dropped: the filler rows, the 670 all-zero traces with their rows (and so the mortality of 419 patients and every exam of 290), and the padding, kept as metadata. The README names each. |
| 12 | Information the connector made up | ☑️ Done once | The Fields table lists every added value with its reference: the unit and scale (author), the rate (Zenodo), the lead and abnormality orders, the splits (ours, from the papers), the names and ids. `start_time` is unset. |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once / ⚠️ Open | Done: every fixture Record rebuilds its trace (`test_signals_rebuild_the_authors_input_bit_for_bit`), also after storage (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: part 17 stored and read back, all 5,771 Records joined to their HDF5 rows by exam id rebuild the published trace bit for bit, 0 mismatches. Open: the read-back of a full build. |
| 14 | Precision lost on a dtype change | ✅ Done / ☑️ Done once | Done: values stay float32, and halving is exact (`test_storage_round_trip_keeps_the_whole_conversion` compares bytes). Done once: halving then doubling returns every value of part 17 exactly, and its read-back is bit for bit (#13). |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | No value is an integer. Exam ids, up to 4,414,126, and patient ids are stored as JSON integers; part 17's read back equal the CSV's. |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once / ⚠️ Open | Done: the whole fixture conversion is written, read back and compared with literal expectations (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: part 17, stored and read back, equals the conversion before storage in every metadata field, annotation, task target and split, and the source in every value. Open: the read-back of a full build. |
| 17 | Build isn't deterministic | ⚠️ Open | Two full builds from the same files are running; their stored files are to be compared. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ☑️ Done once | The values are floats the authors converted with `convert_to_mv` and scaled by 2 (`ecg-preprocessing`). DII's peak per trace has median 1.90, so 0.95 mV, a typical R wave. |
| 19 | Wrong units or scale | ☑️ Done once / ⚠️ Open | Done once: the factor 2 is the author's (issue 5 and the `ecg-preprocessing` README), and DII's median peak of 0.95 mV fits. Open: the "scale 1e-4V" sentence of both READMEs contradicts it, and no measurement in a known unit is published to check the factor against. |
| 20 | Swapped or mislabeled channels | ☑️ Done once | DIII equals DII − DI to 10⁻⁶ of the peak in 250,910 of 345,106 traces, and AVR, AVL and AVF are within 2% of their derivation in 258,355, which only the published order fits. On 1,500 normal exams the share of positive QRS is DI 95%, DII 96%, AVR 2%, and rises V1 5%, V2 19%, V3 38%, V4 83%, V5 96%, V6 96%: the R-wave progression, so the precordial order holds. |
| 21 | Flipped sign convention | ☑️ Done once | The connector only halves, so it flips nothing (#13). The polarity pattern of #20 matches a normal ECG in every lead, so the release's signs are the standard ones. |
| 22 | Missing-value markers stored as values | ✅ Done / ☑️ Done once | Done: a non-finite value fails the scan (`test_a_value_that_isnt_finite_fails_when_scanned`); the authors' fill, zero padding, is left out by the centring rule (`test_centred_padding_follows_preprocess_ecg`); an all-zero trace has no Record (`test_converts_the_fixture`). Done once: 0 non-finite values. Zero runs the rule can't explain stay (#24). |
| 23 | Wrong categorical codebook | ✅ Done / ☑️ Done once | Done: only `True` and `False` pass as booleans (`test_malformed_exams_csv_fails_with_its_path`); `is_male` is decoded to `male`/`female` (`test_converts_the_fixture`). Done once: `death` is `True`, `False` or empty, and `is_male` is the same on every exam of a patient. |
| 24 | Implausible upstream values passed through without a note | ☑️ Done once | Kept and in the README: 12,288 traces start with 2 to 4,048 all-lead zeros, 778 have uneven zeros, 23 end with zeros; 1,103 traces have a lead that is zero throughout; `nn_predicted_age` ranges from −40.2 to 106.9. |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ☑️ Done once | Zenodo states 400 Hz. Heart rates counted from DII's R peaks at 400 Hz on 600 random exams per group: ST median 109 bpm (98% above 100), SB 47 bpm (95% below 60), normal 71 bpm, matching the definitions Ribeiro et al. label by. |
| 26 | Wrong start time or epoch | ⚪ N/A | The release states no exam date or time. `start_time` is unset, and each axis starts at the first sample kept. |
| 27 | Wrong time zone | ⚪ N/A | No clock time in the release (#26). |
| 28 | Daylight-saving errors | ⚪ N/A | No clock time in the release (#26). |
| 29 | Gaps or missing values filled or dropped silently | ✅ Done / ☑️ Done once | Done: only padding the centring rule explains is left out, and other zeros stay (`test_centred_padding_follows_preprocess_ecg`, `test_converts_the_fixture`). Nothing is filled. Done once: the independent scan's layouts over the release (201,257 centred, 130,763 unpadded, 13,089 with zeros the rule keeps) match the rule, and part 17 rebuilds exactly (#8). |
| 30 | Artifacts from resampling or interpolation | ☑️ Done once / ⚠️ Open | Done once: the connector doesn't resample (#13). Open: the authors resampled recordings of 300 to 1000 Hz to 400 Hz with `resample_poly`; its effect isn't measured, and the original recordings aren't public. |
| 31 | Off-by-one sample | ✅ Done / ☑️ Done once | Done: odd padding puts the extra zero after (`test_centred_padding_follows_preprocess_ecg`), and fixture traces rebuild exactly (`test_signals_rebuild_the_authors_input_bit_for_bit`). Done once: part 17 rebuilds exactly (#13). |
| 32 | Sources or channels out of sync | ☑️ Done once | The 12 leads share one axis and one padding, and part 17's rebuild places every sample back at its index (#13). |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ✅ Done / ☑️ Done once | Done: the fixture CSV and parts are in different orders (`test_converts_the_fixture`). Done once: the HDF5 order isn't the CSV's, and every Record of part 17 read back carries the trace of its own exam id (#13). |
| 34 | Off-by-one id | ☑️ Done once / ⚠️ Open | Done once: part 17's Records carry the values of the HDF5 row of their exam id (#13). Open: the read-back of a full build. compares every Record's metadata, annotations and tasks with its CSV row. |
| 35 | Ids or join keys not unique | ✅ Done / ☑️ Done once | Done: `test_malformed_exams_csv_fails_with_its_path`, `test_parts_that_dont_match_exams_csv_fail`. Done once: #9. |
| 36 | Subject metadata on the wrong recording | ✅ Done / ⚠️ Open | Done: a patient with two sexes or two mortality statements fails (`test_malformed_exams_csv_fails_with_its_path`), and the fixture's Records carry their own exam's facts (`test_converts_the_fixture`). Open: the read-back of a full build. |
| 37 | Join silently drops rows | ✅ Done / ☑️ Done once | Done: an exam without a trace or a trace without an exam fails (`test_parts_that_dont_match_exams_csv_fail`). Done once: the 670 all-zero exams are the only exams without a Record. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ✅ Done / ☑️ Done once / ⚠️ Open | Done: an exam with ST and AF gets `("AF", "ST")` (`test_converts_the_fixture`). Done once: the vocabulary is the `automatic-ecg-diagnosis` README's output order. Open: the read-back of a full build. |
| 39 | Label distribution differs from the paper | ⚠️ Open | Neither paper nor the Zenodo record gives CODE-15%'s label counts, so there is nothing to compare with. The README's counts are the CSV's over the exams with a Record. |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | The labels and the age are per exam in the CSV and in both papers; each exam has one task of each kind. |
| 41 | Derived target computed wrongly | ✅ Done / ⚠️ Open | Done: targets are the CSV's `age` as a float and its six booleans as names (`test_converts_the_fixture`). Open: the read-back of a full build. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once | The heart rates of #25 agree with the ST and SB labels on 98% and 95% of exams. |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ✅ Done / ⚠️ Open | Done: age tasks are `test` and abnormality tasks `train` (`test_tasks_carry_their_papers_split`), the roles Lima et al. and Ribeiro et al. give CODE-15%. Open: the release defines no split, and Ribeiro et al.'s 2% validation set isn't published; the maintainer chose the per-task roles. |
| 44 | Leakage across splits | ⚠️ Open | Both task kinds share every Record, so the `train` abnormality tasks show a model the ECGs of the `test` age tasks. The README says so. Within a task kind there is one split, so no patient crosses. |
| 45 | Splits regenerated without a fixed seed | ⚪ N/A | No split is generated: each task kind has one fixed split. |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | Age is a regression in Lima et al. (`ScalarPredictionTask`, years); the abnormalities are six sigmoid outputs in Ribeiro et al. (multi-label `ClassificationTask`). Mortality is a Cox analysis in Lima et al., not a model target, so it has no task. |
| 47 | Task can't be solved from its inputs | ✅ Done | The input is the exam's Record, its 12 leads, which is the authors' model input (`test_signals_rebuild_the_authors_input_bit_for_bit`). |
| 48 | A field in the wrong role | ✅ Done | `nn_predicted_age` and `normal_ecg` are task annotations, and no task has input annotations (`test_targets_stay_off_the_inputs`). |
| 49 | Wrong prompt, or one without a source | ⚪ N/A | The source asks no question in words; the tasks have no prompt. |
| 50 | Series placed at the wrong point in the prompt | ⚪ N/A | No prompts (#49). |
| 51 | An input reveals the target | ✅ Done | The age and the labels stay off the Record, which holds only `sex`, `death` and `follow_up` (`test_targets_stay_off_the_inputs`). |
| 52 | The target leaks through an id, metadata or another field | ☑️ Done once | Record metadata holds the patient and exam ids, the file and the padding, none of them a target; metadata is never model input. |
| 53 | An input carries information from after the target | ✅ Done / ⚠️ Open | Done: no task passes the model an annotation (`test_targets_stay_off_the_inputs`). Open: `death` and `follow_up` are facts after the exam on the Record; a loader that hands a model the Record's annotations would give it the future. |
| 54 | Answer format doesn't match what the evaluation expects | ☑️ Done once | The age is a float in years, scored by mean absolute error as in Lima et al. The labels are names in `abnormality_vocabulary`, whose order maps them to the authors' six outputs for per-class scores. |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ☑️ Done once | Zenodo states `cc-by-4.0`; the card says `CC-BY-4.0`. |
| 56 | Wrong domain, tags or description | ☑️ Done once | Cardiology; the description's counts, place, years and 400 Hz are Zenodo's. |
| 57 | A README number that doesn't match the build | ☑️ Done once / ⚠️ Open | Done once: every count in the README was computed on the release by the independent scan. Open: checking the build's counts against them (#6). |
| 58 | A README claim the data doesn't support | ☑️ Done once | Each README claim names its source (Zenodo, a paper, the code, an issue) or a check on the data; where they conflict, the README says which the data follows. |
| 59 | Known limitations left undocumented | ☑️ Done once | The README's "Known issues upstream" lists the filler rows, the all-zero traces, the row order and the authors' code that misaligns ages, the zeros that aren't padding, the predicted ages, the paper's counts, the lead-order statements and the lengths. |

## Open items

1. **The full build (#6, #8, #13, #16, #17, #34, #36, #38, #41, #57).** A full build read back
   against the CSV and the HDF5 files, and a second build compared with it, are running.
2. **The splits (#43, #44).** The release has none; the per-task roles are ours, and they share inputs
   across splits. Decide whether a patient-level split of our own should be added for training.
3. **The scale (#19).** The factor 2 rests on the author's statement; no measurement in a known unit
   checks it.
4. **Mortality on the Record (#53).** `death` and `follow_up` are after the exam; they are safe only
   while no loader feeds a Record's annotations to a model.
5. **Reused extraction (#3).** A changed value that keeps the counts and zero runs isn't caught after
   the first extraction.
6. **The paper's counts (#6) and label distribution (#39).** No published figure for the release
   to match.
7. **Resampling (#30).** The authors' resampling isn't measured.

## Plan

- Add the one-off scripts (the independent scan, the read-back comparison, the heart-rate and
  polarity checks) to the repository's verification tooling, so the ☑️ Done once rows can be rerun.
- Run the authors' published age model on `preprocessing.authors_input` for a sample of Records and
  compare with `nn_predicted_age`. A match confirms the input, scale included, is the one the model
  was trained on (#19).
- Hash each extracted HDF5 file against a pinned digest when the extraction is reused (#3).
