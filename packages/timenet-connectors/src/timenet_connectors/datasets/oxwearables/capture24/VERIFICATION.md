# Verification: `oxwearables/capture24`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py` unless stated otherwise. The ☑️ Done once
checks are one-off passes run on 2026-10-28 over the pinned archive and two full builds of dataset
version 1.0.0, read back through `TimeNet.load`. The source was read for them with the authors' own
`prepare_data.load_data` and `make_windows`, and with pandas' `round_trip` float parser, which
shares no code with the connector's pyarrow reader. Their scripts aren't in the repo yet (see Plan).

## References

The connector reads one file: `capture24.zip` from the Oxford Research Archive record
`uuid:99d7c092-d865-4a19-b096-cc16440cd001`, 6,902,652,480 bytes, sha256 `69740c22…40e2`. The
archive's `capture24/` directory holds 151 `P<nnn>.csv.gz`, `metadata.csv` and
`annotation-label-dictionary.csv`, nothing else.

Independent copies and references, which the connector doesn't read:

- **The authors' code**, `OxWearables/capture24` at `main` on 2026-10-28: `prepare_data.py`
  (`load_data`, `make_windows`, the dictionary mapping), `benchmark.py` (`train_test_split`, the
  scored schemes), `eval.py` (the metrics) and `train_deep_models.py`. Its code is under an
  academic-use licence, so the connector reimplements the windowing and copies none of it.
- **The authors' datasheet**, `data_info/1_Datasheet.pdf` in the same repository: 151
  participants, 939,017 windows, 206 annotations and six schemes, 100 train users, an AX3 on the
  dominant wrist at 100 Hz and ±8 g, 66% female, randomized timestamps.
- **The paper**, Chan et al., *Sci Data* 11, 1135 (2024), read as arXiv:2402.19229: 151
  participants, 3,883 hours, 2,562 annotated hours, 922,199 windows (P001–P100: 618,129;
  P101–P151: 304,070), the demographics of Table 2, randomized dates, and CC BY 4.0.

The raw AX3 binaries the release was made from are not public, so no value can be compared with
the sensor's own output.

## Fields

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| `time` | Every participant CSV | Segment boundaries and each Source's axis `offset_us`; the first value is the `recording_start_local` record annotation | No zone, randomized dates (#26, #27) |
| `x`, `y`, `z` | Every participant CSV | One lazy float64 Signal each per segment, in `standard_gravity` | Eight significant digits, which float32 loses (#14) |
| `annotation` | Every participant CSV | `activity_label` span annotations, one per run of one label in a segment, on the segment's three signals | An empty field is unlabelled and gets no annotation (#22) |
| `pid` | `metadata.csv` | Core `subject_id` of the record metadata | Joined to the file name (#33) |
| `age`, `sex` | `metadata.csv` | `age` and `sex` record annotations | Never task inputs (#48) |
| Raw label column | `annotation-label-dictionary.csv` | `activity_label_vocabulary` dataset annotation | 206 labels in file order |
| Six `label:<scheme>` columns | `annotation-label-dictionary.csv` | `<scheme>_vocabulary` and `<scheme>_mapping` dataset annotations | All six kept; two have tasks (#11) |
| `make_windows` rule | `prepare_data.py` | The windows, each one `ClassificationTask` per benchmarked scheme | Reimplemented, matched window for window (#41) |
| `benchmark.py` schemes and split | `benchmark.py` | `BENCHMARK_SCHEMES` and `ClassificationTask.split` | `range(101, 152)` tests P101–P151 (#43) |
| `Y_anno` (window's raw-label mode) | `make_windows` | Not a task | No benchmark scores it; the `activity_label` spans rebuild it exactly (#11) |
| `X_feats` | `features.extract_features` | Not stored | Derived from `X`, which the build holds (#11) |
| Validation set | `GroupShuffleSplit` in the model scripts | Not stored | Random, not fixed by the authors (#45) |
| Body-camera images, sleep diaries | Study | Not in the release | The datasheet says the images aren't released |
| File name | Archive | `file` in the record metadata's `capture24` extension | Provenance |
| Spec `acceleration`, `standard_gravity`, float64 | Added | Signal spec | The datasheet states g; the sleep magnitude checks it (#19) |
| 100 Hz `RegularAxis` per segment | Added | Time axis | Every step is exactly 10 ms within a segment (#25) |
| Source name `Axivity AX3 segment <n>` | Added | Source name | The device the datasheet names |
| `source_url`, `source_sha256` | Added | Dataset annotations | The pinned archive |
| Record and task ids | Added | Ids | Numbered by TimeNet when the dataset is built. The connector sets none |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | There is one release, the one the paper, the datasheet and the authors' `prepare_data.download_capture24` link. The authors' URL and the pinned URL both serve `capture24.zip`, and the bytes hash to the pinned sha256. Its 151 participants, 206 labels and 3,883 hours equal the paper's. |
| 2 | Unpinned input | ✅ Done | `release.ARCHIVE_SHA256` is passed to `ensure_archive`, and a download whose bytes differ fails (`test_download_http_rejects_a_sha256_mismatch` in the package tests). |
| 3 | Partial, corrupt or changed download | ✅ Done / ⚠️ Open | Done: the download is hashed (`test_download_http_rejects_a_sha256_mismatch`), and every build parses every recording and fails unless it holds `PINNED_COUNTS` (`test_pinned_counts_mismatch_fails`). Open: `ensure_archive` reuses an extraction without hashing it again, so a changed file that keeps the counts would pass. |
| 4 | Wrong files from the archive | ✅ Done / ☑️ Done once | Done: the participants must be exactly those `metadata.csv` lists (`test_metadata_and_recordings_must_name_the_same_participants`), and the counts are pinned (`test_pinned_counts_mismatch_fails`). Done once: the archive's `capture24/` holds the 151 recordings and the two tables, nothing else. |
| 5 | Source structure misread | ☑️ Done once | All 151 headers are `time,x,y,z,annotation`. The authors' `load_data` and an independent pandas parse read all 1,398,038,232 rows. The only missing-value spelling is an empty `annotation` field, 476,286,350 samples. The datasheet's 939,017 windows and the paper's 922,199 differ from the 934,762 that `make_windows` keeps today (#6). |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ✅ Done / ☑️ Done once / ⚠️ Open | Done: the build fails unless it holds 151 participants, 159 segments and 934,762 windows (`test_pinned_counts_mismatch_fails`). Done once: the full build holds 151 Records, 159 Sources, 477 Signals and 1,869,524 tasks, and the authors' `make_windows` keeps the same 934,762 windows. Open: the paper reports 922,199 and the datasheet 939,017. Dropping the 13,095 windows whose raw-label mode is unlabelled gives 921,667, not the paper's count. The README says so. |
| 7 | Rows dropped silently | ✅ Done / ☑️ Done once | Done: no sample is skipped; a bad timestamp, label or value raises (`test_malformed_recording_fails_with_its_path`, `test_bad_acceleration_fails_when_read`). Done once: every one of the 1,398,038,232 rows is in a segment. Of the 1,398,185 10 s bins, 163 hold fewer than 1000 samples, 463,260 are fully unlabelled, and 934,762 are kept. |
| 8 | Wrong filter or threshold | ☑️ Done once | The window filters are `make_windows`'s: exactly 1000 samples and at least one labelled sample. The authors' own function, run on every participant, keeps the same windows with the same start times. |
| 9 | Duplicate records | ✅ Done / ☑️ Done once | Done: a pid listed twice in `metadata.csv` fails (`test_malformed_tables_fail_with_their_path`). Done once: 151 distinct pids and 151 distinct file digests. |
| 10 | Missing or extra channels | ✅ Done / ☑️ Done once | Done: a header other than `time,x,y,z,annotation` fails (`test_malformed_recording_fails_with_its_path`). Done once: every segment read back has exactly `x`, `y` and `z`. |
| 11 | Source information left unused without a note | ☑️ Done once | The Fields table lists every column, file and input of the authors' code. `Y_anno` has no task but is rebuilt exactly from the spans (#41). The four unbenchmarked schemes are kept as vocabularies and mappings. The README notes what's dropped. |
| 12 | Information the connector made up | ☑️ Done once | The Fields table lists every added value: the unit and rate the datasheet states, the device name, the provenance annotations and the ids. No time zone or start time is assumed; `start_time` is unset. |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once | Done: a segment rereads its own rows (`test_signals_reread_their_segment`). Done once: every value of the full build, read back, equals pandas' `round_trip` parse: 151 recordings, 4,194,114,696 values, 0 mismatches. |
| 14 | Precision lost on a dtype change | ☑️ Done once | Values are float64, and the read-back equals the exact parse (#13). Float32, which the authors load, would change 1,335,601,892 of the values (32%), so the build keeps float64. |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | No source value is an integer type. Timestamps are int64 microseconds: the largest offset, 32.15 hours, is far below the int64 range, and every sample time read back equals the source (#26). |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: the whole fixture conversion is written, read back and compared with literal expectations (`test_storage_round_trip_keeps_the_whole_conversion`). Done once: the full build read back equals the source in every value, sample time, raw label, age, sex, start and file name, and in every task's scope, target and split (#41, #43). |
| 17 | Build isn't deterministic | ☑️ Done once | Two full builds from the same files: all 249 value shards are byte-identical, and all 18 `control.duckdb` tables hold the same rows. Only the DuckDB file's checksum in `manifest.json` differs. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ☑️ Done once | The release is calibrated in g, as the datasheet says: the median acceleration magnitude in sleep is 1.000 g over all participants, 0.881 to 1.059 per participant. |
| 19 | Wrong units or scale | ☑️ Done once | Sleep magnitude is gravity (#18). Mean ENMO rises with intensity: 0.006 g in sleep, 0.022 sedentary, 0.078 light, 0.161 moderate-vigorous. |
| 20 | Swapped or mislabeled channels | ☑️ Done once / ⚠️ Open | Done once: each read-back axis equals the CSV column of its name (#13). Open: no reference shows which body axis the AX3's `x`, `y` and `z` are; the release names only `x`, `y`, `z`. |
| 21 | Flipped sign convention | ☑️ Done once / ⚠️ Open | Done once: values read back with their sign (#13), so the connector flips nothing. Open: the AX3's axis orientation isn't documented in the release (#20). |
| 22 | Missing-value markers stored as values | ✅ Done / ☑️ Done once | Done: a missing or non-finite value fails, and only an empty field is missing, so a written `nan` or `NA` fails too (`test_bad_acceleration_fails_when_read`). Done once: 0 missing or non-finite values. The most common values are dyadic, such as −0.25, at most 2.8% of a recording's values; none is a marker. |
| 23 | Wrong categorical codebook | ✅ Done / ☑️ Done once | Done: a raw label missing from the dictionary fails (`test_malformed_recording_fails_with_its_path`). Done once: all 206 non-empty raw labels in the recordings are in the dictionary, and its six columns are those the authors' `make_windows` reads. |
| 24 | Implausible upstream values passed through without a note | ☑️ Done once | 15,339 values in 138 recordings reach 8 g or more in magnitude, up to 8.666 g, past the ±8 g range, after the authors' calibration. They are kept, and the README notes them. |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ✅ Done / ☑️ Done once | Done: a segment ends at any step other than 10 ms, and a step that doesn't increase fails (`test_converts_the_fixture`, `test_malformed_recording_fails_with_its_path`). Done once: 1,398,038,232 samples over 3,883.4 hours is 100 Hz, as the datasheet and the paper say. |
| 26 | Wrong start time or epoch | ☑️ Done once | Every `recording_start_local` equals its recording's first timestamp, and every sample time rebuilt from it and the axes equals the source, 0 mismatches. The paper says the dates are randomized, so `start_time` is unset. |
| 27 | Wrong time zone | ☑️ Done once | The source states no zone, and the build claims none. Its time of day is local: labelled sleep is 97–100% of the labelled samples from 00:00 to 04:59, and 0% from 10:00 to 20:59. |
| 28 | Daylight-saving errors | ☑️ Done once | Within a segment every step is exactly 10 ms, so no hour repeats or goes missing. The one recording that spans a 2016 clock change on its randomized date, P037, has no gap. The two one-hour gaps (P016, P017) don't fall on a clock change. |
| 29 | Gaps or missing values filled or dropped silently | ✅ Done / ☑️ Done once | Done: a gap starts a new Source with its own offset, and no window crosses one (`test_converts_the_fixture`). Done once: the eight gaps, 1.0 to 3.9 hours, are 159 Sources over 151 Records, and nothing is filled. |
| 30 | Artifacts from resampling or interpolation | ⚪ N/A | The connector doesn't resample. The authors resampled the release to 100 Hz before publishing it, and the raw binaries aren't public. |
| 31 | Off-by-one sample | ☑️ Done once | Every value and sample time of the full build equals the source at its index (#13, #26), and every window's 1000 samples equal the authors' `X` exactly (#41). |
| 32 | Sources or channels out of sync | ☑️ Done once | A segment's three axes share one axis, and each row's `x`, `y` and `z` read back at the same index and time as the source (#13). |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ✅ Done / ☑️ Done once | The labels are on the same row as the values, and `metadata.csv` joins by pid to the file name, never by order (`test_converts_the_fixture`). Done once: every Record's age and sex equal its own pid's row. |
| 34 | Off-by-one id | ☑️ Done once | Every Record's `subject_id` and `file` name the same participant, and every task's input is the Record of the windows the authors cut from that file (#41). |
| 35 | Ids or join keys not unique | ✅ Done / ☑️ Done once | Done: a repeated pid or raw label fails (`test_malformed_tables_fail_with_their_path`). Done once: 151 distinct pids and 206 distinct raw labels. |
| 36 | Subject metadata on the wrong recording | ☑️ Done once | One recording per participant, and each Record's age and sex equal its pid's row in `metadata.csv`, all 151. The demographics equal the paper's Table 2: 99 women; 43, 37, 37 and 34 per age band. |
| 37 | Join silently drops rows | ✅ Done / ☑️ Done once | Done: a pid without a recording, or a recording without a pid, fails (`test_metadata_and_recordings_must_name_the_same_participants`). Done once: 151 rows, 151 recordings, 151 Records. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ✅ Done / ☑️ Done once | Done: the classes come from the dictionary's columns, and ties go to the class first in string order (`test_converts_the_fixture`). Done once: every target equals the authors' `make_windows` label, which maps through the same columns, in both schemes. |
| 39 | Label distribution differs from the paper | ☑️ Done once / ⚠️ Open | Done once: the class counts per split equal the authors' code's, and sleep is 36% of windows, as the paper's "more than a third". Open: the paper gives its class balance only as a figure, so the counts weren't compared with it, and its window total differs (#6). |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | One task per 10 s window and scheme, as the paper and `make_windows` define it. The raw labels stay per sample, as spans. |
| 41 | Derived target computed wrongly | ☑️ Done once | The build, read back, rebuilds `make_windows`'s output exactly for all 151 participants: `X` (float32, 934,762 × 1000 × 3), `T`, `Y_WillettsSpecific2018`, `Y_Walmsley2020` and `Y_anno`, 0 mismatches. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once / ⚠️ Open | Done once: sleep has gravity-only magnitude, and mean ENMO rises from sleep to moderate-vigorous (#19); sleep labels fall at night (#27). Open: no model check against shuffled labels was run. |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ✅ Done / ☑️ Done once | Done: P001–P100 are train and P101–P151 test (`test_split_follows_benchmark_py`, `test_tasks_of_selects_the_benchmark_split`). Done once: every task's split equals `benchmark.train_test_split`, compiled from the authors' file, and the paper's sets; 626,314 train and 308,448 test windows. The repository README's P102 is wrong. |
| 44 | Leakage across splits | ☑️ Done once | A participant has one recording and one split, so no subject spans splits. Window starts are at least 10 s apart in every recording, so no windows overlap. |
| 45 | Splits regenerated without a fixed seed | ✅ Done | The split is a function of the pid, with no randomness (`test_split_follows_benchmark_py`). The authors' random validation split isn't reproduced. |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | The paper and `benchmark.py` classify each window into a scheme's classes, so `ClassificationTask`, with the scheme's vocabulary as `target_schema`. |
| 47 | Task can't be solved from its inputs | ☑️ Done once / ⚠️ Open | Done once: the authors classify from the window's acceleration alone, which the task's scope covers, and the build holds their exact `X` (#41). Open: no baseline was trained on the build. |
| 48 | A field in the wrong role | ☑️ Done once | The Fields table gives each role. The scope names the window and no input annotation, so the label spans, age and sex aren't task inputs. Pid and file name are record metadata. |
| 49 | Wrong prompt, or one without a source | ⚪ N/A | The source and the authors' benchmark state no question in words. The tasks have no prompt. |
| 50 | Series placed at the wrong point in the prompt | ⚪ N/A | No prompt (#49). |
| 51 | An input reveals the target | ✅ Done / ☑️ Done once / ⚠️ Open | Done: the scope is the input boundary. No task names an annotation in `input_annotations`, and its scope is one window of one segment (`test_tasks_mark_only_their_window_as_input`). Done once: all 1,869,524 tasks of the full build, 0 exceptions. Open: every target derives from the `activity_label` spans on the input Record's signals. TimeNet's torch view leaves signal annotations out of its items, but a loader that turns them into model context passes the answer. The README says so. |
| 52 | The target leaks through an id, metadata or another field | ☑️ Done once / ⚠️ Open | Done once: record metadata holds only the pid and file name, and the record annotations `age`, `sex` and `recording_start_local` say nothing about a window. Open: a torch item holds every task of its Record, targets included, so a consumer must keep the other tasks from the model. The README says so. |
| 53 | An input carries information from after the target | ⚠️ Open | The scope is the window's own 10 s, but the input Record holds the whole day, and no TimeNet loader cuts a signal to the scope. Adjacent windows share their class in 99.995% of 921,616 pairs (Walmsley2020) and 99.990% (WillettsSpecific2018), so the surrounding signal nearly gives the class away. The scope names no signal on purpose: the authors' models use different context for the same target, the window alone (RF, XGBoost, CNN), 8 consecutive windows read in both directions (RNN, `seq_length: 8`), or the participant's whole sequence (`+ HMM`). The README gives each row's input and tells per-window models to cut to the scope. A library helper that returns a task's values cut to its scope would close this. |
| 54 | Answer format doesn't match what the evaluation expects | ☑️ Done once | The authors' `eval.metrics_report`, given the read-back targets of all 308,448 test tasks per scheme and their own labels, scores 1.000 on all four metrics. A target reads as `str` through `read_value`, as the README says. |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ☑️ Done once | The Oxford Research Archive record and the paper both state CC BY 4.0 for the data. The authors' code is under a separate academic-use licence, which the connector avoids by copying none of it. |
| 56 | Wrong domain, tags or description | ☑️ Done once | `activity` and `motion`, and the description matches the datasheet: wrist accelerometry from 151 participants, labelled from camera images and sleep diaries. |
| 57 | A README number that doesn't match the build | ☑️ Done once | Every README number comes from the passes above or the pinned counts. Only the counts are checked by the build (`test_pinned_counts_mismatch_fails`). |
| 58 | A README claim the data doesn't support | ☑️ Done once | Each claim was checked by a pass above; the earlier claim that timestamps are local wall-clock time was corrected to the paper's randomized dates. |
| 59 | Known limitations left undocumented | ☑️ Done once | The README notes the window counts, the gaps, the values past ±8 g, the split typo, the broken deep-model script, and that a per-window model must cut to the scope, with what the rest of the Record gives away (#51 to #53). |

## Open items

1. **The scope is the input boundary, and nothing applies it (#51, #52, #53).** A task's input is
   its participant's whole Record: the day's signal, its `activity_label` spans and, in a torch
   item, every other task's target. The connector keeps the whole Record so that one copy of the
   values serves the per-window and the `+ HMM` rows, as sleep_edfx does with its epochs, and the
   README tells per-window models to cut to the scope. A TimeNet helper that returns a task's
   values cut to its scope, and leaves out annotations no task names as input, would close this
   for every connector built this way.
2. **The paper's window count (#6, #39).** The paper's 922,199 and the datasheet's 939,017 both
   differ from the 934,762 the authors' code keeps today. Ask the authors which windowing the
   paper used, and compare class balance with the paper's figure.
3. **A cached extraction isn't rehashed (#3).** `ensure_archive` hashes the download but not a
   reused extraction. Pinning per-file digests in `release.py` would close it, at the cost of
   rereading 6.9 GB each build.
4. **Model check (#42, #47).** Train the authors' random forest, or a linear probe, on the train
   tasks, and compare test macro F1 with the paper's (0.711 for Walmsley2020, 0.388 for
   WillettsSpecific2018).
5. **Axis orientation (#20, #21).** The release doesn't say which body axis `x`, `y` and `z` are.

## Plan

- Commit the three one-off scripts, the source pass, the read-back against the authors' code and
  the determinism comparison, under a `real_data` pytest marker that skips without the archive.
- Run the model check of item 4 with the authors' `rf.py` on the read-back windows.
- Propose the scope helper of item 1 to core, as its own issue, since sleep_edfx has the same
  shape.
