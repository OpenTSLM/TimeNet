# Verification: `bash-lab/sensorcaps`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in `tests/test_connector.py` unless stated otherwise. The ☑️ Done once
checks are one-off passes run on 2026-10-23 over the pinned Hugging Face files, two full builds of
this connector, and the independent copies listed under References. Their scripts aren't in the
repo yet (see Plan).

## References

The connector reads the five `data/*_v2-00000-of-00001.jsonl` files of the Hugging Face dataset
`BASH-Lab/SensorCaps` at revision `4242e07` (35,960 rows). Independent copies of the same data,
which it doesn't read:

- **The v1 files of the same revision.** `hhar_v1`, `motion_v1`, `shoaib_v1` and `uci_v1`, 26,288
  rows. They hold the same windows with shorter captions (500 tokens) and no feature text.
- **LIMU-BERT's preprocessed arrays**, `dataset/<name>/data_20_120.npy` and `label_20_120.npy` at
  commit `decffee` of LIMU-BERT-Public, with its `data_config.json`. These are the arrays the
  authors' generator loads for HHAR, MotionSense, Shoaib and UCI: 120 samples at 20 Hz, with
  activity and user labels.
- **The authors' code** at commit `bf4d00d` of BASHLab/LLaSA: `llasa_v2_data_generation.py`,
  `llasa_v2_data_generation_pamap2.py`, `data/dataset_activity_label.json`, and
  `OpenSQA-Parsing-per-Dataset.ipynb`. The PAMAP2 per-location `.npy` files the PAMAP2 generator
  reads aren't published, and neither is the code that made them.
- **Upstream PAMAP2**, the UCI archive (`PAMAP2_Dataset.zip`, sha256 `6c2282b5…06f4`), `Protocol/`
  files of subjects 101–108 at 100 Hz.
- **The paper**, arXiv:2406.14498 v4, and the Hugging Face card, which both give 35,960 pairs.

## Fields

What every field of the source and of the authors' code becomes, then every value the connector
adds. "Row" is one JSONL line, a chat of system, user and assistant messages.

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| File name | `data/<partition>-00000-of-00001.jsonl` | Record id, `source_dataset`, `source_partition` and `source_file` annotations | As published |
| Line number | Row order in the file | Record id, `source_row` annotation | Row order is the authors' window order: LIMU-BERT array order, or PAMAP2 subject, then location, then window (#36) |
| System message | Row | Start of the task prompt | The authors' generator, verbatim. It describes sections the prompt no longer holds (#49) |
| `Summary:` | User message | `activity_summary` annotation, a task input annotation | Names from `dataset_activity_label.json` and LIMU-BERT's label order (#38). PAMAP2's map isn't in the repo |
| `Gyroscope:`, `Accelerometer:` | User message | Six float32 Signals, 60 values each | LIMU-BERT arrays for four partitions (#5). PAMAP2 columns identified upstream (#20) |
| Moving average, FFT, minimum, maximum, median, variance | User message | Prompt text after the system message | Computed by the authors over all 120 samples. FFT, minimum, maximum, median and variance run across the three axes at each step, not over time (#41) |
| `Location:` | User message, PAMAP2 only | `sensor_location` annotation, a task input annotation | The PAMAP2 generator's `ankle`, `chest`, `hand` |
| Assistant message | Row | Caption target | GPT-4o-mini output, kept verbatim. Can't be regenerated (#41) |
| `*_v1` files | Same revision | Not read | Same values for all 26,288 rows. Only "decending" is spelled differently (#5). The README doesn't mention them |
| `README.md`, `.gitattributes`, `refs/convert/parquet` | Hugging Face | Not read | Not data, or a converted copy |
| The 60 dropped samples of each window | LIMU-BERT arrays, PAMAP2 | Not in the release | The features depend on them, so they can't be regenerated from the 60 kept values |
| User, device model and body position labels | LIMU-BERT `label_20_120.npy` | Not in the release | HHAR: 9 users, 3 models. MotionSense 24 users, UCI 30, Shoaib 10 users and 5 positions of 2,100 windows each. The generator drops them, and the README doesn't say so |
| PAMAP2 subject | Row order: 8 blocks of 417, 440, 289, 385, 454, 417, 385 and 437 windows, each repeated for 3 locations | Not exposed | Only recoverable from `source_row` (#36, #44) |
| Time offsets | Added | Irregular time axis, `round(linspace(0, 119, 60)) / 20` s | The generator's `sensor_subsampled_string` and LIMU-BERT's `sr: 20`, `seq_len: 120`. Checked on every row (#25, #31) |
| `declared_sampling_rate_hz` 10, `source_sampling_rate_hz` 20, `source_window_samples` 120, `time_axis_reconstruction` | Added | Dataset annotations | Hugging Face card for 10 Hz; LIMU-BERT config and the PAMAP2 fit for 20 Hz |
| PAMAP2 signal names, magnetometer spec | Added | Signal names and specs | Matched against upstream PAMAP2 (#20) |
| Spec types, `unit_value` None | Added | Signal specs | Units left unset, though the data fits m/s² and rad/s (#19) |
| Source name `IMU`, record, source, signal and task ids | Added | Ids | Built from partition and row |
| Split | Added | `train` for every task | The source has no splits (#43) |
| Prompt join | Added | `"\n\n"` between the system message and the feature text | Formatting |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | The repository has three commits; `4242e07` is `main` and the latest. The authors' generator writes `sensorcaps_10hz_<name>_v2.jsonl`, the v2 files read here. Their 35,960 rows equal the paper's and card's total, and the four LIMU-BERT partitions equal `data_config.json`'s sizes exactly (9,166, 4,534, 10,500 and 2,088). |
| 2 | Unpinned input | ☑️ Done once / ⚠️ Open | Done once: `SOURCE_REVISION` pins the commit, and the build's snapshot came from `snapshots/4242e07…`. The five files hash to the Hub's LFS sha256 at that revision. Open: no test checks that `download` passes the pin to `hub_snapshot`. `test_conversion_preserves_source_semantics_and_reiterable_tasks` only checks the `source_revision` annotation. |
| 3 | Partial, corrupt or changed download | ✅ Done / ⚠️ Open | Done: every build checks each file's row count against `SOURCE_ROWS` (`test_pinned_downloads_enforce_partition_counts`), and every row must parse (`test_invalid_sensor_data_has_source_context`). Open: no sha256 is pinned in the connector, so a file edited in place with the same row count and valid rows would pass. |
| 4 | Wrong files from the archive | ✅ Done / ☑️ Done once | Done: `download` asks only for the five `SOURCE_ROWS` files, and each one's row count is checked (`test_pinned_downloads_enforce_partition_counts`). Done once: the revision holds 11 files. The build reads the five v2 files and none of the four v1 files, `README.md` or `.gitattributes`. |
| 5 | Source structure misread | ☑️ Done once | LIMU-BERT arrays, all 26,288 rows of four partitions: each published triple equals `round(data[i][round(linspace(0, 119, 60))], 6)`, `0:3` as `Accelerometer` and `3:6` as `Gyroscope`, with 0 mismatches. The same values swapped between sensors match 0 rows, and against the previous window 0 rows. v1 against v2, by row: values equal in 26,288 of 26,288; labels equal except "decending stairs" in v1 for all 3,204 "descending stairs" rows. Every one of the 35,960 rows has one system prompt per partition and the same 15 user headings in the same order, plus `Location` in PAMAP2. |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ✅ Done | `SOURCE_ROWS` pins 9,166, 4,534, 9,672, 10,500 and 2,088 rows, and every build checks them (`test_pinned_downloads_enforce_partition_counts`). The sum, 35,960, is the paper's. A full build holds 35,960 Records and 35,960 tasks. |
| 7 | Rows dropped silently | ✅ Done | Every row becomes a Record or raises with its file and row (`test_invalid_sensor_data_has_source_context`), and the pinned counts catch a short file. The full read-back found a Record for each of the 35,960 rows, with none left over. |
| 8 | Wrong filter or threshold | ✅ Done / ☑️ Done once | Done: the connector filters nothing; a row with other than 60 readings raises (`test_unexpected_sample_count_rejects_timing_reconstruction`). Done once: the authors' only filter is PAMAP2's skip of activity 0, and all 48 PAMAP2 windows matched upstream fall in labeled activities (#20). |
| 9 | Duplicate records | ✅ Done / ⚠️ Open | Done: record ids come from partition and row, and `add_record` rejects a repeated one. The read-back found 35,960 distinct ids. Open: in MotionSense, rows 828–861 repeat rows 1162–1195 exactly, 34 pairs with the same label and different captions. The README doesn't say so. Also, each PAMAP2 window appears three times, once per location, and LIMU-BERT's Shoaib holds each moment at five body positions (#44). |
| 10 | Missing or extra channels | ✅ Done / ☑️ Done once | Done: each sensor needs 60 triples, and fixture records have six Signals with the expected names (`test_invalid_sensor_data_has_source_context`, `test_conversion_preserves_source_semantics_and_reiterable_tasks`). Done once: every row of the build has six Signals, named as the README says. The authors take `0:6` of LIMU-BERT's Shoaib array, which has 9 channels, so its magnetometer is dropped upstream. |
| 11 | Source information left unused without a note | ☑️ Done once / ⚠️ Open | Done once: the Fields table lists every file and field of the release and every input of the authors' code. Open: the README doesn't mention the v1 files, the user, device and body position labels the authors dropped, or that PAMAP2 subjects are recoverable only from row order. |
| 12 | Information the connector made up | ☑️ Done once / ⚠️ Open | Done once: the Fields table cites a reference for each added value. The time offsets and 20 Hz rest on LIMU-BERT's config and the feature check of #25. The PAMAP2 names rest on the upstream match. Open: units are left unset as if unknown, which the data contradicts (#19). |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once | Done: `test_channel_reads_reuse_rows_and_reread_offsets` rereads evicted rows by byte offset, with UTF-8 text. Done once: a full build read back through `TimeNet.load` against `json.loads` of the files. All 35,960 Records, all six channels, as float32: 0 mismatches. |
| 14 | Precision lost on a dtype change | ☑️ Done once | Source values have at most 6 decimals (0 values with more). Rounded back to 6 decimals, float32 fails to restore 243,916 of 12,945,600 values: 233,680 in PAMAP2, 8,403 MotionSense, 1,259 HHAR, 574 UCI, 0 Shoaib. The largest error is 7.6e-6, in PAMAP2's magnetometer, whose values reach 176.8. The README documents float32 rounding. |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | No source value is converted to an integer. The only integers are the added offsets, 0 to 5,950,000 µs in int64, which the read-back matched on all 215,760 Signals. |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: `test_pipeline_preserves_source_semantics_after_source_removal`. Done once: the full read-back (#13) also compared offsets, annotations, and every task's id, input, target, prompt, input annotations and split, with 0 mismatches. |
| 17 | Build isn't deterministic | ☑️ Done once | Two full builds from the same revision. The three value files are byte-identical. All 16 `control.duckdb` tables hold the same rows, apart from the writer's generated ids in `annotation_occurrences.occurrence_id` and `axes.axis_id`. The manifests differ only in that file's checksum and size. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ⚪ N/A | The source publishes physical values. Median accelerometer norms are 9.74–10.02, and the PAMAP2 match fits the upstream values without scaling. |
| 19 | Wrong units or scale | ☑️ Done once / ⚠️ Open | Done once: median accelerometer norm is 9.742 (HHAR), 9.854 (MotionSense), 9.914 (PAMAP2), 9.881 (Shoaib) and 10.023 (UCI), so m/s². Static labels give a gyroscope standard deviation of 0.007–0.068, and motion labels 0.36–2.4, which fits rad/s. PAMAP2 matches its upstream m/s², rad/s and µT columns unscaled (#20). Open: the README says the source has no consistent units and leaves `unit_value` unset. The four LIMU-BERT partitions aren't compared with their own upstream recordings. Needs a decision. |
| 20 | Swapped or mislabeled channels | ☑️ Done once | LIMU-BERT partitions: all 26,288 rows equal the arrays in channel order, with 0 swapped matches (#5). The authors' minimum, maximum, median and variance agree with their sensor's values in 35,960 of 35,960 rows, and with the other sensor's in 0. PAMAP2: 48 seeded windows, 6 per subject over 3 locations, fit to the Protocol files under each of 121 assignments of three consecutive IMU columns to each sensor. The README's assignment, `acc16_y`, `acc16_z`, `acc6_x`, `gyro_y`, `gyro_z`, `mag_x`, fits best in all 48, with an RMS error of 0.20 channel standard deviations at the median and 0.51 at most. The assignment the prompt implies, 16 g and gyroscope XYZ, gives a median of 3.19. Within an assignment, 16 g and 6 g columns are hard to tell apart (#59). |
| 21 | Flipped sign convention | ☑️ Done once / ⚠️ Open | Done once: the PAMAP2 fit matches each channel with its upstream sign, and the LIMU-BERT partitions equal LIMU-BERT's arrays. Open: LIMU-BERT's signs aren't compared with the original HHAR, MotionSense, Shoaib and UCI recordings. |
| 22 | Missing-value markers stored as values | ✅ Done / ☑️ Done once / ⚠️ Open | Done: NaN and infinity raise (`test_invalid_sensor_data_has_source_context`). Done once: no value equals −999, 999, −9999, 9999, −99, 99, 32767 or −32768. The most repeated values are 0 and small gyroscope steps. Open: PAMAP2 gaps were filled upstream, so they are values now (#29). |
| 23 | Wrong categorical codebook | ☑️ Done once | Each partition's label set equals `dataset_activity_label.json`, and its order equals LIMU-BERT's `activity_label` lists. PAMAP2's twelve names are the protocol's activities, and all 48 matched windows carry their upstream activity ID's name. Sensor locations are the generator's three. |
| 24 | Implausible upstream values passed through without a note | ☑️ Done once / ⚠️ Open | Done once: Shoaib's accelerometer peaks at 19.60 m/s², with 5,333 values in 1,319 rows at 19.5 or more, so it clips at about 2 g. UCI peaks at 19.61. PAMAP2 reaches 95.8 m/s² and MotionSense 57.0 in running and jogging, inside their sensors' ranges. Open: the README doesn't mention the clipping. |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ✅ Done / ☑️ Done once | Done: the offsets are pinned (`test_conversion_preserves_source_semantics_and_reiterable_tasks`), and a row of other than 60 readings raises (`test_unexpected_sample_count_rejects_timing_reconstruction`). Done once: every row's feature arrays hold 120 steps, and at `round(linspace(0, 119, 60))` they equal the published values in 35,960 of 35,960 rows. Shifted by one step, 0 rows match. LIMU-BERT gives 20 Hz for its four. PAMAP2 fits 100 Hz upstream at 5 samples per step, so 20 Hz. The card's 10 Hz is the mean rate of the 60 kept values, and the README explains it. |
| 26 | Wrong start time or epoch | ⚪ N/A | The source has no timestamps. Records carry relative offsets only, and every `start_time` reads back as `None`. |
| 27 | Wrong time zone | ⚪ N/A | No clock time exists in the source. |
| 28 | Daylight-saving errors | ⚪ N/A | No clock time exists in the source. |
| 29 | Gaps or missing values filled or dropped silently | ☑️ Done once / ⚠️ Open | The connector fills nothing; the offsets keep the step skipped at index 60. Upstream PAMAP2 did fill gaps: 18 of the 48 matched 6 s spans have NaN in the six source columns, and they fit with NaN read as 0, though no other reading was tried. The README gives 3,231 of 9,672 windows. Open: that count was not reproduced here, and nothing marks the affected records. |
| 30 | Artifacts from resampling or interpolation | ☑️ Done once / ⚠️ Open | The connector doesn't resample. Done once: the LIMU-BERT partitions are exact picks from LIMU-BERT's 20 Hz arrays. PAMAP2 is not an exact pick from the 100 Hz files: 0 of 288 channels match to 1e-5. The mean of each 5-sample block fits better than every fifth sample in all 48. Open: the authors' PAMAP2 resampler isn't published, so its artifacts are undocumented. |
| 31 | Off-by-one sample | ☑️ Done once | The index check of #25 matches all 35,960 rows and fails for every row shifted by one step. The read-back matches every offset. |
| 32 | Sources or channels out of sync | ✅ Done / ☑️ Done once | Done: the six Signals share one axis (`test_conversion_preserves_source_semantics_and_reiterable_tasks`). Done once: both sensors match their features at the same 60 steps in every row (#25). In PAMAP2, the ankle, chest and hand copies of each of 16 matched windows start at the same upstream sample. |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ✅ Done / ☑️ Done once | Values, label, location and caption share one JSON line. Tasks reread the files and find their Record by partition and row. Done: fixture rows have distinct values and text, and each task's target, prompt and Record are checked per row on two passes (`test_conversion_preserves_source_semantics_and_reiterable_tasks`). Done once: in the full build, every task matched its own line, 35,960 of 35,960. |
| 34 | Off-by-one id | ☑️ Done once | Row order equals LIMU-BERT's array order in all 26,288 rows, and a shift by one matches 0 (#5). The labels agree with LIMU-BERT's by row in 26,288 of 26,288. Captions name their own label verbatim in 34,870 of 35,960 rows: 8,839 of 9,166 HHAR, 4,405 of 4,534 MotionSense, 9,521 of 9,672 PAMAP2, 10,089 of 10,500 Shoaib and 2,016 of 2,088 UCI. |
| 35 | Ids or join keys not unique | ✅ Done | Partition names are unique, which `convert` enforces, and `add_record` rejects a repeated id. The read-back found 35,960 distinct record ids and 35,960 tasks. |
| 36 | Subject metadata on the wrong recording | ☑️ Done once | The only metadata, the location, comes from the same line. PAMAP2 rows form 8 subject blocks of ankle, chest and hand, each of equal length, 3,224 per location in total, with identical label sequences in all 8. |
| 37 | Join silently drops rows | ✅ Done | A task without its Record raises a `KeyError`, and the row counts are pinned (`test_pinned_downloads_enforce_partition_counts`). One-off: 35,960 tasks and 0 Records without a task. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ☑️ Done once | The authors' map, applied to LIMU-BERT's label at index 2 for HHAR and 0 for the others, as the generator does, gives the published label for 26,288 of 26,288 rows. The map's order equals LIMU-BERT's `activity_label` lists. PAMAP2's map isn't in the repo: the generator reads a `PAMAP2` key that `dataset_activity_label.json` lacks. All 48 matched windows agree with their upstream activity. |
| 39 | Label distribution differs from the paper | ☑️ Done once / ⚠️ Open | Done once: HHAR, MotionSense, Shoaib and UCI equal LIMU-BERT's labels row by row (#38). Open: the paper reports no class balance. HHAR holds 2,079 walking, 1,909 sitting, 1,497 standing, 1,435 climbing, 1,198 biking, 1,048 descending. PAMAP2 holds 12 classes, from 1,197 ironing to 213 rope jumping. Nothing records these for a later build (plan D). |
| 40 | Wrong granularity, such as per record instead of per epoch | ☑️ Done once | One label per window, from the window's first step in the generator. LIMU-BERT's activity label is constant across the window in MotionSense, Shoaib and UCI. HHAR's isn't constant in every window, and the generator takes step 0. |
| 41 | Derived target computed wrongly | ☑️ Done once / ⚠️ Open | Done once: the authors' feature functions, rerun on LIMU-BERT's arrays, reproduce the feature text byte for byte in 22,514 of 26,288 rows, and within one unit of the sixth decimal in the other 3,774. The FFT, minimum, maximum, median and variance are taken across the three axes at each step, not over time: the 3-point FFT matches in 20,000 of 20,000 checks. The system message says they are "computed for each axis". Open: the captions are GPT-4o-mini output and can't be regenerated, and PAMAP2's features can't be rebuilt without the authors' arrays. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once / ⚠️ Open | Done once: a gradient-boosted classifier on 49 summary features, cross-validated with neighbouring rows, or whole PAMAP2 subjects, kept together. Accuracy is 0.987, 0.969, 0.752, 0.941 and 0.968 for HHAR, MotionSense, PAMAP2, Shoaib and UCI. With shuffled labels it is 0.191, 0.218, 0.106, 0.143 and 0.157, near the majority class. Open: some captions contradict their label, for example 2 "descending stairs" captions describe only climbing. No count over all labels yet. |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ✅ Done / ☑️ Done once | Done: every task is `train` (`test_conversion_preserves_source_semantics_and_reiterable_tasks`). Done once: the release, card and paper define no SensorCaps split, and the full build holds 35,960 `train` tasks. |
| 44 | Leakage across splits | ⚪ N/A | There is one split. Anyone who cuts one must group the 3 PAMAP2 locations of a window, the 5 Shoaib positions, subjects, and the 34 MotionSense duplicate pairs (#9, #59). |
| 45 | Splits regenerated without a fixed seed | ⚪ N/A | Nothing generates a split. |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | One caption per window, the system and user messages as input and the assistant message as target, which is how the generator builds each row and the card's "sensor-text-to-text". Reviewed against the generator and the card. |
| 47 | Task can't be solved from its inputs | ☑️ Done once / ⚠️ Open | Done once: the activity named in 96–98% of captions per partition comes from the `activity_summary` input, the numbers quoted come from the series and the feature text, and the location from `sensor_location`. Open: PAMAP2 captions describe channels that aren't there. 6,149 say "x-axis" and 9,513 "gyroscope", and 1 mentions a magnetometer, while the sixth channel is one. The model check (#42) shows the series supports the label. No caption baseline has run (plan G). |
| 48 | A field in the wrong role | ✅ Done / ☑️ Done once | Done: provenance stays in record annotations, and the task's input annotations are only the activity and location (`test_conversion_preserves_source_semantics_and_reiterable_tasks`). Done once: the same holds for all 35,960 tasks of the full build, with no rationale. |
| 49 | Wrong prompt, or one without a source | ✅ Done / ⚠️ Open | Done: the prompt is the source's system message, then the feature text, and the tests pin it (`test_conversion_preserves_source_semantics_and_reiterable_tasks`). The full build matched all 35,960. Open: the system message tells the model to find `Summary:`, `Gyroscope:`, `Accelerometer:` and, in PAMAP2, `Location:` in the text, which now arrive as annotations and series. It also says the features are per axis (#41). |
| 50 | Series placed at the wrong point in the prompt | ⚠️ Open | The prompt has no placeholder, so the renderer places the series. In the source, the series came between `Summary` and the features. No rendered task has been read yet (plan G). |
| 51 | An input reveals the target | ☑️ Done once | By design. The activity label is an input and the README says so, and 34,870 of the 35,960 captions name it verbatim. The prompt doesn't contain the label in any of the 35,960 tasks. The caption is never an input. |
| 52 | The target leaks through an id, metadata or another field | ☑️ Done once | A loader passes the record id and the record annotations besides the inputs. The annotations are partition, file, row, activity and location. The activity and location are the task's own inputs, and the others are provenance with no caption text. Rows are sorted by label in MotionSense, with 6 runs for 6 labels, so `source_row` follows the label there. No input carries it and no split is cut from the ids, so under the rule this isn't a leak, and it reveals only the activity, which is an input anyway. |
| 53 | An input carries information from after the target | ☑️ Done once | The features come from the same 120-sample window as the series (#25), and the caption was written from those inputs. Nothing comes from after the window. |
| 54 | Answer format doesn't match what the evaluation expects | ⚠️ Open | TimeNet has no evaluation for this task. Targets are free text with `Features:` and `Narration:` sections; 2 lack the word "Narration", and one Shoaib caption of 502 characters ends mid-sentence. Needs a dummy evaluation (plan G). |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ⚠️ Open | The card's `other` and its URL equal the Hugging Face card's Hippocratic License 3.0 link and the LLaSA README badge. The repository has no root license file, and its `LLaSA/` model code is Apache 2.0. The upstream datasets keep their own terms, which nobody has compared with the Hippocratic modules. The card already asks for legal review. |
| 56 | Wrong domain, tags or description | ✅ Done / ☑️ Done once | Done: the card validates (`test_every_connector_ships_a_valid_dataset_card` in the package tests). Done once: the domains, tags and description were reviewed. The PAMAP2 channel claim held (#20), and the zero filling is consistent with a sample (#29). |
| 57 | A README number that doesn't match the build | ☑️ Done once / ⚠️ Open | Done once: 35,960 records, 9,672 PAMAP2 examples, 60 values per signal and 0–5.95 s offsets match the build. Open: the 3,231 PAMAP2 windows with missing values wasn't reproduced (#29). No test ties README numbers to the build (plan D). |
| 58 | A README claim the data doesn't support | ⚠️ Open | Found three. The feature text includes "NumPy ellipses", but 0 of 35,960 user messages contain one. The features "cannot be regenerated from the 60 readings alone", but the FFT, minimum, maximum, median and variance at the 60 kept steps can, as #41 did. The source "does not establish consistent physical units", but every partition fits m/s² (#19). The other claims held: the PAMAP2 channels, the time offsets, features over 120 samples, and LLaSA's training dividing acceleration by 9.8. |
| 59 | Known limitations left undocumented | ⚠️ Open | The README has a "Data limitations" section, which leaves out several findings. The features run across axes, not over time (#41). The 34 MotionSense duplicate pairs (#9). Repeated windows across PAMAP2 locations and Shoaib positions (#44). Shoaib's clipping at 2 g (#24). The dropped user and position labels (#11). PAMAP2's unknown resampler (#30). PAMAP2's 16 g and 6 g columns can't be told apart by value alone, so the names rest on the column layout (#20). |

## Open items

Most important first.

1. **Feature semantics (#41, #49, #59).** The FFT, minimum, maximum, median and variance in every
   prompt are taken across the three axes at each of 120 steps, not over time per axis, and the
   system message says otherwise. Document this in the README. Whether to keep the system message
   as published is a decision for maintainers.
2. **Units (#19, #12, #58).** All five partitions fit m/s² and rad/s, and PAMAP2's magnetometer
   µT. Decision: set `unit_value`, or keep it unset and correct the README's reason.
3. **README corrections (#11, #24, #58, #59).** The ellipses, the regeneration claim, the v1
   files, the dropped labels, the MotionSense duplicates, repeated windows across positions, and
   Shoaib's clipping.
4. **PAMAP2 missing values (#22, #29, #57).** Reproduce the 3,231 count over all 9,672 windows,
   and decide whether to flag those records.
5. **PAMAP2 captions (#47).** They describe the wrong channels. The README says so. Decide whether
   to keep PAMAP2 in captioning evaluations.
6. **Pin checks (#2, #3).** Test that `download` passes `SOURCE_REVISION`, and pin each file's
   sha256.
7. **Upstream for the LIMU-BERT partitions (#19, #21).** Compare LIMU-BERT's arrays with the
   original HHAR, MotionSense, Shoaib and UCI recordings for signs and axes.
8. **Repeatable checks on the real data.** The one-off passes below ran once. Commit them (Plan).
9. **Rendered tasks and evaluation (#50, #54).** Plan G.
10. **Decisions for maintainers:** the license once reviewed (#55), and a grouped split if one is
    ever offered (#44).

## Plan

Checks on the real data run in a `real_data` pytest marker, skipped unless an environment variable
points at the downloaded files. Every check below except D, F and G ran once for this record.

- **A1. LIMU-BERT arrays.** Compare every HHAR, MotionSense, Shoaib and UCI row with
  `data_20_120.npy` at the 60 kept steps, and every label with `label_20_120.npy` through the
  authors' map.
- **A2. The authors' features.** Rerun their feature functions on the LIMU-BERT arrays and compare
  the text, and check the per-step features against the published values in every partition.
- **A3. v1 against v2.** Compare values and labels row by row.
- **A4. Upstream PAMAP2.** Fit seeded windows to the Protocol files under every column
  assignment. Extend it to all 9,672 windows, to count the NaN spans and to test the resampler.
- **B. Model check.** Rerun the classifier against shuffled labels per partition.
- **C. Full read-back and determinism.** Compare a full build with the files, and two builds with
  each other.
- **D. Stats snapshot.** Commit the counts, label balance, value ranges, clipping counts and
  duplicate groups, and have a test compare them.
- **F. Corruption tests.** Swap the two sensors, shift the offsets by one step, and move every
  caption one row over. Each must fail a check above.
- **G. Review sheet and evaluation.** Render 10 seeded tasks, two per partition, with their series,
  and run a dummy caption evaluation.
