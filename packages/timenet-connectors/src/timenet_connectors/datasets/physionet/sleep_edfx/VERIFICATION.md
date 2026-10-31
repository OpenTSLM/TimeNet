# Verification: `physionet/sleep-edfx`

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

Tests named below are in this connector's `tests/` folder, or in the package tests
(`test_edf_reader.py`, `test_cards.py`). Every one of them runs on synthetic fixtures. A ✅ Done
row means the rule the test pins also runs on the real data at build time, usually as a raise.
The ☑️ Done once checks are one-off passes run on 2026-10-23. They used the S3 archive the
connector downloads, extracted by the connector's own `download_async`, and two full builds of
this connector, read back through the SDK (`timenet.client.TimeNet(...).load`). Their scripts
aren't in the repo yet (see Plan).

## References

The connector reads `s3://physionet-open/sleep-edfx/sleep-edfx-1.0.0.zip`. That object is
8,715,317,980 bytes, has S3 ETag `c77c517107c25ae7d9f4d64f4cbc292d-1039`, and was last modified
on 2024-02-04. The bytes fetched for this record hash to sha256 `a6d978fd…3a80a0`. It extracts to
`sleep-edf-database-expanded-1.0.0/`. Independent references, which the connector doesn't read:

- **`SHA256SUMS.txt` in the archive.** It lists a sha256 for each of the 398 other files.
- **An independent EDF reader.** `mne.io.read_raw_edf` and `mne.read_annotations` (mne 1.13.2),
  plus a stdlib and numpy parser written for this pass. The parser reads the 256-byte header, the
  per-signal fields and the int16 data records, and scales them in float64. It shares no code
  with `edfio` or `bases/edf`.
- **The subject tables read a second way.** `SC-subjects.xls` and `ST-subjects.xls`, read with
  `pandas.read_excel`, with the column meanings taken from the sheet headers (`sex (F=1)`,
  `M1/F2`, `Placebo night`, `Temazepam night`).
- **The hypnogram headers.** Each `*-Hypnogram.edf` repeats its PSG's start moment and patient
  field, which gives a check on the PSG↔hypnogram pairing that doesn't rely on filenames.
- **The PhysioNet page** for version 1.0.0 (DOI 10.13026/C2X676): the two studies, the signals
  and their rates, the eight hypnogram codes, the three lost nights, and the license.
- **Sleep physiology.** Delta power in N3, chin EMG atonia in REM, and eye movements in REM, used
  as domain checks on the built signals.

No paper defines a task, prompt or split for this connector. The README says the tasks are the
connector's own decisions, so there is no authors' code to compare against.

## Fields

"PSG" is `*-PSG.edf`, "hypnogram" is `*-Hypnogram.edf`, "SC table" is `SC-subjects.xls` and "ST
table" is `ST-subjects.xls`.

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| Filename `<SC4\|ST7><subject><night><recorder>0` | PSG name | Record id `sleep-edfx-<name>`, `subject_ids`, `night`, the join keys | PhysioNet page. The id names the subject and night, which no input carries (#52) |
| Study directory | `sleep-cassette/`, `sleep-telemetry/` | `study` annotation, which table to join | As published |
| Signal labels | PSG header | Signal names, spec lookup in `specs.py` | Checked against the header for all 197 (#10, #20) |
| Samples per record, record duration | PSG header | Each signal's regular axis | Checked for all 197 (#25). One file, `SC4362F0`, writes 60 s records |
| Physical and digital min and max | PSG header | Gain and offset of each signal | Per file (#18) |
| Physical dimension | PSG header | Dropped. The unit comes from `specs.py` | Blank for `Resp oro-nasal` in 153 files, for `Temp rectal` in 68, and for `Event marker`. `ID+M-E` for the telemetry `Marker`. The README says the unit comes from the header (#58) |
| Transducer, prefilter, reserved fields | PSG header | Dropped | Not read. The prefilter text is the only record of the EMG envelope processing (#30) |
| Data records | PSG | float32 Signals | Compared with two readers (#13, #18) |
| Start date and time | PSG header | `recording_start_local` text. No `start_time` is set | edfio prefers the EDF+ `Startdate` field, which is a day earlier than the legacy date in 7 files (#26) |
| Patient field `X <sex> X <Sex>_<age>yr` | PSG header | `demographics_note` where it disagrees with the table | 24 notes (#36) |
| Recording field | PSG header | Dropped, except as edfio's start date | Holds `Startdate`, plus `Polyman` in one file |
| Annotations (onset, duration, label) | Hypnogram | `sleep_stage` annotations on the four scoring signals, epoch tasks, `sleep_period` task | Compared with `mne.read_annotations` for all 197 (#40) |
| Hypnogram header (start, patient, recording) | Hypnogram | Dropped | Used here only to check the pairing (#33) |
| Technician letter (8th character) | Hypnogram name | Dropped. The pairing matches the first seven characters | 15 distinct letters |
| `subject`, `night`, `age`, `sex (F=1)`, `LightsOff` | SC table | Join key, `age`, `sex`, `lights_off`, and the age, sex and lights-off tasks | Read with pandas for all 153 rows (#36) |
| `Nr`, `Age`, `M1/F2`, placebo and temazepam `night nr` and `lights off` | ST table | Join key, `age`, `sex`, `condition`, `lights_off`, and their tasks | Read with pandas for all 22 rows (#36) |
| `RECORDS`, `RECORDS-v1`, `SHA256SUMS.txt` | Archive | Not read | `SHA256SUMS.txt` would give a per-file pin (#2, #3) |
| Units of the six specs, such as `degC` for `Temp rectal` | Added, `specs.py` | Signal units | An assumption where the header is blank. `degC` contradicts the values (#19) |
| Eight-label, sex and condition vocabularies | Added | Dataset annotations, `target_schema` | The page's code list (#38). Their descriptions print the enum repr (#56) |
| Lights-off offset | Added: table clock time minus header clock time, wrapped forward by a day | `lights_off` span and task | Recomputed for all 197 (#41) |
| Sleep period | Added: first to last epoch scored 1, 2, 3, 4 or R | `sleep_period` task target | Recomputed for all 197 (#41) |
| Session span | Added: the later of the signal end and the scoring end | `Record.time_span` | Recomputed for all 197 (#29) |
| `condition` | Added from the position of the ST table's column pair | `condition` annotation and task | The sheet's group headers (#23) |
| Annotation descriptions, ids, spec names | Added | Text and ids | Descriptions say the release names no zone, which holds |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | ☑️ Done once | The connector reads the 1.0.0 archive, the only release PhysioNet publishes. The page labels it 1.0.0 and its history describes the 197-PSG expansion. The archive holds 153 cassette and 44 telemetry PSGs, the counts the page gives. No paper is followed, so no other release is in play. |
| 2 | Unpinned input | ⚠️ Open | Nothing pins the input. `SLEEP_EDFX_ZIP_URL` names the S3 key, and no `sha256` is passed to `ensure_archive`, whose `sha256` argument applies to HTTP downloads only. The object was rewritten on 2024-02-04, as its LastModified shows. This pass recorded its size, its ETag and the sha256 `a6d978fd…3a80a0` (References). Needs a pin: hash the archive on S3 downloads too, or check the extracted files against `SHA256SUMS.txt`. |
| 3 | Partial, corrupt or changed download | ✅ Done / ⚠️ Open | Done: every PSG and hypnogram is opened through `open_edf`, which raises on a file shorter than its header (`test_open_edf_refuses_a_file_that_lost_whole_records`, `test_open_edf_refuses_a_file_that_lost_part_of_a_record`) and on a file that isn't EDF (`test_open_edf_refuses_a_file_that_is_not_edf`). One-off: all 398 files listed in `SHA256SUMS.txt` match their sha256, and all 197 PSGs are exactly header + records × samples × 2 bytes. Open: no build checks a checksum, so a changed file of the right length passes. `ensure_archive` deletes the zip after extraction and reuses the extraction without rehashing it. |
| 4 | Wrong files from the archive | ✅ Done / ☑️ Done once | Done: each PSG must have exactly one hypnogram beside it (`test_a_recording_with_no_scoring_beside_it_raises`, `test_a_recording_with_two_scorings_raises`), a name outside the study's shape raises (`test_a_name_that_does_not_fit_the_shape_raises`), and both subject tables must exist. Done once: 153 + 44 PSGs and 153 + 44 hypnograms, 0 unpaired hypnograms. The connector reads every PSG and hypnogram and both tables. It skips `RECORDS`, `RECORDS-v1` and `SHA256SUMS.txt`. No pinned inventory fails on an extra or missing recording (plan D). |
| 5 | Source structure misread | ☑️ Done once / ⚠️ Open | Done once: headers against the page, all 197 PSGs. The cassette files hold EEG Fpz-Cz, EEG Pz-Oz and EOG at 100 Hz, and EMG, Resp oro-nasal, Temp rectal and Event marker at 1 Hz. The telemetry files hold EEG, EOG and EMG at 100 Hz and `Marker`, as the page says. All PSGs are plain EDF, and all 197 hypnograms are EDF+C. Open, three disagreements the README doesn't mention: the telemetry `Marker` is at 10 Hz in all 44 headers, against the page's 1 Hz. `Temp rectal` has no unit in 68 headers. The legacy start date and the EDF+ `Startdate` differ by one day in 7 telemetry files (#26). |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | ☑️ Done once / ⚠️ Open | Done once: a full build holds 197 Records, for 100 subjects: 78 cassette and 22 telemetry. The SC table has 153 rows. Subjects 39, 68, 69, 78 and 79 are absent, and 13, 36 and 52 have one night, which are the three lost nights the page names. The ST table has 22 rows, two nights each. Open: no test pins these counts against a build (plan D). |
| 7 | Rows dropped silently | ✅ Done / ☑️ Done once | Done: every PSG becomes a Record or raises (`test_one_record_for_each_recording`, `test_a_recording_with_no_row_raises`), and every scored entry becomes an annotation or raises (`test_a_label_the_release_does_not_write_raises`, `test_an_entry_with_no_span_raises_rather_than_being_dropped`). Done once: read back, the build has a Record for each of the 197 PSGs, and the 28,529 hypnogram entries mne reads, 0 missing. |
| 8 | Wrong filter or threshold | ☑️ Done once | The connector applies no filter: no trimming to the sleep period or to lights off, no dropped label, no merge of stages 3 and 4. The build keeps all 483,419 scored epochs, including 25,047 `Sleep stage ?` and 211 `Movement time`, and all of every recording's signal. The README and card say this. |
| 9 | Duplicate records | ✅ Done / ☑️ Done once | Done: one hypnogram per PSG, and one table row per key (`test_one_pair_of_subject_and_night_twice_raises`, `test_a_recording_with_two_scorings_raises`). Done once: 197 distinct record ids. The data sections of the 197 PSGs, hashed after the header, are 197 distinct values. Three pairs of recordings share a start date and time, and their data differ. |
| 10 | Missing or extra channels | ✅ Done / ☑️ Done once | Done: a header signal that `SPECS` doesn't hold raises in `timeseries.build`, and a missing scoring signal fails the build (`test_a_recording_missing_a_scoring_signal_fails_the_build`). Done once: each record's signal names equal its header labels, 197 of 197: 7 per cassette file and 5 per telemetry file. None is derived. The telemetry study has no Resp or Temp channel, which the page confirms. |
| 11 | Source information left unused without a note | ☑️ Done once / ⚠️ Open | Done once: the Fields table lists every file and header field. Open: the README doesn't say the header's unit is dropped, though it claims to take it from the header (#58). It also doesn't mention the prefilter text, the technician letter, or `SHA256SUMS.txt`. |
| 12 | Information the connector made up | ⚠️ Open | The check found a problem. Most added values cite a reference (Fields table): the code list, the sheet headers, the derivations, which this pass recomputed (#41). The exception is the units in `specs.py`. Where the header is blank, they are an assumption, and the README doesn't name them as one. `degC` for `Temp rectal` contradicts the values (#19). `Resp oro-nasal` and the markers are `dimensionless`, an assumption. |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | ✅ Done / ☑️ Done once | Done: `test_the_series_values_match_the_recording_they_were_read_from` and `test_read_signal_gives_every_sample_of_one_signal` read fixture values through the loader. Done once: 16 seeded recordings (12 cassette, 4 telemetry), every value of every signal, 344,722,440 values, read through the SDK and compared with the numpy reader. Every length matches, and the largest difference is 0.00027 µV, 0.0007 of one LSB. For the 100 Hz channels, mne gives the same result. |
| 14 | Precision lost on a dtype change | ☑️ Done once | The source is int16 counts, and the build stores float32 after scaling in float32. Against float64 scaling over the 344,722,440 values of #13, the largest error is 0.0022 of one LSB, on the EMG and Temp channels. So no stored count is lost. |
| 15 | Overflow or wraparound in integer conversion | ☑️ Done once | EDF counts are int16 and convert to float32 exactly. Ten signal-recordings hold counts outside their declared digital range, and those come through unchanged, because the #13 comparison holds there too. Offsets are int64 microseconds. The longest record is about 24 h, far inside int64. |
| 16 | Write and read don't round-trip | ✅ Done / ☑️ Done once | Done: `test_convert_round_trips_through_the_writer`. Done once: the full build, read back for all 197 Records and compared with the references, matches on signal names, rates, units, spans, `age`, `sex`, `condition`, `night`, `study`, `subject_ids`, `lights_off`, the 24 notes and all 28,529 stage entries. The only mismatches are 7 `recording_start_local` dates, a connector choice (#26). All 484,248 streamed tasks were compared too (#40, #41). The SDK returns a record's signals in name order, not header order. |
| 17 | Build isn't deterministic | ✅ Done / ☑️ Done once | Done: `test_a_second_build_names_the_same_records_and_asks_the_same_questions`. Done once: two full builds from the same extraction. The 131 value parquet files are byte-identical. All 16 `control.duckdb` tables hold the same rows, apart from generated ids: `annotation_contents.content_id`, `annotation_occurrences.occurrence_id`, `axes.axis_id` and `tasks.task_id`. The manifests differ only in that file's checksum and size. |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | ✅ Done / ☑️ Done once | Done: each file's own range sets its gain (`test_convert_digital_to_physical_uses_the_range_of_its_own_file`). Done once: the #13 comparison against float64 scaling and mne. The cassette files hold 153 distinct sets of physical ranges, and 117 distinct EEG Fpz-Cz ranges. The telemetry files share one. |
| 19 | Wrong units or scale | ☑️ Done once / ⚠️ Open | Done once: EEG and EOG are labeled `uV` in every header, centered near 0 (median per-record medians −0.04 µV) and spanning about ±170 µV for the cassette EEG. Sleep physiology holds in µV (#20). Open: `Temp rectal` is labeled `degC`, and its values don't fit a rectal temperature. The median per-record median is 14.26, and 95 of 153 recordings have a median outside 34–40. Most headers state a 0–30 physical range, 68 with no unit. The values may be uncalibrated, or offset by an unknown amount. The cassette EMG is an RMS envelope at 1 Hz (page), but its spec says plain EMG. Needs a decision on the Temp unit and on documenting the envelope. |
| 20 | Swapped or mislabeled channels | ☑️ Done once | Names come from each header (#10). In the #13 sample, no signal matches another channel, or itself shifted by one sample, as well as its own header channel. Physiology over 24 seeded recordings, inside the sleep period: chin EMG is lowest in REM (median log 0.44, against 1.13 in W and 0.85 in N2). EOG activity is higher in REM than in N2 (log std 3.51 against 2.57). Slow-wave power on Fpz-Cz is highest in N3 (7.62, against 6.14 in N2). The two EEG derivations were not told apart. |
| 21 | Flipped sign convention | ☑️ Done once / ⚠️ Open | Done once: the sign is the file's own. Values, sign included, equal mne and the numpy reader (#13). Open: no physical check of EEG or EOG polarity, such as the eye-blink deflection on Fpz-Cz. |
| 22 | Missing-value markers stored as values | ☑️ Done once | Over all 197 PSGs: no signal is constant, and EDF has no NaN marker. Counts at the digital limits are saturation, not markers (#24). The marker channels hold small integer codes: up to ±350 in cassette files, and −33 to 34 in telemetry files. |
| 23 | Wrong categorical codebook | ✅ Done / ☑️ Done once / ⚠️ Open | Done: each sheet decodes sex with its own map, and an unknown code raises (`test_the_code_1_means_the_opposite_thing_in_each_sheet`, `test_a_sex_code_the_sheet_does_not_define_raises`). The condition comes from its column pair (`test_the_condition_comes_from_the_column_the_night_number_sits_in`). Done once: pandas reads the headers `sex (F=1)` and `M1/F2`. The decoded sex agrees with the hypnogram header's sex on all 197 recordings, and with the PSG header's on 194. Placebo and temazepam each cover 22 telemetry nights. Open: the meaning of the marker codes is undocumented. |
| 24 | Implausible upstream values passed through without a note | ⚠️ Open | Found, and not in the README. The `Temp rectal` values (#19). Saturated EEG at the digital limits: 134,729 Fpz-Cz samples in 109 recordings and 88,059 Pz-Oz samples in 71, plus 1,897 EOG samples in 7 recordings and 3,144 EMG samples in 4. Ten signal-recordings hold counts outside their declared digital range. The header ages for `SC4772G0` (100, against 85 in the table) and `SC4642E0` (88, against 85). |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | ☑️ Done once | Every signal's period equals record duration ÷ samples per record in its header, for all 1,291 signals of the 197 records, 0 mismatches. That holds for the one 60 s-record file too. The values have exactly records × samples per record each (#13). The rates match the page, except for the telemetry marker (#5). |
| 26 | Wrong start time or epoch | ⚠️ Open | The check found a problem. No `start_time` is set, and each record's zero is its first sample. The `recording_start_local` text equals the legacy header date and time for 190 of 197. For `ST7021J0`, `ST7071J0`, `ST7092J0`, `ST7131J0`, `ST7132J0`, `ST7141J0` and `ST7142J0`, it is one day earlier. Those files start between 00:00 and 00:52, and their EDF+ `Startdate` names the evening before. edfio prefers that field, and the build only warns. mne uses the legacy date. Which one is true needs a decision, and the README should name the choice. |
| 27 | Wrong time zone | ✅ Done / ☑️ Done once | The source states no zone, and the connector invents none. It sets no `start_time` and keeps the header clock as text with no offset (`test_the_header_clock_is_carried_and_no_start_time_is_set`, `test_a_start_moment_that_names_a_zone_raises`). In the full build, all 197 Records have no start timestamp. |
| 28 | Daylight-saving errors | ☑️ Done once | The timeline comes from sample counts, so DST could only affect the lights-off offset, which is a clock difference. Assuming Europe/Amsterdam and the legacy dates, 0 of the 197 recordings span a DST switch. |
| 29 | Gaps or missing values filled or dropped silently | ✅ Done / ☑️ Done once | Done: a discontinuous EDF+D raises (`test_open_edf_refuses_a_discontinuous_file`), and unscored time gets no task (`test_a_hole_between_two_entries_gets_no_task`, `test_a_scoring_that_starts_late_gets_no_task_for_the_time_before_it`). Done once: all 197 PSGs are plain continuous EDF. In the scoring, `ST7121J0` has a 30 s hole and `ST7221J0` a 1,950 s hole. 26 telemetry scorings start late, by 30–6,780 s. 155 scorings end after their signals, by up to 25,200 s (`SC4592G0`, median 3,720 s), and 38 end before. Each session span is the later of the two ends, 197 of 197. Nothing is filled. |
| 30 | Artifacts from resampling or interpolation | ☑️ Done once / ⚠️ Open | Done once: the connector doesn't resample. Every signal keeps its header rate, and its values equal the source (#13). Open: the cassette EMG was rectified, low-pass filtered and sampled at 1 Hz upstream. The page says so, and the README and spec don't. |
| 31 | Off-by-one sample | ☑️ Done once | Lengths equal the header's for every signal in the #13 sample. A one-sample shift never fits the source as well as the unshifted signal does. |
| 32 | Sources or channels out of sync | ☑️ Done once | All signals of a file share its record timeline. Every hypnogram states the same start date and time as its PSG, 197 of 197, so onsets need no offset. The label-shift test of #42 peaks at shift 0. |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | ✅ Done / ☑️ Done once | Done: the hypnogram pairs on the seven-character prefix, the SC table on (subject, night) from columns, and the ST table on the subject number from its column (`test_telemetry_rows_key_by_the_subject_number_and_not_by_the_position`, `test_cassette_rows_key_by_subject_and_night`). Done once, pairing: each hypnogram header repeats its PSG's start moment, 197 of 197. The patient field matches in 194, and in the other 3 the hypnogram's sex agrees with the table. The recording field matches in 196; the other differs by `Polyman`. |
| 34 | Off-by-one id | ☑️ Done once | The header age, an independent reading, equals the joined table age in 175 of 197, and is one year lower in 20. The other two are header errors (#24). A join one subject over would break that. The header sex agrees in 194 of 197. Stage labels align with the EEG at shift 0 in 21 of 21 records (#42). |
| 35 | Ids or join keys not unique | ✅ Done | A repeated table key raises (`test_one_pair_of_subject_and_night_twice_raises`), and so do two hypnograms for one PSG (`test_a_recording_with_two_scorings_raises`). One-off: 197 distinct record ids, 153 distinct SC keys and 22 distinct ST subject numbers. |
| 36 | Subject metadata on the wrong recording | ☑️ Done once | All 100 subjects have one age and one sex across their nights. Read with pandas and joined independently, `age`, `sex` and `condition` equal the build's for 197 of 197. The 24 notes sit exactly on the 24 recordings whose header disagrees with the table: 20 off by one year, 1 by 3 years, 1 by 15 years, 3 on sex, one of them also by a year. |
| 37 | Join silently drops rows | ✅ Done / ☑️ Done once | Done: a recording with no row raises (`test_a_recording_with_no_row_raises`, `test_a_telemetry_subject_with_no_row_raises`), and so does a night that matches no column (`test_a_night_number_that_matches_neither_column_raises`). Done once: all 153 SC rows and all 22 ST rows (44 nights) join a recording, and all 197 recordings join a row. |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | ✅ Done / ☑️ Done once | Done: the stage vocabulary holds the eight labels (`test_the_stage_set_holds_the_eight_labels_and_no_others`), and a label outside it fails the build (`test_a_label_the_release_does_not_write_raises`). Done once: the labels mne reads from all 197 hypnograms are exactly those eight, and they match the page's W, 1, 2, 3, 4, R, M and ?. The target is the label as written, with no N3 merge, so no mapping can go wrong. The order follows no authors' code, because there is none. |
| 39 | Label distribution differs from the paper | ☑️ Done once / ⚠️ Open | No paper defines the task. Done once: epoch targets count W 290,365, 1 25,175, 2 88,983, 3 12,191, 4 7,263, R 34,184, ? 25,047 and M 211. These equal both the mne expansion and the README. Open: W dominates because nothing is trimmed (#8). Nothing records these counts for a later build (plan D). |
| 40 | Wrong granularity, such as per record instead of per epoch | ✅ Done / ☑️ Done once | Done: one task per 30 s epoch (`test_a_run_expands_into_one_task_for_each_epoch`, `test_an_epoch_task_covers_each_scored_epoch`). An off-grid entry fails the build (`test_an_entry_that_does_not_divide_into_epochs_names_its_duration`). Done once: all 28,529 entries sit on the 30 s grid. The 483,419 epoch tasks equal the mne expansion, onset and label, for 197 of 197 records, with 0 duplicates. Each is exactly 30 s, and none names a series. |
| 41 | Derived target computed wrongly | ☑️ Done once | Recomputed independently and compared with all streamed tasks: 197 sleep-period intervals, first sleep onset to last sleep end, 0 mismatches. 194 lights-off points, table clock minus header clock wrapped by a day, 0 mismatches. 197 ages, 197 sexes and 44 conditions from pandas, 0 mismatches. The three recordings that start 30 s after lights off (`ST7021J0`, `ST7022J0`, `ST7162J0`) wrap to 86,370 s and get no task, as the README says. |
| 42 | Label misaligned with, or contradicted by, its signal | ☑️ Done once | 24 seeded recordings (18 cassette, 6 telemetry), inside each sleep period. The AUC of 0.5–2 Hz Fpz-Cz power for N3 against N2 has a median of 0.966 at shift 0. At ±1 epoch it is 0.915 and 0.899, at ±10 epochs 0.851 and 0.777, and at ±60 epochs 0.527 and 0.479. In 21 of the 21 records that have enough N3, the best shift is 0. The AUC of chin EMG for W against R is 0.938 at shift 0, falling to 0.51–0.60 at ±60. |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | ⚪ N/A | The release publishes no split, and the connector assigns none. All 484,248 tasks read back with no split. |
| 44 | Leakage across splits | ⚪ N/A | No split exists. A consumer who splits must group by `subject_ids`: 97 of the 100 subjects have two recordings, and the README doesn't say so (#59). |
| 45 | Splits regenerated without a fixed seed | ⚪ N/A | Nothing generates a split. |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | ☑️ Done once | No source defines a task. Reviewed: per-epoch classification matches how R&K scoring and the page define the labels. Age is a scalar with a unit, and sex and condition are classifications. Sleep period and lights off are sparse temporal localizations. The README gives the counts by type, and the full build has the same: 483,419, 197, 197, 197, 44 and 194. |
| 47 | Task can't be solved from its inputs | ☑️ Done once / ⚠️ Open | Done once for staging: the signals separate N3 from N2 (AUC 0.966) and W from R (0.938) (#42). Open: no baseline has run for age, sex, condition, sleep period or lights off. Lights off is defined by the table, and the signals may not show it. |
| 48 | A field in the wrong role | ✅ Done / ⚠️ Open | Done: provenance becomes no task (`test_provenance_becomes_no_task`). Tasks name no series and carry no prompt or input annotations (`test_the_scope_covers_the_epoch_and_names_no_series`). The full build has 0 rows in `task_annotation_refs`. Open: each task's only input is its whole Record. That Record, and the record annotations the torch loader returns, carry the targets (#51, #52). |
| 49 | Wrong prompt, or one without a source | ✅ Done | No task has a prompt (`test_no_task_carries_a_prompt`). The full build read back 0 prompts over 484,248 tasks. The README says the connector invents none. |
| 50 | Series placed at the wrong point in the prompt | ☑️ Done once | There is no prompt to place a series in. Every one of the 484,248 tasks has exactly one input, its own Record, 0 exceptions. No review sheet with plots yet (plan G). |
| 51 | An input reveals the target | ⚠️ Open | The check found a leak. No task has input annotations, but the input is the whole Record. The Record holds the target for every task type. The four scoring signals carry `sleep_stage` annotations, which hold every epoch target and the sleep period. The Record's annotations hold `age`, `sex`, `condition` and the `lights_off` point. A consumer that renders a Record's annotations gives the answer away, and the README doesn't warn against it. |
| 52 | The target leaks through an id, metadata or another field | ⚠️ Open | Found, through the record annotations. `TimeFTorchDataset.__getitem__` returns `record.annotations` with every sample, beside the task's inputs. For this build those annotations hold `age`, `sex` and `condition`, the targets of three whole-record tasks, and the `lights_off` point, the target of a fourth. `demographics_note` also restates the table's age and sex in 24 records. Not a leak on its own: the record id and `subject_ids` name the subject and night. No prompt or input carries them, and no split is cut from them. Not measured with a classifier on those fields alone (plan H). |
| 53 | An input carries information from after the target | ⚠️ Open | Each epoch task's input is the whole recording, including later epochs and the scorer's annotations. Context is normal for staging, but the README doesn't state what the input holds. The whole-record tasks ask about facts that hold before the recording. |
| 54 | Answer format doesn't match what the evaluation expects | ⚠️ Open | TimeNet has no evaluation for these tasks. Epoch targets are the eight raw labels, so a 5-class evaluation must merge 3 and 4 and drop M and ?, and the card says so. Needs a dummy evaluation (plan G). |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | ☑️ Done once | The PhysioNet page for 1.0.0 states the Open Data Commons Attribution License v1.0. The card declares `ODC-By-1.0`. |
| 56 | Wrong domain, tags or description | ✅ Done / ⚠️ Open | Done: the card validates (`test_every_connector_ships_a_valid_dataset_card`), and its domain, tags and description were reviewed against the page. Open: "whole-night polysomnograms" undersells the cassette recordings, which run about 20 h. The three vocabulary annotations' descriptions render the enum repr, as in "The values a <AnnotationKey.SLEEP_STAGE: 'sleep_stage'> target takes", because `tasks.py` formats `{key!r}`. |
| 57 | A README number that doesn't match the build | ☑️ Done once / ⚠️ Open | Done once: these README numbers were recomputed and match. 484,248 tasks over 197 records, the six task counts, 28,529 entries and 483,419 epochs, a mean of 16.9 per entry, the longest run of 1,351 epochs, all eight label counts, 26 late telemetry scorings, the 30 s and 1,950 s holes, 155 overruns, 24 notes, 3 sex contradictions, 117 cassette EEG ranges and 1 telemetry range, and 86,370 s. Open: "one of them begins 750 s in". Two begin 750 s in, and the latest begins 6,780 s in (`ST7151J0`). "21 recordings differ by a year": 20 do, and 2 differ by 3 and 15 years. "117 distinct ranges across the 153 cassette files" holds for EEG Fpz-Cz alone. The full range sets are all 153 distinct. No test ties these numbers to the build (plan D). |
| 58 | A README claim the data doesn't support | ⚠️ Open | Found three. The source-of-truth table says a signal's unit comes from its EDF header, and it comes from `specs.py` (#11, #12). The one-year age differences are put down to "the age at the recording against the age at enrollment", which nothing in the release states, and which the 3- and 15-year differences don't fit. The README never says the build keeps edfio's EDF+ start date over the legacy field (#26). The rest held: the pairing, the table winning on age and sex, the per-file gain, the sex decoding, and the lights-off wrap. |
| 59 | Known limitations left undocumented | ⚠️ Open | Missing from the README: the `Temp rectal` values and unit (#19). EEG saturation (#24). The EMG envelope (#30). The 7 start dates (#26). The 10 Hz telemetry marker (#5). The targets held in each input Record's annotations (#51, #52). That two nights share a subject, which matters to any split (#44). |

## Open items

Most important first.

1. **The target is in the input and in what the loader returns (#51, #52, #48).** Every task's
   input is its whole Record, whose scoring signals carry a `sleep_stage` annotation for every
   epoch. The torch loader also returns `record.annotations` with every sample, and those hold
   `age`, `sex`, `condition` and `lights_off`, and `demographics_note` restates age and sex. Fix:
   document that a model must read only the signals, or move the answers out of what a loader
   returns. Decision for maintainers.
2. **`Temp rectal` unit (#19, #12, #24).** The values have a median of 14.26 and lie outside
   34–40 in 95 of 153 recordings, yet the spec says `degC`. 68 headers state no unit. Decision:
   label it uncalibrated, or find the calibration, and document it.
3. **Start date in 7 telemetry files (#26).** edfio's EDF+ `Startdate` is one day earlier than
   the legacy header date for recordings that start just after midnight. The build keeps edfio's
   and only warns. Decide which date is true, read it explicitly rather than through edfio's
   preference, and document it.
4. **Unpinned input (#2, #3).** Pin the archive's sha256 on S3 downloads too, or check the
   extracted files against `SHA256SUMS.txt` on every build, including a reused extraction.
5. **README corrections (#57, #58, #59, #11, #30, #5).** The 750 s claim, the "21 by a year"
   count and its explanation, the 117 count, the unit-from-header claim, the start-date choice,
   the EMG envelope, the 10 Hz marker, saturation, and the two nights per subject.
6. **Vocabulary descriptions (#56).** `tasks.build_vocabularies` formats the key with `!r` and
   stores the enum repr. Use `{key}` or `key.value`.
7. **Unchecked items.** EEG polarity (#21), baselines for the whole-record tasks (#47), marker
   codes (#23), an evaluation format (#54).
8. **Repeatable checks on the real data.** Everything marked ☑️ Done once ran from scratch
   scripts. Commit them so they run again (Plan).

## Plan

Checks on the real data run in a `real_data` pytest marker, skipped unless an environment variable
points at an extracted release and a built version. Every check below except D, F, G and H ran
once for this record.

- **A1. Reference reader.** For a seeded sample of recordings, compare every value, rate, length,
  name and unit with mne and the numpy reader (#13, #18, #25, #31).
- **A2. Hypnograms.** Compare every stage annotation and epoch task with `mne.read_annotations`.
  Check the PSG and hypnogram headers agree on the start moment and patient field (#33, #40).
- **A3. Subject tables.** Read both sheets with pandas, join them independently, and compare age,
  sex, condition, lights-off offset and the notes (#36, #41).
- **A4. Archive pin.** Check every extracted file against `SHA256SUMS.txt`, and the archive
  against a committed sha256 (#2, #3).
- **B1. Sleep physiology.** Delta power by stage, EMG atonia in REM, and the label-shift AUC, which
  must peak at shift 0 (#20, #42). Add a blink-polarity check on Fpz-Cz for #21.
- **C1. Start dates.** Compare the legacy and EDF+ start dates for every file, and fail when the
  build's choice differs from the documented one (#26).
- **D. Stats snapshot.** Commit a `verification.json` with the record, subject and task counts,
  label counts, per-signal percentiles, saturation counts, Temp ranges and the README's numbers,
  and a test that compares a build against it (#6, #24, #39, #57).
- **E. Determinism.** Build twice, and compare the value files byte for byte and the control
  tables without generated ids (#17).
- **F. Corruption tests.** Shift the hypnogram by one epoch, swap EEG Fpz-Cz and EOG, and join
  one subject over. Each must fail A1, A2, A3 or B1.
- **G. Review sheet and evaluation.** Render seeded tasks of each type with the signals plotted
  over the mne reading, and run a dummy 5-class staging evaluation (#50, #54).
- **H. Leakage outside the signals.** Train on the record annotations a loader returns with
  each sample, without the signals. After item 1 is fixed, they must score at chance on every
  task (#52). Add baselines for the whole-record tasks (#47).
