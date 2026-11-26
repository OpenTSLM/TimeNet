# CHB-MIT: the decisions of this connector, and the inconsistencies of the release

The CHB-MIT Scalp EEG Database (PhysioNet `chbmit` 1.0.0) holds days of scalp EEG from children
with intractable seizures, recorded at Children's Hospital Boston. This document states what the
connector builds, and each place where the release disagrees with itself or with its own page.

*(measured)* marks a number that comes from the release files, not from the dataset page.

## What a build holds

| | count |
| --- | --- |
| cases | 23 (chb01 to chb23) |
| subjects | 22 (chb21 is chb01) |
| records, one per EDF file | 664 |
| marked seizures | 182 |
| `TemporalLocalizationTask`, one per record | 664 |
| `ClassificationTask`, one per whole 2 s window | 1 730 932, of which 5644 are `seizure` |

All counts are *(measured)*.

## Decisions

### One record is one EDF file

A case is a sequence of continuous EDF files with gaps between them. The gaps run from 2 s to
four days *(measured: chb10)*. TimeNet has no axis yet that holds one signal with gaps in it
(TNET-167). A record for each case would need one source for each file. Its exhaustive seizure
task would then state that no seizure happened during time that nobody recorded.

Each record carries `case_offset`: the seconds from the start of the earliest file of its case to
the start of the record. It comes from the EDF header clocks. A consumer can place the files of a
case on one timeline and see each gap. The record metadata holds `case_id` and `file_name`.

### The dates are not real

The release replaced every date with a surrogate and kept the time between the files of a case.
The connector sets no `Record.start_time`. It reads the header clock only to measure
`case_offset`, which crosses midnight correctly because the header states a date. The summary
clock does not wrap: it counts hours past 24, up to 27.

### Channels come from each file

The montage changes between files in chb04, chb09, chb11, chb12, chb13 and chb15 to chb19
*(measured)*. The connector reads the labels of each file from its own header, checks them
against the channel list that the summary states for that file, and maps each label to a spec.
An unknown label stops the build.

- Bipolar (`FP1-F7`) and referential (`FC1-Ref`, `F7-CS2`, `CZ`) leads are EEG.
- `ECG` and `EKG1-EKG2` are ECG, `LOC-ROC` is EOG and `VNS` is the vagus nerve stimulator
  channel. The release documents neither `EKG1-CHIN` nor `LUE-RAE`, so both get a generic
  biopotential spec.
- chb12 writes `01` where the 10-20 system writes `O1`. The signal keeps the label as written.
- The placeholders `-` and `.` become no signal. All 1791 of them hold only zeros *(measured over
  every file of chb01 to chb23)*.

### `T8-P8` is written twice, and both become signals

Every file lists `T8-P8` twice. The first is in the right temporal chain of the double banana
(FP2-F8, F8-T8, T8-P8, P8-O2). The second closes the extra chain through the low temporal
electrodes (P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8).

In 583 of the 633 files the second is a byte-identical copy *(measured)*. In chb12_32 to chb12_42 and in
the 39 chb15 files it differs *(measured)*. The gain differs (0.88 in chb12, 1.07 in chb15), and
so does the signal above about 13 Hz (chb12) or 60 Hz (chb15). chb12 is also one sample late.
Below those frequencies the two agree. This looks like a separate amplifier input that recorded
the same electrode pair. The release does not say.

The connector keeps both. The first keeps the name `T8-P8`, so selecting `T8-P8` works the same in
every file. The second is `T8-P8 (2)` in every montage. Its header position (23, 28 or 29) varies,
so the name does not use it.

### Tasks

- **Where are the seizures?** One exhaustive `TemporalLocalizationTask` for each record. The
  release marks every seizure, so unmarked time holds none. A file with no seizure gets an empty
  target, which states that answer.
- **Does this window hold a seizure?** One `ClassificationTask` for each whole 2 s window from the
  file start, with the labels `seizure` and `no seizure`. A window is `seizure` when it overlaps a
  marked seizure by any amount. A last stretch shorter than 2 s gets no task (36 files have an odd
  length in seconds). The 2 s epoch is the one that Shoeb and Guttag (ICML 2010) use. Their paper
  ships with the release. It is the authors' choice, not part of the annotations. Their classifier
  stacks three of these epochs. A consumer can do the same with adjacent windows.
- The seizure intervals are also whole-record `seizure` annotations on each record.
- No task has a prompt. The release states no question in words.

`timenet_connectors.bases.seizures` builds both tasks from the parsed intervals. A second seizure
database needs only its own parser.

### Splits

The release defines no split, and the paper evaluates within each patient. The connector assigns
whole subjects, so no child is in two splits, and chb21 shares the split of chb01. The split was
chosen once, by hand: validation (chb03, chb10, chb14, chb23) and test (chb05, chb13, chb20, chb22)
each hold four subjects and about 15 % of the seizures (29 and 28). chb12, with 40 seizures, stays
in train.

## Inconsistencies of the release

### chb24 is left out

The page states 23 cases and 182 seizures. The release holds 24 cases and 198 seizures
*(measured)*, because chb24 came later. The connector leaves chb24 out:

- `SUBJECT-INFO` has no row for it, so it may be one of the other 22 subjects. Placed in another
  split than that subject, it would leak one child into two splits.
- Its summary states no file times and leaves out 10 of its 22 files.
- The one disagreement between a `.seizures` file and a summary is in chb24: `chb24_21` starts its
  seizure at 2804 s in the summary and at 2404 s in the `.seizures` file *(measured)*.

### `RECORDS-WITH-SEIZURES` lists the wrong file once

It lists `chb07_18`, and the summary and the `.seizures` file both place that seizure in
`chb07_19` *(measured)*. The seizure starts at 13688 s, and chb07_18 lasts 12060 s. The connector
reads the summary and does not read `RECORDS-WITH-SEIZURES`. For chb01 to chb23, the 129 `.seizures`
files agree with the summary to the second *(measured)*.

### `chb17b_69.edf` holds 256 bytes after its last record

The file holds all 3600 records its header states, then 256 zero bytes *(measured)*. The shared
EDF reader reads the stated records, leaves those bytes out, and logs a warning. A file that has
fewer records than its header states still stops the build.

### File order is not time order once

`chb03_25` starts before `chb03_24` *(measured)*. Records follow the file names, and `case_offset`
follows the clock.
