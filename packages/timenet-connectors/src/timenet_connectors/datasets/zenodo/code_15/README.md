# CODE-15%

`zenodo/code-15` holds CODE-15% ([Zenodo record 4916206][zenodo], version 1.0.0): 345,779
12-lead ECGs from 233,770 patients, recorded by the Telehealth Network of Minas Gerais, Brazil,
from 2010 to 2016. Each exam comes with the patient's age and sex, six abnormality labels, and,
on at most one exam per patient, mortality follow-up.

The data license is CC BY 4.0. The authors use it in two papers, with code in three repositories:

| Source | What it gives the connector |
| --- | --- |
| [Lima et al. 2021][lima], [`ecg-age-prediction`][age] | The age task, its split, `nn_predicted_age`, the mortality analysis |
| [Ribeiro et al. 2020][ribeiro], [`automatic-ecg-diagnosis`][diag] | The six abnormalities, their order, their split |
| [`ecg-preprocessing`][prep] | How the HDF5 files were made: the scale, the leads, the padding |

The connector pins each file by the SHA-256 it computed on download; Zenodo publishes md5s,
which `release.py` keeps beside them.

[zenodo]: https://zenodo.org/records/4916206
[lima]: https://doi.org/10.1038/s41467-021-25351-7
[ribeiro]: https://doi.org/10.1038/s41467-020-15432-4
[age]: https://github.com/antonior92/ecg-age-prediction
[diag]: https://github.com/antonior92/automatic-ecg-diagnosis
[prep]: https://github.com/antonior92/ecg-preprocessing

## The source data

| File | Content |
| --- | --- |
| `exams.csv` | One row per exam, 15 columns (below) |
| `exams_part0.zip` … `exams_part17.zip` | Each holds one `exams_part<n>.hdf5`: 20,001 rows, 5,780 in part 17 |

Each HDF5 file holds `exam_id`, `(N,)` int64, and `tracings`, `(N, 4096, 12)` float32, in the same
order. That order isn't `exams.csv`'s.

`exams.csv`'s columns:

| Column | Meaning (Zenodo) |
| --- | --- |
| `exam_id`, `patient_id` | The exam and the patient |
| `age` | The patient's age in years at the exam: 17 to 100 |
| `is_male` | `True` if the patient is male |
| `nn_predicted_age` | The age the network of Lima et al. predicts from the trace |
| `1dAVb`, `RBBB`, `LBBB`, `SB`, `ST`, `AF` | Whether the exam shows each abnormality |
| `normal_ecg` | `True` if the automatic annotation system calls the ECG normal |
| `death` | Whether the patient died in the follow-up; on at most one exam per patient (none for 123 patients), else empty |
| `timey` | Years to death, or to the end of follow-up for a survivor; with `death` |
| `trace_file` | The HDF5 file that holds the exam's trace |

- **Signal.** 400 Hz, leads DI, DII, DIII, AVR, AVL, AVF, V1 to V6. The authors made the files
  with `generate_h5.py --scale 2 --use_all_leads --new_freq 400 --new_len 4096` of
  `ecg-preprocessing` (its README, "CODE-15%"): each recording resampled to 400 Hz, multiplied
  by 2, and padded with zeros to 4096 samples, centred, or cut around its centre if longer. The
  baseline wasn't removed.
- **Scale.** A published value is twice the millivolt value; the author confirms the factor in
  `ecg-age-prediction` issue 5. Both READMEs say the values are "at the scale 1e-4V", which
  contradicts itself and the files (issues 10 and 34 of `automatic-ecg-diagnosis`). Measured on
  the four CODE-15% exams whose raw WFDB files `ecg-preprocessing` publishes, the factor is
  1.953125 (= 2 × 1000/1024), not 2: either those files' gains are off by 1.024, or a Record's
  values are about 2.3% below the true millivolts. The connector keeps the author's 2 until
  the author says which (VERIFICATION.md #19).
- **Leads.** `--use_all_leads` computes DIII, AVR, AVL and AVF from DI and DII. The traces only
  partly agree. Of the 345,106 traces whose DI and DII aren't both zero, DIII equals DII − DI to
  within 10⁻⁶ of the peak in 250,910, and AVR, AVL and AVF are within 2% of their derivation in
  258,355 (median 0.6%), so close but not exact. In 62,658 at least one of the four is more than
  5% off. The lead order is the one the derivation fits.
- **Labels.** From the cardiologist's report, text-mined and checked against the Glasgow program's
  statements and Minnesota codes, with disagreements reviewed by hand (Ribeiro et al., Methods).
  `normal_ecg` combines the report with the Glasgow program's measurements (Lima et al.).
- **Mortality.** From probabilistic linkage to Brazil's national mortality register (Lima et al.).
  233,647 exams state it, one per patient; 123 patients have no exam that does. 8,341 died.

## The TimeNet dataset

### Records

One Record per exam whose trace isn't zero everywhere: 345,109 of the 345,779, from 233,480
patients. Its one Source, `12-lead ECG`,
holds 12 lazy float32 signals named as the release names the leads, in millivolts, on a 400 Hz
axis:

- **Values.** The published values halved. Halving a float32 only lowers its exponent, so it is
  exact: every value times 2 is the published value, bit for bit.
- **Padding.** The zeros the authors added are left out. `preprocess_ecg` puts
  `(4096 - n) // 2` zeros before a trace of `n` samples and the rest after, so the zeros after are
  the zeros before or one more. Of the samples that are zero in every lead at each end, only those
  this rule explains are padding: the shorter run, and one more after if there is one
  (`preprocessing.centred_padding`). Any other zeros are zeros of the published trace, and stay.
- **Length.** 201,722 traces keep 2,934 samples, 7.335 s, after 581 zeros at each end;
  143,051 keep all 4,096; 159 keep 4,000; and 177 have one of 50 other lengths, from 174 to
  4,095.
- **Time.** The release gives no exam date, so `start_time` is unset and each axis starts at the
  first sample kept.

The patient id is the shared `subject_id` of `Record.metadata`, beside `exam_id`, `trace_file`,
`n_pad_before` and `n_pad_after`; `records.record_metadata(record)` reads them back. `sex` is a Record annotation, `male` or `female`: `is_male` decoded into the shared
`timenet_connectors.demographics.Sex`. On the exam that states them, `death` and `follow_up`
(`timey`, in years) are Record annotations too.

The age and the labels are the tasks' targets, so they stay off the Record.

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_url`, `source_doi` | The Zenodo record |
| `abnormality_vocabulary` | `1dAVb`, `RBBB`, `LBBB`, `SB`, `AF`, `ST`: the abnormality tasks' `target_schema` |

### Tasks

Each exam has the two tasks the authors define on it, with neither prompt nor scope:

| | Age | Abnormalities |
| --- | --- | --- |
| Paper | Lima et al. 2021 | Ribeiro et al. 2020 |
| Type | `ScalarPredictionTask` | `ClassificationTask`, multi-label |
| `targets` | `age` as a float, `unit` `year`, `target_name` `age` | Each abnormality the exam has, in the authors' output order; empty if none |
| Task annotation | `nn_predicted_age`, in years | `normal_ecg` |
| `split` | `test` | `train` |
| Loader method | `age_tasks` | `abnormality_tasks` |

The task stream holds every exam's age task, then every exam's abnormality task. After
`populate`, `TimeFDataset.tasks_of` selects one task type and `TimeFDataset.get_split` one split.

The abnormality order is the authors' model output order, `1dAVb, RBBB, LBBB, SB, AF, ST`;
`exams.csv` writes ST before AF. Over the 345,109 abnormality tasks:

| 1dAVb | RBBB | LBBB | SB | AF | ST | None | Two or more |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5,716 | 9,672 | 6,026 | 5,601 | 7,031 | 7,584 | 307,340 | 3,671 |

Task annotations are never model input: `nn_predicted_age` is the authors' prediction, kept to
compute the gap between ECG-age and age, and `normal_ecg` selects Lima et al.'s normal-ECG
subgroup.

**Split.** The release defines none, so each task takes the role its paper gives CODE-15%:

- Lima et al. hold it out: "The CODE cohort was divided into 85-15% splits, with the 85% split
  being used for developing the model and 15% hold-out being the one used in subsequent analyses
  and referred to as CODE-15%." The age tasks are `test`.
- Ribeiro et al. train on 98% of the CODE cohort, and the `automatic-ecg-diagnosis` README says
  "Part of the training data (the CODE-15% dataset) is openly available". Which of its exams fell
  in their 2% validation set isn't published. The abnormality tasks are `train`; their test set is
  a separate 827-exam release.

Both tasks of an exam take the same Record, so the splits share every input: a model trained on
the `train` abnormality tasks has seen the ECGs of the `test` age tasks, though not their ages. To
test an age model as Lima et al. do, train it on other data. Within either task, a patient's exams
are all in one split.

The code splits nothing that can be reproduced: `ecg-age-prediction/train.py` validates on the
first `n_valid` rows of the HDF5 file (default 100; `<=` takes 101), and
`automatic-ecg-diagnosis/datasets.py` on the last 2%, both in file order.

**Mortality.** No task: neither paper trains a model to predict death. Lima et al. fit a Cox
model, `duration=timey`, `event=death`, adjusted for age and sex, on groups of the ECG-age gap
(more than 8 years above the age, within 8, more than 8 below). Every input is in the dataset: the
age task's target and `nn_predicted_age`, and the Record's `sex`, `death` and `follow_up`. A
survivor's `follow_up` is censored, so "died" and "survived" don't compare across patients
followed for different times.

## Rebuilding the authors' input

`preprocessing.authors_input(millivolts, padding)` rebuilds the published `(4096, 12)` trace bit
for bit from a Record's 12 signals, stacked in `vocabulary.Lead` order, and the padding in its
metadata:

```python
import numpy as np

from timenet_connectors.datasets.zenodo.code_15.preprocessing import (
    authors_input,
)
from timenet_connectors.datasets.zenodo.code_15.records import (
    record_metadata,
)
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Lead

code15 = record_metadata(record)
by_name = {signal.name: signal for signal in record.sources[0].signals}
millivolts = np.stack(
    [by_name[lead.value].to_numpy() for lead in Lead], axis=1
)
trace = authors_input(millivolts, code15.padding)
```

The authors' networks read it transposed, `(12, 4096)`, with no other preprocessing.

## How the connector builds it

```text
download  exams.csv ─────── sha256 ──────────────────────► code15/exams.csv
          exams_part<n>.zip ─ ensure_archive, sha256 ─────► code15/exams_part<n>.hdf5

parse     exams.csv ── read_exams ──► ExamRow per exam (checked across rows)

for each  exams_part<n>.hdf5 ── scan_part, once ──► TraceScan per row
part        filler row checked; exam_id, location,
            leading and trailing all-lead zeros

join      index_exams: ExamRow + TraceScan by exam_id ──► ExamIndex
            exams: Exam(row, scan), padding = scan.padding (centred_padding)
            all_zero_exam_ids: set aside, checked against the pin

          build_record(Exam, LeadLoader) ──► Record: 12 lazy leads

populate  dataset annotations, then the Records, then the task stream:
          age_tasks ─────────► ScalarPredictionTask, every exam
          abnormality_tasks ─► ClassificationTask, every exam

write     LeadLoader(exam, lead)
            └─ rereads the exam's row once for its 12 leads, checks its
               zeros against the scan, halves it, leaves the padding out
```

| Module | Holds |
| --- | --- |
| `connector.py` | The download, the provenance annotations, and `convert` |
| `release.py` | What is pinned: the record, its files and digests, the HDF5 layout, the counts |
| `preprocessing.py` | What the authors' preprocessing fixes: the scale, the length, the padding rule |
| `vocabulary.py` | The closed sets: leads and abnormalities |
| `keys.py` | Every annotation key the connector writes |
| `exams.py` | `exams.csv`: `ExamRow` and `read_exams` |
| `tracings.py` | The HDF5 parts: `scan_part`, `TraceScan`, `TraceReader` |
| `index.py` | The join: `Exam`, `ExamIndex`, `index_exams` |
| `records.py` | An exam's Record and the lazy `LeadLoader` |
| `tasks.py` | `age_task` and `abnormality_task`, which build one exam's TimeNet tasks, and their splits |
| `loader.py` | `Code15DatasetLoader`, which runs the parse and streams the tasks: `age_tasks`, then `abnormality_tasks` |

Parsing reads every trace once, about 68 GB, to find its padding: a Signal's length must be known
before its values are read. Writing reads each trace again, once for its 12 leads.

## Known issues upstream

The data keeps each of these as published, and the build checks each.

- **Filler rows.** Every HDF5 file ends with a row whose `exam_id` is 0 and whose trace is zero
  everywhere; `exams.csv` doesn't list it. The build checks it is there and skips it.
- **Traces that are zero everywhere.** 670 exams of `exams.csv` have a trace that is zero in
  every lead and sample. They have no Record; `all_zero_exam_ids.txt` pins their ids, and every
  build checks them. 290 patients have no other exam, so no Record; 419 patients' mortality is
  stated only on such an exam, so it isn't in the dataset. Moving it to another exam would break
  `timey`, which counts from the exam that states it.
- **Row order.** The HDF5 rows aren't in `exams.csv`'s order. The author says to join by
  `exam_id` (`automatic-ecg-diagnosis` issue 32), as the build does. With its defaults,
  `ecg-age-prediction/train.py` pairs the CSV's ages with the HDF5 rows by position, and with
  `--ids_dset` it reindexes the table but not the ages it read before, so neither pairs a trace
  with its exam's age.
- **Zeros that aren't padding.** Runs of all-lead zeros the centring rule can't explain stay as
  published. 12,288 traces start with 2 to 4,048 such samples and end with none; 778 have 3 to
  3,615 more at one end than the rule allows; 23 end with 31 to 4,001, of which the rule takes
  one. 1,103 traces have a lead that is zero throughout, 3 of them both DI and DII.
- **Predicted ages.** `nn_predicted_age` ranges from −40.2 to 106.9 years; it is the network's
  output, kept as published.
- **The paper and the release disagree.** Lima et al. sample CODE-15% from ages 16 to 85, ~3100
  exams per age, and analyse 218,169 patients' first exams. The release has ages 17 to 100 and
  233,770 patients.
- **Lead order.** The Zenodo record, both READMEs and `preprocess.py` give DI, DII, DIII, AVR, AVL,
  AVF, V1 to V6. `automatic-ecg-diagnosis/data/README.md`, about the separate 827-exam test set,
  and its issue 3 give AVL, AVF, AVR. The traces fit the first.
- **Lengths.** Zenodo says the recordings last 7 or 10 s, so 2,800 or 4,000 samples, padded by
  648 or 48 at each end. The traces don't match: 201,722 have 581 zeros at each end around
  2,934 samples, only 159 have 4,000 samples, and none 2,800. 143,051 fill all 4,096 samples,
  cut from a longer recording or never padded.
