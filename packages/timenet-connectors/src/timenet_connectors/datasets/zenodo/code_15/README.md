# zenodo/code-15

CODE-15% ([Zenodo record 4916206][zenodo], version 1.0.0, license CC BY 4.0) holds 12-lead ECGs
of the Telehealth Network of Minas Gerais, Brazil, recorded from 2010 to 2016. The record lists
345,779 exams from 233,770 patients. Each exam has the patient's age and sex, six abnormality
labels, and, on at most one exam per patient, mortality follow-up.

The authors use the data in two papers and three code repositories:

| Source | What the connector takes from it |
| --- | --- |
| [Lima et al. 2021][lima], [`ecg-age-prediction`][age] | The age task, its split, `nn_predicted_age`, the mortality fields |
| [Ribeiro et al. 2020][ribeiro], [`automatic-ecg-diagnosis`][diag] | The six abnormalities, their order, their split |
| [`ecg-preprocessing`][prep] | How the HDF5 files were made: the scale, the leads, the padding |

[zenodo]: https://zenodo.org/records/4916206
[lima]: https://doi.org/10.1038/s41467-021-25351-7
[ribeiro]: https://doi.org/10.1038/s41467-020-15432-4
[age]: https://github.com/antonior92/ecg-age-prediction
[diag]: https://github.com/antonior92/automatic-ecg-diagnosis
[prep]: https://github.com/antonior92/ecg-preprocessing

## The source files

- `exams.csv` has one row per exam: `exam_id`, `patient_id`, `age`, `is_male`,
  `nn_predicted_age` (the age that the network of Lima et al. predicts), the six labels `1dAVb`,
  `RBBB`, `LBBB`, `SB`, `ST` and `AF`, `normal_ecg`, `death`, `timey` and `trace_file`.
  `death` and `timey` (years to death, or to the end of follow-up for a survivor) are empty
  together.
- `exams_part0.zip` to `exams_part17.zip` each hold one HDF5 file. Its `tracings` dataset has the
  shape `(N, 4096, 12)` in float32, and its `exam_id` dataset has the exam of each row. The rows
  are not in the order of `exams.csv`. The last row of each file is an all-zero filler with
  exam id 0.

The authors made the HDF5 files with `generate_h5.py --scale 2 --use_all_leads --new_freq 400
--new_len 4096` of `ecg-preprocessing`. Each recording is resampled to 400 Hz, multiplied by 2,
and padded with zeros to 4,096 samples (centred), or cut around its centre if it is longer.

The connector checks the SHA-256 of each file when it downloads it. Zenodo publishes only md5
sums, so we computed the SHA-256 sums on a first download.

## The TimeNet dataset

### Records

Each exam whose trace is not zero everywhere becomes one Record. A trace that is zero everywhere
holds no signal, so that exam has no Record and no tasks. If it was the only exam of a patient
that states `death` and `timey`, that mortality is not in the dataset. The connector does not move
it to another exam, because `timey` counts from the exam that states it.

A Record has one Source, `12-lead ECG`, with 12 lazy float32 signals on a 400 Hz axis. The names
and the order are those of the release: `DI`, `DII`, `DIII`, `AVR`, `AVL`, `AVF` and `V1` to
`V6`. The values stay in the HDF5 file until they are written.

- **Unit.** A published value is twice the millivolt value (the author confirms the factor in
  `ecg-age-prediction` issue 5), so the connector halves it. Halving a float32 is exact. The
  authors' READMEs say "scale 1e-4V", which contradicts the files. The connector uses the
  factor of 2 that the authors state. Our own fit on the four raw exams that
  `ecg-preprocessing` publishes gave 1.953125 (2 × 1000/1024). With that factor, the values
  would be 2.3% low. We did not use our fit, because four exams are too few to overrule the
  authors.
- **Padding.** The connector leaves out the zeros the authors added. The authors put
  `(4096 - n) // 2` zeros before a trace of `n` samples and the rest after. So the connector
  takes as padding the zeros that are in every lead at both ends and that this rule explains:
  the shorter of the two runs before, and at most one more after. Other zeros stay as published.
  The raw lengths are not published, so this is an inference. A recording that itself starts
  and ends with zeros in every lead loses them.
- **Time.** The release has no exam date, so `start_time` is unset and the axis starts at the
  first kept sample.

`Record.metadata` has `exam_id`, `trace_file`, `n_pad_before` and `n_pad_after`. To rebuild a
published trace, put zeros back around the 12 signals (stacked in the order above) and multiply
by 2. The tests do this for every exam of the fixture.

Record annotations:

| Key | Value |
| --- | --- |
| `subject_ids` | The patient id, as a list with one text |
| `sex` | `male` or `female`, from `is_male` |
| `death`, `follow_up` | `death` and `timey` (in years), only on the exam that states them |

The age and the labels are the targets of the tasks, so the Record does not hold them.

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_url`, `source_doi` | The Zenodo record |

### Tasks

Each exam has the two tasks that the authors define. Neither has a prompt or a scope. The stream
holds every exam's age task, then every exam's abnormality task.

| | Age | Abnormalities |
| --- | --- | --- |
| Paper | Lima et al. 2021 | Ribeiro et al. 2020 |
| Type | `ScalarPredictionTask` | `ClassificationTask`, multi-label |
| Target | `age` as a float, unit `year`, name `age` | Each abnormality of the exam, in the authors' output order. Empty if none |
| Task annotation | `nn_predicted_age`, in years | `normal_ecg` |
| Split | `test` | `train` |

The abnormality order is the output order of the authors' model: `1dAVb, RBBB, LBBB, SB, AF, ST`.
`exams.csv` writes ST before AF.

All abnormality tasks share one label set, `abnormality`. A label set is the closed set of labels
that a classification target takes. It lists the six abnormalities in the order above and gives the
full name of each, such as `AF`: "Atrial fibrillation.". The set is multi-label, so a target
holds every abnormality that applies to the exam. An empty target means that none of the six applies.
TimeNet makes sure that each target fits the set when it writes the dataset.

Task annotations are never model input. `nn_predicted_age` lets a user compute the gap between the
ECG age and the age, and `normal_ecg` selects the normal-ECG group of Lima et al.

**Split.** The release has no split, so each task takes the role that its paper gives CODE-15%.
Lima et al. hold it out to test their age model ("15% hold-out ... referred to as CODE-15%"), so
the age tasks are `test`. Ribeiro et al. train on the CODE cohort, and the `automatic-ecg-diagnosis`
README says "Part of the training data (the CODE-15% dataset) is openly available". So the
abnormality tasks are `train`. The authors do not say which exams were in their validation set.
Both tasks of an exam use the same Record, so a model that trains on the abnormality tasks has
seen the ECGs of the age test set (not their ages). To test an age model as Lima et al. do, train
it on other data.

There is no mortality task. Neither paper trains a model to predict death. Lima et al. fit a Cox
model with `timey` and `death`. Every input of it is in the dataset. A survivor's `follow_up` is
censored, so "died" and "survived" do not compare across patients with different follow-up.

## How the connector builds it

1. **Download.** It fetches `exams.csv` and the 18 zips, checks their SHA-256 sums, and extracts
   each zip once.
2. **First pass.** It reads `exams.csv` and checks each row and the rules across rows: no exam
   twice, one sex per patient, mortality on at most one exam per patient. Then it reads each
   HDF5 file named in `exams.csv` once, in blocks. It checks the filler row, the datasets, the
   types and that every value is finite. It joins each trace to its row by `exam_id`, as the
   author says in `automatic-ecg-diagnosis` issue 32. It keeps the padding and a digest of each
   trace, not the samples. A signal needs its length before its values are read, so this pass
   reads all the traces.
3. **Second pass.** It builds the Records and streams the tasks. When the writer reads a trace,
   the connector rereads it once for its 12 leads and checks its digest. If the file changed
   since the first pass, the build fails with the file and row.

A malformed file raises `TimeFFormatError` with the file and the line or row. The HDF5 files must
stay in place until the dataset is written.

## Known issues in the source

The connector keeps the data as published and does not repair it.

- **Row order.** The HDF5 rows do not follow `exams.csv`. With its defaults,
  `ecg-age-prediction/train.py` pairs the ages with the HDF5 rows by position, so it may pair a
  trace with the age of another exam. The connector joins by `exam_id`.
- **Lead order.** The Zenodo record and the authors' code give DI, DII, DIII, AVR, AVL, AVF, V1 to
  V6. The README for the separate 827-exam test set gives AVL, AVF, AVR. The connector uses the
  first.
- **Predicted ages.** `nn_predicted_age` is the output of the authors' network, kept as published.
  Some values are negative.

## What is not verified

The tests use a small fixture of two parts with five exams, one of them zero everywhere. The
connector has not been run on the real release, so these are not checked on real data: the
SHA-256 sums (copied from an earlier download), the counts in `dataset.yaml` (345,109 exams and
233,480 patients after the exclusion, from an earlier scan), the padding layouts of the real
traces, the memory use, and the build time. The connector does not check the counts of the
release. It leaves out every trace that is zero everywhere, so an HDF5 file with unwritten chunks
(which read as zeros) would lose those exams without an error.
