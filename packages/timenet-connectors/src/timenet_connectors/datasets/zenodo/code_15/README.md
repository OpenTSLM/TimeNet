# CODE-15%

`zenodo/code-15` converts the 12-lead ECG release from [Zenodo record 4916206][zenodo], version
1.0.0, under CC BY 4.0. The record lists 345,779 exams from 233,770 patients in Minas Gerais,
Brazil, recorded from 2010 to 2016.

`exams.csv` contains demographics, six diagnosis flags, `normal_ecg`, predicted age and optional
mortality follow-up. The 18 archives contain float32 HDF5 tracings shaped `(N, 4096, 12)` and exam
IDs. Downloads use SHA-256 hashes retained from an earlier download; Zenodo publishes MD5 hashes.

## Records and signal conversion

Each exam with a nonzero trace becomes one Record. Traces join metadata by `exam_id`, not row
position: the CSV and HDF5 orders differ. Each Record has twelve lazy float32 signals on a 400 Hz
axis, named `DI`, `DII`, `DIII`, `AVR`, `AVL`, `AVF`, and `V1` through `V6`.

Two transformations follow the authors' [preprocessing code][prep]:

- **Scale:** divide published values by two to obtain millivolts, following the author's stated
  scale factor. This conflicts with the “scale 1e-4V” wording in their READMEs. An earlier fit to
  four raw examples suggested 1.953125 instead, a roughly 2.3% difference; that estimate was not
  used to override the stated factor.
- **Padding:** remove all-lead zero runs at the ends only to the extent explained by centred
  padding: equal runs, or one additional trailing zero. Original lengths are unavailable, so this
  is an inference and can remove genuine zeros at both ends. Other zeros remain.

There are no exam timestamps. The axis begins at the first retained sample. Record metadata keeps
`exam_id`, `trace_file`, `n_pad_before` and `n_pad_after`, allowing the published trace to be
reconstructed by restoring padding and multiplying by two.

Record annotations retain patient identity as `subject_ids`, sex, and any published `death` and
`follow_up` in years. All-zero exams are excluded. Mortality is not moved to another exam of the
same patient because its follow-up starts from the exam that states it.

## Tasks and splits

Each retained exam has two tasks sharing the same Record. Neither introduces a prompt.

| | Age | Abnormalities |
| --- | --- | --- |
| Basis | [Lima et al. 2021][lima] | [Ribeiro et al. 2020][ribeiro] |
| Type | `ScalarPredictionTask` | Multi-label `ClassificationTask` |
| Target | Age in years | Published diagnosis flags, with the extensions below |
| Split | Test | Train |
| Task annotation | Published `nn_predicted_age` | None |

The abnormality label set starts with the authors' output order: `1dAVb`, `RBBB`, `LBBB`, `SB`,
`AF`, `ST`. It adds two mutually exclusive fallback labels:

- `normal_ecg` when the release marks the exam normal.
- `abnormal_unspecified` when all six diagnosis flags and `normal_ecg` are false. This is a
  connector-defined label, not an identified clinical finding.

These additions keep targets nonempty but extend the authors' six-output classification task.
The release does not fully document the annotation procedure for its diagnosis and normal flags;
absence of all six flags does not establish a normal ECG. Predicted ages are retained as published,
including negative values, and are task annotations rather than model inputs.

The release has no per-exam split. The age study holds out CODE-15%, while the diagnosis repository
identifies it as training data. The connector uses those roles. **Training on abnormality tasks
exposes the ECGs used by the age test tasks.** Reproducing the age study requires training on other
data; the authors' diagnosis validation membership is unavailable.

There is no mortality task. Follow-up for surviving patients is censored; the annotations retain
the data used by the age paper's survival analysis without inventing a prediction target.

## Source checks and limitations

Conversion checks metadata consistency, joins every exam by ID, and scans the HDF5 files for
structure, finite values and padding. One trace reread supplies all twelve leads and checks its
fingerprint against the scan. Source failures raise `TimeNetConnectorError` with file and row
context. Raw files must remain available until writing finishes.

The connector uses the lead order stated by this release and its preprocessing code; the separate
827-exam test set documents a different augmented-lead order. It does not repair predicted ages
or assign diagnoses absent from the source.

Tests use small fixtures. A full build has not been verified. Download hashes and the card's
post-exclusion counts come from earlier work and have not been rechecked by these tests. Real
padding patterns, peak memory and build time also remain unverified. Counts are not enforced;
unwritten HDF5 chunks that read as all zeros would be excluded along with other all-zero traces.

## Local build

```bash
uv run timenet-build build zenodo/code-15 --out /tmp/code15-registry
```

[zenodo]: https://zenodo.org/records/4916206
[lima]: https://doi.org/10.1038/s41467-021-25351-7
[ribeiro]: https://doi.org/10.1038/s41467-020-15432-4
[prep]: https://github.com/antonior92/ecg-preprocessing
