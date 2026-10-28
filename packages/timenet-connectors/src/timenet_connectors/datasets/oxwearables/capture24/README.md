# CAPTURE-24

`oxwearables/capture24` holds CAPTURE-24 ([Chan et al., Sci Data 2024][paper]): about a day of
free-living wrist accelerometry from each of 151 participants, labelled from body-camera images
and sleep diaries, for human activity recognition.

The connector reads the one zip of the [Oxford Research Archive record][ora], pinned by its
SHA-256 (`69740c22…40e2`, 6.9 GB). The data license is CC BY 4.0. The authors' benchmark code is
[OxWearables/capture24][code]; it is under a separate academic-use license, so the connector
reimplements its windowing and copies no code.

[paper]: https://doi.org/10.1038/s41597-024-03960-3
[ora]: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
[code]: https://github.com/OxWearables/capture24

## The source data

The archive's `capture24/` directory holds:

| File | Content |
| --- | --- |
| `P001.csv.gz` … `P151.csv.gz` | One row per sample: `time,x,y,z,annotation` |
| `metadata.csv` | `pid,age,sex`: an age band (`18-29`, `30-37`, `38-52`, `53+`) and `F` or `M` |
| `annotation-label-dictionary.csv` | Each of the 206 raw labels, and its class in six label schemes |

- **Signal.** An Axivity AX3 on the wrist, three axes in g, at 100 Hz. Values are written with
  eight significant digits, so the connector keeps them as float64; float32, which the authors
  load, would round them.
- **Clock.** `time` is a timestamp with no zone. For anonymity the authors randomized the dates
  and shifted the times by a small random amount, so every recording is dated 2016, though most
  were recorded in 2014 and 2015. The time of day still holds: labelled sleep is 97–100% of the
  labelled samples from 00:00 to 04:59, and 0% from 10:00 to 20:59. Every step is exactly 10 ms,
  except for one gap in each of eight recordings (P008, P016, P017, P060, P077, P094, P097, P142),
  of 1.0 to 3.9 hours.
- **Labels.** `annotation` is a [Compendium of Physical Activity][cpa] entry, such as
  `7030 sleeping;MET 0.95`, or empty: 476,286,350 of the 1,398,038,232 samples (34%) are
  unlabelled. Every non-empty value is in the dictionary. The release has 3,883 hours of
  recording, as the paper reports, and 2,560.4 labelled hours, where the paper reports 2,562.

[cpa]: https://sites.google.com/site/compendiumofphysicalactivities/home

## The TimeNet dataset

### Records

One Record per participant, with one Source per segment: a run of samples exactly 10 ms apart.
A gap starts a new segment, and nothing fills it. Each segment's `x`, `y` and `z` are lazy float64
signals in `standard_gravity`, on a 100 Hz axis whose `offset_us` is the segment's first sample's
time from the recording's first sample. 143 Records have one Source, and 8 have two.

The timestamps carry no zone and their dates are randomized, so `start_time` is unset; the first
sample's timestamp, as written, is the `recording_start_local` annotation. Its time of day is the
participant's local time; its date is not the recording's. `age` and `sex` are Record annotations.
The participant id is the core `subject_id` of `Record.metadata`, and the file name its `capture24`
extension; `records.record_metadata(record)` reads both back.

Each run of one raw label within a segment is an `activity_label` span annotation, shared by the
segment's three signals. Unlabelled samples carry none.

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_url`, `source_sha256` | The pinned archive |
| `activity_label_vocabulary` | The 206 raw labels, in dictionary order |
| `<scheme>_vocabulary` | A scheme's classes, sorted; a task's `target_schema` |
| `<scheme>_mapping` | A scheme's class of each raw label, aligned with `activity_label_vocabulary` |

The six schemes are `willetts_specific_2018`, `willetts_met_2018`, `doherty_specific_2018`,
`willetts_2018`, `doherty_2018` and `walmsley_2020`. The mappings let a reader turn any
`activity_label` into a class of any scheme.

### Tasks

The benchmark has one task: classify a 10 s window's activity from its acceleration. Each window
that the authors' `prepare_data.make_windows` keeps becomes one `ClassificationTask` per scheme that
their `benchmark.py` scores, `WillettsSpecific2018` (10 classes) and `Walmsley2020` (4 classes):

| Field | Value |
| --- | --- |
| `scope` | The window's 1000 samples, naming no signal, so the whole Record stays available |
| `targets` | The window's class in the scheme |
| `target_schema` | `willetts_specific_2018_vocabulary` or `walmsley_2020_vocabulary` |
| `split` | `train` for P001 to P100, `test` for P101 to P151 |

There is no prompt; the benchmark asks no question in words.

`make_windows` cuts each recording into 10 s bins anchored at its first sample, so after a gap the
bins stay on the first segment's grid. It keeps a bin with exactly 1000 samples, no NaN, and at
least one labelled sample, so no window crosses a gap. A window's class is the most common class
among its labelled samples; a tie goes to the class first in string order, as `Series.mode` sorts.
A window's scope starts at its first sample, which after a gap can be up to 10 ms into its bin.

**Split.** The paper's derivation set is P001 to P100 and its test set P101 to P151, 51
participants, and the authors' `benchmark.py` tests `range(101, 152)`, so the tasks carry that
split: 626,314 train and 308,448 test windows. The README of the authors' repository says P102 to
P151. The authors take a validation set from train with a random group shuffle split, which is not
fixed, so no task is `validation`.

**The scope and the Record.** A task's `inputs` is its participant's whole Record, about a day,
and its `scope` is the window. The scope names no signal, so the whole Record stays available, as
the authors' models use different amounts of it. Every one predicts the same per-window target:

| Paper rows | What labels one window |
| --- | --- |
| RF, XGBoost, CNN | The window alone |
| RNN (a CNN-LSTM) | 8 consecutive windows, 80 s, read in both directions (`seq_length: 8`) |
| `+ HMM` | The participant's whole sequence of per-window predictions |

The window's samples are the `x`, `y` and `z` of the one segment whose window holds the scope, cut
with each signal's own axis (`index_at_or_after` of the scope's start and end): the authors'
1000 × 3 `X` row. In a Record with a gap, the other segment's signals carry the same names, so pick
the segment by its time window, not the Record's signals by name. For the sequence rows, take a
participant's tasks in `scope.start_us` order. No TimeNet loader does either for you; the torch
view returns the whole Record with all of its tasks.

A model that sees more than its row uses gets the class nearly for free:

- **Neighbouring windows.** Two adjacent windows have the same class in 99.995% of 921,616 pairs
  in Walmsley2020, and 99.990% in WillettsSpecific2018, since the labels come from photos taken
  every 20 to 40 seconds and from sleep diaries. A per-window model shown the windows around its
  own scores well above the paper's per-window rows.
- **The raw labels.** Every target is derived from the `activity_label` spans on the Record's
  signals. They are not model input, since no task names them in `input_annotations`, and the
  torch view leaves signal annotations out of its items. A loader that turns signal annotations
  into model context hands the model its answer.
- **The other tasks.** A torch item holds every task of its Record, targets included.

The authors' RNN chunks windows in array order, so a chunk can span two participants, or both
sides of a gap; their HMM treats a participant's windows as consecutive across a gap.

**Scoring as the authors do.** Read each target with `timenet.types.text.read_value`, which gives
the class as a `str`, and pass the targets and predictions of one scheme's test tasks to the
authors' `eval.metrics_report`, with the participant id as `groups` for their bootstrap.

## How the connector builds it

Each step's output carries what the next step needs:

```
download   capture24.zip ── ensure_archive, sha256 ──► capture24/

parse      annotation-label-dictionary.csv ──► LabelDictionary
                                                 rows: DictionaryRow, ...
           metadata.csv ─────────────────────► Participant
                                                 pid, age, sex,
                                                 recording_path

for each   P<nnn>.csv.gz ── read_recording ──┬─► Recording
Participant                                  │     path, n_rows,
                                             │     start_local, segments
                                             │       └─ Segment
                                             │            first_row,
                                             │            n_values,
                                             │            offset_us,
                                             │            label_runs
                                             │              └─ LabelRun
                                             │
                                             └─► Samples (one participant
                                                   at a time)
                                                   offsets_us, label_codes
                                                     │
                                         make_windows ▼
                                                 Windows
                                                   start_us, class_codes

           build_record(Participant, Recording) ──► Record
             one Source per Segment: lazy x, y, z
             one activity_label span per LabelRun
           _ParsedParticipant(participant, record, windows)

populate   dataset annotations, then the Records, then the task stream:
           Capture24WindowTask per window and scheme ──► ClassificationTask

write      AccelerationLoader(recording, column, segment)
             └─ read_acceleration(recording): x, y, z, once per recording
```

| Module | Holds |
| --- | --- |
| `connector.py` | The download, the provenance annotations, and `convert` |
| `release.py` | What is pinned: the archive and its sha256, files, columns, rate, counts |
| `benchmark.py` | What the authors' benchmark fixes: the window, the schemes, the split |
| `keys.py` | Every annotation key the connector writes |
| `vocabulary.py` | The closed sets: label schemes, sexes, age bands, the label dictionary |
| `participants.py` | `metadata.csv`: `Participant` and `read_participants` |
| `recordings.py` | One `P<nnn>.csv.gz`: `Recording`, `Segment`, `LabelRun`, `Samples` |
| `windows.py` | The authors' `make_windows` rule: `Windows` and `make_windows` |
| `records.py` | A participant's Record and the lazy `AccelerationLoader` |
| `tasks.py` | `Capture24WindowTask`, which becomes a `ClassificationTask` |
| `loader.py` | `Capture24DatasetLoader`, which runs the parse and streams the tasks |

## Verification

`VERIFICATION.md` records each check. In short, on the pinned archive:

- 151 participants, 159 segments and 934,762 windows, pinned in `release.PINNED_COUNTS` and
  checked by every build.
- A full build, read back, rebuilds the authors' `make_windows` output for every participant
  exactly: `X`, `T`, `Y_WillettsSpecific2018`, `Y_Walmsley2020`, `Y_anno` and `P`, and the split
  of their `benchmark.train_test_split`. Every acceleration value, sample time and raw label
  equals an independent parse of the CSV files.
- Two full builds store the same values, byte for byte, and the same rows in every table.

## Known issues upstream

- **Window counts.** The paper reports 922,199 windows (618,129 derivation, 304,070 test), and
  the datasheet in the authors' repository 939,017. Their `make_windows` today keeps 934,762, which
  this build holds. Dropping the 13,095 windows whose most common raw label is "unlabelled" gives
  921,667, which isn't the paper's count either, so the difference is unexplained.
- **Values past the sensor range.** The AX3 was set to ±8 g, and 15,339 values in 138 recordings
  reach 8 g or more in magnitude, up to 8.67 g, after the authors' gravity calibration. They are
  kept as published.
- `train_deep_models.py` imports `old_code`, which isn't in the repository, and loads
  `Y_willetts.npy` and `pid.npy`, which `prepare_data.py` doesn't write. The CNN and RNN results
  can't be reproduced from the repository as published.
