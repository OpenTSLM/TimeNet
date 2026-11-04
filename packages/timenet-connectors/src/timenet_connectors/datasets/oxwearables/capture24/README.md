# CAPTURE-24

`oxwearables/capture24` holds CAPTURE-24 ([Chan et al., Sci Data 2024][paper]).
The dataset contains wrist acceleration recordings from 151 participants during daily life.
Each recording covers about one day. Activity labels come from body-camera images and sleep
diaries. The dataset supports human activity recognition.

The connector reads the one zip of the [Oxford Research Archive record][ora], pinned by its
SHA-256 (`69740c22…40e2`, 6.9 GB). The dataset card records the same digest as `source_revision`.
The data license is CC BY 4.0. The authors' benchmark code is
[OxWearables/capture24][code]; it is under a separate academic-use license, so the connector
reimplements its windowing and copies no code.

The connector is one file, `connector.py`. Its tests build a small fake release, so they need no
download.

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
  eight significant digits. The connector reads and stores them as float32, which keeps about
  seven, as the authors' own loader does. The rounding is far below the sensor's resolution, and
  float32 halves the stored size.
- **Clock.** `time` is a timestamp with no zone. For anonymity the authors randomized the dates, so
  every recording is dated 2016. They also shifted each timestamp by a small random amount, so the
  time of day is close to the participant's local time but not exact. Every step is exactly 10 ms,
  except for one gap of 1 to 4 hours in each of eight recordings (P008, P016, P017, P060, P077,
  P094, P097, P142).
- **Labels.** `annotation` is a [Compendium of Physical Activity][cpa] entry, such as
  `7030 sleeping;MET 0.95`, or empty. An empty field is an unlabeled sample.

[cpa]: https://sites.google.com/site/compendiumofphysicalactivities/home

## The TimeNet dataset

### Records

The connector creates one Record per participant and one Source per segment.
A segment is a run of samples exactly 10 ms apart. A gap starts a new segment, and nothing fills it.
Each segment has three float32 signals: `x`, `y`, and `z`.
The signals load when needed and use `standard_gravity` units.

The `offset_us` value gives the time from the recording's first sample to the segment's first
sample.
There are 143 Records with one Source and 8 with two.

The timestamps carry no zone and their dates are randomized, so `start_time` is unset. These facts
are kept instead:

| Where | Key | Value |
| --- | --- | --- |
| Record metadata | `file` | The recording's file name |
| Record annotation | `subject_ids` | A list that holds the participant id, such as `["P001"]` |
| Record annotation | `age_band` | The age band of `metadata.csv`, such as `38-52`. Other connectors use `age` for an exact age in years, so this key is different |
| Record annotation | `sex` | `female` or `male`, from the `F` or `M` of `metadata.csv` |
| Record annotation | `recording_start_local` | The first sample's published timestamp, in ISO 8601 with microseconds and no zone, such as `2016-11-13T02:18:00.000000` |

`recording_start_local` is the anonymized timestamp that the authors published. Its date is not
the recording's. Its time of day is the participant's local time, shifted by a small random amount.
The annotation's `description` says this too, so a reader of the dataset alone does not take it as
the true local time.

### Activity annotations

Each Source carries the raw labels of its segment. A run of consecutive samples with the same
raw label becomes one annotation on the Source:

| Field | Value |
| --- | --- |
| `key` | `activity` |
| `value` | The raw label, such as `7030 sleeping;MET 0.95` |
| `span` | The run, from its first sample to one period after its last |
| `metadata` | The raw label's class in each of the six schemes of the label dictionary, keyed by scheme name, such as `{"Walmsley2020": "sleep", ...}` |

Unlabeled samples belong to no run, so the annotations also show which samples have a label. A
run never crosses a gap. The signals carry no annotations.

The activity annotations hold the answer to every task. A model input must leave them out. A task
names no annotation in `input_annotations`, so a model that reads only a task's inputs and scope
does not see them.

### Dataset annotations

| Key | Value |
| --- | --- |
| `willetts_specific_2018_vocabulary` | WillettsSpecific2018's 10 classes, in string order |
| `walmsley_2020_vocabulary` | Walmsley2020's 4 classes, in string order |

### Vocabularies

Each task draws its target from one of two vocabularies, one for each scheme that the authors'
`benchmark.py` scores. A vocabulary is the closed list of a scheme's classes. Each task names its
vocabulary in `target_schema`.

| Vocabulary | Classes, in string order |
| --- | --- |
| `willetts_specific_2018_vocabulary` | `bicycling`, `household-chores`, `manual-work`, `mixed-activity`, `sitting`, `sleep`, `sports`, `standing`, `vehicle`, `walking` |
| `walmsley_2020_vocabulary` | `light`, `moderate-vigorous`, `sedentary`, `sleep` |

The classes are written in `connector.py`, not read from the release. The build fails if the label
dictionary gives a class that the scheme's vocabulary does not list.

The build also fails if the label dictionary does not have exactly the raw label column and the six
scheme columns, or if a class is blank in any scheme. A blank class in a scheme without tasks would
otherwise go into the activity annotations unnoticed. In the same way, the build fails on an age
band or a sex that `metadata.csv` does not use, and on a `metadata.csv` that lists no participants.
Both tables must have exactly their columns, and the build fails on a row with more fields than the
header, so no extra value is dropped unnoticed. These failures raise `TimeNetConnectorError`.

The dictionary has four more schemes (`WillettsMET2018`, `DohertySpecific2018`, `Willetts2018` and
`Doherty2018`). They have no tasks. Their classes are in the metadata of each activity annotation.
A task on the raw labels or on another scheme needs a change to the connector.

### Tasks

The benchmark has one task: classify a 10 s window's activity from its acceleration. Each window
that the authors' `prepare_data.make_windows` keeps becomes one `ClassificationTask` per scheme that
their `benchmark.py` scores:

| Field | Value |
| --- | --- |
| `scope` | The window: a 10 s interval that names no signal, so the whole Record stays available |
| `targets` | The window's class in the scheme |
| `target_schema` | `willetts_specific_2018_vocabulary` or `walmsley_2020_vocabulary` |
| `split` | `train` for P001 to P100, `test` for P101 to P151 |

There is no prompt because the benchmark asks no question in words.

The task stream holds one benchmark per scheme, as `benchmark.py` runs them: first every
WillettsSpecific2018 task, then every Walmsley2020 task. Within a scheme, the tasks follow the
participants in id order and each participant's windows in time order.

`make_windows` cuts each recording into 10 s bins that start at its first sample. After a gap the
bins stay on the first segment's grid. It keeps a bin with exactly 1000 samples and at least one
labeled sample, so a kept window can have as few as one labeled sample. The activity annotations
show how much of a window has a label. The connector also fails the build if a kept window holds
1000 samples across a gap, so no window crosses a gap. A window's class is the most common class
among its labeled samples. A tie goes to the class first in string order, as the authors'
`Series.mode` does. A window's scope starts at its first sample, which after a gap can be up to 10
ms into its bin.

**Split.** The paper's test set is P101 to P151, and the authors' `benchmark.py` tests
`range(101, 152)`, so the tasks carry that split. The README of the authors' repository says P102
to P151. The authors take a validation set from train with a random group shuffle, which is not
fixed, so no task is `validation`.

**The scope and the Record.** A task's `inputs` is its participant's whole Record, and its `scope`
is the window. A model that reads one window uses the `x`, `y` and `z` of the one segment that
holds the scope, 1000 samples each. In a Record with a gap, both segments have signals with the
same names, so pick the segment by its time span, not the signal by name. The authors' sequence
models read more: their RNN reads 8 consecutive windows and their HMM a participant's whole
sequence. For those, take one scheme's tasks of a participant, which are already in time order.

A model that reads samples outside its window can guess the class from its neighbors. Two adjacent
windows have the same class in more than 99.9% of pairs, because the labels come from photos taken
every 20 to 40 seconds and from sleep diaries.

## What a real build gave

A full build of the pinned archive, made with an earlier version of this connector that applied
the same rules in several modules, gave:

- 151 Records and 159 Sources: 143 Records with one Source and 8 with two.
- 1,869,524 classification tasks, which is 934,762 windows with two schemes each.

We have not run this version of the connector on the full archive yet. Its tests check the rules on
a small fake release.

## Known issues upstream

- **Window counts.** The paper reports 922,199 windows, and the datasheet in the authors'
  repository 939,017. Their `make_windows` today keeps 934,762.
- **Values past the sensor range.** The AX3 was set to ±8 g. Some values reach 8 g or more in
  magnitude after the authors' gravity calibration. They are kept as published.
- `train_deep_models.py` imports `old_code`, which is not in the repository, and loads
  `Y_willetts.npy` and `pid.npy`, which `prepare_data.py` does not write. The CNN and RNN results
  cannot be reproduced from the repository as published.
