# CAPTURE-24

`oxwearables/capture24` holds CAPTURE-24 ([Chan et al., Sci Data 2024][paper]).
The dataset contains wrist acceleration recordings from 151 participants during daily life.
Each recording covers about one day. Activity labels come from body-camera images and sleep
diaries. The dataset supports human activity recognition.

The connector reads the 6.9 GB zip file from the [Oxford Research Archive record][ora].
It uses the SHA-256 digest (`69740c22…40e2`) to identify the expected archive and detect changes.
The dataset card records the same digest as `source_revision`.
The data license is CC BY 4.0, and the dataset DOI is [10.5287/bodleian:NGx0JOMP5][doi]. The
archive's terms also ask each user to keep the participants' data confidential and not to try to
identify them. The authors' benchmark code is [OxWearables/capture24][code]. It has a separate
academic-use license. The connector reimplements its windowing without copying that code.

The connector is one file, `connector.py`. Its tests build a small fake release, so they need no
download.

[paper]: https://doi.org/10.1038/s41597-024-03960-3
[ora]: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
[doi]: https://doi.org/10.5287/bodleian:NGx0JOMP5
[code]: https://github.com/OxWearables/capture24/tree/f861b44

## The source data

The archive's `capture24/` directory holds:

| File | Content |
| --- | --- |
| `P001.csv.gz` … `P151.csv.gz` | One row per sample: `time,x,y,z,annotation` |
| `metadata.csv` | `pid,age,sex`: an age band (`18-29`, `30-37`, `38-52`, `53+`) and `F` or `M` |
| `annotation-label-dictionary.csv` | Each of the 206 raw labels, and its class in six label schemes |

An Axivity AX3 on the dominant wrist records acceleration along three axes, in g, at 100 Hz with a
range of ±8 g. The published values are not raw device output. The authors resampled each recording
to exactly 100 Hz with nearest-neighbor interpolation, which selects the nearest existing sample.
They also applied gravity autocalibration, which corrects the sensor's calibration errors by using
the periods when the device is still and measures only gravity.

The source values have eight significant digits. The connector reads and stores them as float32,
which keeps about seven, as the authors' own loader does. This rounding is far below the sensor's
resolution, and float32 halves the stored size.

The `time` column contains timestamps without a time zone. For anonymity, the authors randomized
the dates so that every recording is dated 2016. They also shifted each timestamp by a small random
amount. The paper names no time zone, so these timestamps do not give exact local times.

Consecutive samples are exactly 10 ms apart, except for one gap of 1 to 4 hours in each of eight
recordings. These recordings are P008, P016, P017, P060, P077, P094, P097, and P142.

The `annotation` column contains a [Compendium of Physical Activity][cpa] entry, such as
`7030 sleeping;MET 0.95`, or an empty field. An empty field means that the sample has no label.

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

The timestamps have no time zone, and their dates are randomized, so the connector leaves
`start_time` unset. It stores the following information instead:

| Where | Key | Value |
| --- | --- | --- |
| Record metadata | `file` | The recording's file name |
| Record annotation | `subject_ids` | A list that holds the participant id, such as `["P001"]` |
| Record annotation | `age_band` | The age band of `metadata.csv`, such as `38-52`. Other connectors use `age` for an exact age in years, so this key is different |
| Record annotation | `sex` | `female` or `male`, from the `F` or `M` of `metadata.csv` |
| Record annotation | `recording_start_local` | The first sample's published timestamp, in ISO 8601 with microseconds and no zone, such as `2016-11-13T02:18:00.000000` |

`recording_start_local` is the anonymized timestamp that the authors published. Its date is not
the original recording date, and its time of day includes a small random shift. The annotation's
`description` explains these changes so that readers do not mistake it for the true local time.

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

Build model inputs from the signals and exclude the Source activity annotations. Those annotations
contain the task answers, so including them exposes the answers to the model.

A task's `inputs` contains the whole Record, including its Source annotations.
The task's `input_annotations` is empty. That field lists selected annotations but does not remove
annotations from the Record.

### Vocabularies

A label scheme groups raw labels into activity classes. Its vocabulary lists the allowed classes.
The authors' `benchmark.py` scores two schemes, so the connector defines two vocabularies.

Each task uses the vocabulary for its scheme. The dataset stores each vocabulary as a dataset
annotation, and each task refers to it through `target_schema`. The dataset has no other dataset
annotations.

| Dataset annotation | Classes, sorted by their text |
| --- | --- |
| `willetts_specific_2018_vocabulary` | `bicycling`, `household-chores`, `manual-work`, `mixed-activity`, `sitting`, `sleep`, `sports`, `standing`, `vehicle`, `walking` |
| `walmsley_2020_vocabulary` | `light`, `moderate-vigorous`, `sedentary`, `sleep` |

The classes are written in `connector.py`, not read from the release. The build fails if the label
dictionary gives a class that the scheme's vocabulary does not list.

The label dictionary must contain exactly the raw label column and the six scheme columns.
Each row must name a class in every scheme, including schemes without tasks.
This prevents blank classes from entering the activity annotations.

The `metadata.csv` table must list at least one participant and use the expected age bands and sex
values. Both tables must contain exactly the expected columns. A blank or missing field fails the
build. A row with more fields than the header also fails, so no extra value is dropped unnoticed.
Violations of these rules raise `TimeNetConnectorError`.

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

The task stream follows the order used by `benchmark.py`: all WillettsSpecific2018 tasks, then all
Walmsley2020 tasks. Within each scheme, participants appear in ID order.
Each participant's windows appear in time order.

A bin is a fixed 10 s time interval. `make_windows` starts the bins at the recording's first sample.
After a gap, the bins stay on this original time grid. A bin becomes a window if it contains exactly
1000 samples and at least one labeled sample. So a window's class can come from a single labeled
sample. The activity annotations show how much of each window has a label.

A bin that crosses a gap usually has fewer than 1000 samples, so it is dropped. A step between one
and two sample periods is also a gap, but the bin around it can still hold 1000 samples. The build
fails on such a bin, so no window crosses a gap.
A window's class is the most common class among its labeled samples. If classes tie, the class
that sorts first by its text wins, as in the authors' `Series.mode`.
A window's scope starts at its first sample. After a gap, that sample can fall up to 10 ms after
the start of its bin.

The tasks use P101 to P151 as the test set, as specified in the paper and by `range(101, 152)` in
`benchmark.py`. The authors' repository README instead says P102 to P151.
The authors randomly select groups from the training set for validation.
This selection is not fixed, so the connector assigns no task to `validation`.

A task's `inputs` contains the participant's whole Record, and its `scope` identifies the window.
For a model that reads one window, find the Source whose time span contains the scope, and read the
1000 samples of its `x`, `y`, and `z` within the scope. In a Record with a gap, both Sources have
signals with the same names, so pick the Source by its time span, not a signal by its name.

The authors' sequence models read several windows together. Their recurrent neural network (RNN)
reads 8 consecutive windows. Their hidden Markov model (HMM) reads a participant's whole sequence.
For these models, use one participant's tasks from one scheme. Those tasks are already in time
order.

A model that reads samples outside its window can guess the class from its neighbors. Two adjacent
windows have the same class in more than 99.9% of pairs, because the labels come from photos taken
every 20 to 40 seconds and from sleep diaries.

## What a real build gave

An earlier version of this connector applied the same rules across several modules.
A full build of the pinned archive with that version produced:

- 151 Records and 159 Sources: 143 Records with one Source and 8 with two.
- 1,869,524 classification tasks, which is 934,762 windows with two schemes each.

This version of the connector has not run on the full archive yet. Its tests use a small fake
release to exercise the same rules.

## Known issues upstream

- The paper reports 922,199 windows, and the authors' repository datasheet reports 939,017.
  Their `make_windows` at commit `f861b44` keeps 934,762.
- The AX3 was set to ±8 g. Some values reach 8 g or more in magnitude after the authors' gravity
  calibration. The connector keeps these values as published.
- `train_deep_models.py` imports `old_code`, which is not in the repository, and loads
  `Y_willetts.npy` and `pid.npy`, which `prepare_data.py` does not write. The CNN and RNN results
  cannot be reproduced from the repository as published.
