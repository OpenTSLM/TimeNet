# CAPTURE-24

`oxwearables/capture24` holds CAPTURE-24. The dataset contains wrist acceleration recordings from
151 participants during daily life. Each recording covers about one day. Activity labels come from
body-camera images and sleep diaries. The dataset supports human activity recognition.

The connector reads the 6.9 GB zip file from the Oxford Research Archive. The authors' benchmark
code, `OxWearables/capture24` at commit `f861b44`, has a separate academic-use license. The
connector reimplements its windowing without copying that code.

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
which keeps about seven, as the authors' own loader does. This rounding is far more precise than
the sensor's resolution, and float32 halves the stored size.

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

The connector creates one Record for each segment of a recording. A segment is a run of samples
exactly 10 ms apart. A gap starts a new segment, and nothing fills it. So 143 participants have one
Record each, and the 8 participants whose recording has a gap have two. That makes 159 Records for
151 participants.

Each Record has one Source, `Axivity AX3`, with three float32 signals: `x`, `y`, and `z`. The
signals use `standard_gravity` units.

The timestamps have no time zone, and their dates are randomized, so the origin has no timestamp.
The connector stores the following information instead:

| Where | Key | Value |
| --- | --- | --- |
| Record metadata | `file` | The recording's file name |
| Record metadata | `segment` | The index of the Record's segment in its recording, from 0 |
| Record annotation | `subject_ids` | A list that holds the participant id, such as `["P001"]` |
| Record annotation | `age_band` | The age band of `metadata.csv`, such as `38-52` |
| Record annotation | `sex` | `female` or `male`, from the `F` or `M` of `metadata.csv` |
| Record annotation | `recording_start_local` | The published timestamp of the recording's first sample, in ISO 8601 with microseconds and no zone, such as `2016-11-13T02:18:00.000000` |

### Activity annotations

Each Source carries the labels of its segment. A run is a sequence of consecutive samples with the
same raw label. Each run becomes seven annotations on the Source, all with the run as their `span`,
from its first sample to one period after its last:

| Key | Value |
| --- | --- |
| `activity` | The raw label, such as `7030 sleeping;MET 0.95` |
| `activity_willetts_specific_2018` | The raw label's class in WillettsSpecific2018, such as `sleep` |
| `activity_willetts_met_2018` | Its class in WillettsMET2018 |
| `activity_doherty_specific_2018` | Its class in DohertySpecific2018 |
| `activity_willetts_2018` | Its class in Willetts2018 |
| `activity_doherty_2018` | Its class in Doherty2018 |
| `activity_walmsley_2020` | Its class in Walmsley2020 |

Unlabeled samples belong to no run, so the annotations also show which samples have a label. A run
never crosses a gap.

Build model inputs from the signals and exclude all Source annotations whose key starts with
`activity`. Those annotations contain the task answers, so including them exposes the answers to the
model.

### Label sets

A label scheme groups raw labels into activity classes. A label set lists the allowed classes of
one scheme. The authors' `benchmark.py` scores two schemes, so the connector defines two label
sets:

| Label set | What it classifies | Classes, sorted by their text |
| --- | --- | --- |
| `willetts_specific_2018` | Activity type | `bicycling`, `household-chores`, `manual-work`, `mixed-activity`, `sitting`, `sleep`, `sports`, `standing`, `vehicle`, `walking` |
| `walmsley_2020` | Movement behavior | `light`, `moderate-vigorous`, `sedentary`, `sleep` |

The other four schemes have no tasks. Their classes are only in the activity annotations.

### Tasks

The benchmark has one task: classify a 10 s window's activity from its acceleration. Each window
that the authors' `prepare_data.make_windows` keeps becomes one `ClassificationTask` per scheme that
their `benchmark.py` scores:

| Field | Value |
| --- | --- |
| `scope` | The window: a 10 s interval that names no signal, so the whole Record stays available |
| `targets` | The window's class in the scheme |
| `label_set` | `willetts_specific_2018` or `walmsley_2020` |
| `split` | `train` for P001 to P100, `test` for P101 to P151 |

A bin is a fixed 10 s time interval. `make_windows` starts the bins at the recording's first sample.
After a gap, the bins stay on this original time grid. A bin becomes a window if it contains exactly
1000 samples and at least one labeled sample, so a window's class can come from a single labeled
sample. No window crosses a gap.

A window's class is the most common class among its labeled samples. If classes tie, the class
that sorts first by its text wins, as in the authors' `Series.mode`.

## Known issues upstream

- The paper reports 922,199 windows, and the authors' repository datasheet reports 939,017.
  Their `make_windows` at commit `f861b44` keeps 934,762.
- The authors' repository README gives P102 to P151 as the test set. The paper and `benchmark.py`
  use P101 to P151.
- The AX3 was set to ±8 g. Some values reach 8 g or more in magnitude after the authors' gravity
  calibration. The connector keeps these values as published.
- The paper does not explain the gaps in eight recordings. P016 and P017 each skip exactly one
  hour at about 1 a.m., and the samples after the gap have labels. This matches a clock change in
  the local timestamps. If a clock change is the cause, no data is missing, and the second Record
  starts one hour later than the true elapsed time. P060 and P094 end with only 10 s and 13 s of
  samples after a gap of more than two hours. The connector keeps the published timestamps and
  marks no timestep as missing.
