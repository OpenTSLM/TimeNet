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
- **Clock.** `time` is a wall-clock timestamp with no zone. Every step is exactly 10 ms, except
  for one gap in each of eight recordings (P008, P016, P017, P060, P077, P094, P097, P142), of
  1.0 to 3.9 hours. The two gaps of exactly one hour (P016, P017) don't fall on a UK clock change.
- **Labels.** `annotation` is a [Compendium of Physical Activity][cpa] entry, such as
  `7030 sleeping;MET 0.95`, or empty: 476,286,350 of the 1.40 billion samples (34%) are
  unlabelled. Every non-empty value is in the dictionary.

[cpa]: https://sites.google.com/site/compendiumofphysicalactivities/home

## The TimeNet dataset

### Records

One Record per participant, with one Source per segment: a run of samples exactly 10 ms apart.
A gap starts a new segment, and nothing fills it. Each segment's `x`, `y` and `z` are lazy float64
signals in `standard_gravity`, on a 100 Hz axis whose `offset_us` is the segment's first sample's
time from the recording's first sample. 143 Records have one Source, and 8 have two.

The timestamps carry no zone, so `start_time` is unset; the first sample's timestamp, as written,
is the `recording_start_local` annotation. `age` and `sex` are Record annotations. The participant
id is the core `subject_id` of `Record.metadata`, and the file name its `capture24` extension;
`records.record_metadata(record)` reads both back.

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
`benchmark.py` scores, `WillettsSpecific2018` (10 classes) and `Walmsley2020` (4 classes):

| Field | Value |
| --- | --- |
| `scope` | The window's 1000 samples, naming no signal |
| `targets` | The window's class in the scheme |
| `target_schema` | `willetts_specific_2018_vocabulary` or `walmsley_2020_vocabulary` |
| `split` | `train` for P001 to P100, `test` for P101 to P151 |

There is no prompt; the benchmark asks no question in words.

`make_windows` cuts each recording into 10 s bins anchored at its first sample, so after a gap the
bins stay on the first segment's grid. It keeps a bin with exactly 1000 samples, no NaN, and at
least one labelled sample, so no window crosses a gap. A window's class is the most common class
among its labelled samples; a tie goes to the class first in string order, as `Series.mode` sorts.
A window's scope starts at its first sample, which after a gap can be up to 10 ms into its bin.

**Split.** The README of the authors' repository says P001 to P101 train and P102 to P151 test.
`benchmark.py` tests `range(101, 152)`, P101 to P151, and that is the split the published numbers
come from, so the tasks carry it. The authors take a validation set from train with a random group
shuffle split, which is not fixed, so no task is `validation`.

## Verification

On the pinned archive:

- 151 participants, 159 segments and 934,762 windows, pinned in `release.PINNED_COUNTS` and
  checked by every build.
- The authors' own `make_windows` (at `OxWearables/capture24` `main`, 2026-10-28), run on every
  participant, keeps the same 934,762 windows: the same start timestamps, and the same class in
  both schemes, window for window.
- No sample has a missing or non-finite acceleration value, and no timestamp repeats or goes back.

## Known issues upstream

- `train_deep_models.py` imports `old_code`, which isn't in the repository, and loads
  `Y_willetts.npy` and `pid.npy`, which `prepare_data.py` doesn't write. The CNN and RNN rows of the
  README can't be reproduced from the repository as published.
