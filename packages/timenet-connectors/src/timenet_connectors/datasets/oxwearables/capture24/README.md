# CAPTURE-24

`oxwearables/capture24` contains about one day of wrist acceleration from each of 151 participants
([Chan et al., Sci Data 2024][paper]). Activity labels come from body-camera images and sleep diaries.
The dataset supports human activity recognition.

The connector uses the pinned 6.9 GB archive from the [Oxford Research Archive][ora]. The data is
licensed under CC BY 4.0; its DOI is [10.5287/bodleian:NGx0JOMP5][doi]. The archive's terms also ask
users to keep participant data confidential and not to attempt identification. The authors'
[benchmark code][code] has a separate academic-use license. The connector reimplements its
windowing without copying that code.

[paper]: https://doi.org/10.1038/s41597-024-03960-3
[ora]: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
[doi]: https://doi.org/10.5287/bodleian:NGx0JOMP5
[code]: https://github.com/OxWearables/capture24/tree/f861b44

## What the connector produces

An Axivity AX3 on the dominant wrist measured three-axis acceleration in g, with a range of ±8 g.
The authors resampled the recordings to exactly 100 Hz using nearest-neighbor interpolation and
applied gravity autocalibration. The connector stores float32 values, as the authors' loader does;
this rounds the source's eight significant digits to about seven.

Each participant becomes one Record. Each uninterrupted segment becomes one Source with `x`, `y`,
and `z` signals in `standard_gravity` units. Samples are 10 ms apart. Gaps start new Sources and are
not filled. Signal times are relative to the recording's first sample. Signals load lazily to
limit memory use.

The published timestamps have no time zone. Their dates were randomized to 2016 and their times
shifted slightly for anonymity. The connector leaves `start_time` unset and preserves the first
published timestamp as an annotation; it is not the true recording start time.

| Where | Key | Value |
| --- | --- | --- |
| Record metadata | `file` | Recording file name |
| Record annotation | `subject_ids` | Participant ID, such as `["P001"]` |
| Record annotation | `age_band` | Published age band: `18-29`, `30-37`, `38-52`, or `53+`; no exact age is available |
| Record annotation | `sex` | `female` or `male` |
| Record annotation | `recording_start_local` | First published timestamp in ISO 8601, with microseconds and no zone |

Each consecutive run of one raw label becomes an `activity` annotation on its Source:

| Field | Value |
| --- | --- |
| `value` | Raw label, such as `7030 sleeping;MET 0.95` |
| `span` | From the first labeled sample to one sample period after the last, exclusive |
| `metadata` | The label's class in all six original schemes, keyed by scheme name, such as `{"Walmsley2020": "sleep", ...}` |

These mappings preserve the original label dictionary alongside each raw label, including schemes
without benchmark tasks. Unlabeled samples have no activity annotation, and runs never cross gaps.

## Benchmark tasks

The connector follows the authors' benchmark at commit `f861b44`. Each retained 10-second window
produces one `ClassificationTask` for each of the two benchmarked label sets:

| Label set | Classes |
| --- | --- |
| `willetts_specific_2018` | `bicycling`, `household-chores`, `manual-work`, `mixed-activity`, `sitting`, `sleep`, `sports`, `standing`, `vehicle`, `walking` |
| `walmsley_2020` | `light`, `moderate-vigorous`, `sedentary`, `sleep` |

A task's `targets` holds the window's class, `label_set` identifies the scheme, and `scope` identifies
its 10-second interval. There is no text prompt.

Windows follow these rules:

- Bins start at the recording's first sample and remain on that 10-second grid after gaps.
- A bin must contain exactly 1,000 samples and at least one labeled sample. A window's class can
  therefore come from a single labeled sample.
- The target is the most common class among labeled samples, with ties broken alphabetically.
- A retained window cannot cross a gap. A full, labeled bin that crosses a gap fails the build.
- The scope starts at the first sample in the bin. After a gap, this can be up to 10 ms after the
  bin's start.

P001–P100 are `train`; P101–P151 are `test`. This follows the paper and `benchmark.py`, although the
upstream README says P102–P151. There is no fixed `validation` split: the authors select random
groups from the training participants for validation.

Tasks appear scheme by scheme: WillettsSpecific2018 first, then Walmsley2020. Within each scheme,
participants are in ID order and their windows are in time order.

## Using the data correctly

**Exclude Source activity annotations from model inputs.** They contain the answers. A task's
`inputs` contains the whole Record, including those annotations. An empty `input_annotations`
field does not remove them.

For a model that reads one window, find the Source whose time span contains the task's scope and
read the 1,000 samples of its `x`, `y`, and `z` signals within that scope. Sources on either side of
a gap use the same signal names, so select the Source by time span. The scope identifies the
window but does not restrict access to the rest of the Record.

Sequence models can use one participant's ordered tasks from one scheme. The authors' RNN uses
eight consecutive windows; their HMM uses the participant's whole sequence. Compare these models
separately from models restricted to one window, because they use context outside that window.

## Validation and limitations

A full build from the cached extracted release passed stored-file verification on
2026-11-06. It produced:

- 151 Records, 159 Sources, and 477 signals: 143 Records with one Source and eight with two.
- 934,762 windows, yielding 1,869,524 classification tasks across the two schemes.

See the [build record](BUILD_RECORD.md) for the method, measurements, and limits. Stored-file
verification checks integrity, not agreement with the authors' benchmark. A full build of the
current revision and a comparison of every window's participant, time, and target against the
authors' implementation remain outstanding. Synthetic tests check conversion rules and the
storage round-trip.

The connector does not clip acceleration to the device's configured range.
