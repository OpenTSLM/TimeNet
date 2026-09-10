# SLIP Evaluation Sets

The eleven benchmarks SLIP was evaluated on. One record is one window. A window is a fixed-length
slice of a recording, with one class and one split.

- **id**: `leochen085/slip-eval`
- **source**: https://huggingface.co/datasets/LeoChen085/SlipDataset
- **licence**: MIT

MIT is what the release states. It covers what SLIP added: the windowing, the splits and the
packaging. The studies the windows were cut from keep their own terms.

## The source of truth

| fact | source of truth | the other reading |
| --- | --- | --- |
| how many classes `Beijing_AQI` has | the data: **5** *(measured)* | the card says 4 |
| how many distinct signals `wesad` has | the data: **3** *(measured)* | the card says 13, and the file does ship 13 columns |
| what `AsphaltObstacles` asks | the data: an air-quality prompt over road-surface classes | the card implies a road-surface question |
| the sampling rate of every folder | the card | the data states none |
| whether a folder's values are in a physical unit | the values *(measured)* | the release says nothing |
| how many rows each split holds | both agree, for all 11 folders and both splits *(measured)* | — |

The data wins where the two disagree, because a reader can measure it again. The rate is the one
fact the data does not state. The card is the only source there, and the connector uses it as
stated.

## What the description states

> 11 downstream sensor classification datasets used for linear-probing and zero-shot retrieval
> evaluation, covering activity recognition, clinical diagnosis, stress prediction, and urban
> sensing.

Decides: the question asked of a window is which class it is. Every task is a `ClassificationTask`.

> The evaluation datasets are curated to cover heterogeneous sensor configurations (varying channel
> counts, sampling rates, and sequence lengths) across four distinct application domains.

Decides: the folder fixes the shape of a record, not the release. Signal count and window length are
constant inside every folder. They differ between folders *(measured)*.

> This dataset should **not** be used for clinical decision-making. It is intended for research
> purposes only.

Decides: nothing in the conversion. It is repeated here because five of the eleven folders are
clinical.

The card holds a table with one row per folder: sensor, rate and class count. It is the only
statement of the sampling rate in the release.

## What one record holds

One record is one window.

**Signals.** One `TimeSeries` per inner list of the row's `X` column. The count is constant inside
each folder *(measured)*:

| folder | signals |
| --- | --- |
| `AsphaltObstacles`, `PPG_CVA`, `PPG_DM`, `PPG_HTN` | 1 |
| `sleepEDF` | 2 |
| `wisdm` | 3 |
| `uci_har` | 6 |
| `Beijing_AQI` | 7 |
| `studentlife` | 10 |
| `ptbxl` | 12 |
| `wesad` | 13 |

Every signal in one window has the same length.

**Four folders keep a physical unit. Seven do not.** The release states this nowhere, so the
connector measured it.

The four that keep their units *(measured)*:

| folder | unit | how it was read |
| --- | --- | --- |
| `sleepEDF` | microvolts | values run -211 to 209, with per-channel standard deviations of 21.17 and 12.94 |
| `wisdm` | g | the resultant magnitude over its three axes has a median of 1.0004, which is gravity |
| `uci_har` | g, all six signals | signals 3-5 have a median resultant magnitude of 1.0226, and signals 0-2 have 0.0628 |
| `AsphaltObstacles` | metres per second squared | every window's mean falls inside [-0.011, 0.146], and the per-window standard deviation spans 0.20 to 3.08 |

`AsphaltObstacles` needs care. The scale is real metres per second squared. The zero point is not. A
value is a deviation from the mean of that window, not an absolute magnitude. Its global range over
both splits is -10.04 to 47.96.

The other seven folders are dimensionless. Somebody rescaled them by three methods that do not
agree *(measured)*:

| method | folders | evidence |
| --- | --- | --- |
| a global per-channel z-score | `Beijing_AQI`, `studentlife` | per-channel means and standard deviations sit at 0 and 1 over the whole folder, and per-window means scatter from -2.53 to 7.82 |
| a per-window z-score | `ptbxl` exactly, the three PPG folders approximately | all 155,640 `ptbxl` window-channels have mean 0.0000 and standard deviation 1.0000 |
| min-max into roughly [0, 1] | `wesad` | its three distinct channels run [0.000, 1.163], [-0.192, 1.167] and [-0.107, 1.072] |

For those seven the connector ships the numbers as the release states them. Nobody published the
constants that undo the rescaling.

**The time axis** is a `RegularAxis` at the rate the card gives for that folder. The data states no
rate. The connector uses the card as stated, even where the duration looks wrong.

**Annotations.** No annotation sits on a record, and none is scoped to a signal. Every one is
registered once for the whole dataset. The tasks that need one refer to it by id.

| key | value | where it came from |
| --- | --- | --- |
| `source_benchmark` | the folder, for example `wisdm` | the folder name |
| `split` | `train` or `test` | the file name |
| `label` | one class of that folder | the `text_label` column |
| `vocabulary` | that folder's whole set of classes, in the release's order | every `text_label` the folder holds, ordered by its `label` column |

A record merged from the three PPG folders belongs to all three. It can be `train` for one diagnosis
and `test` for another. So neither the folder nor the split is a property of the record.

**Subjects.** Three folders ship a `participant_id` column *(measured)*:

| folder | what the column holds |
| --- | --- |
| `wisdm` | 46 participants |
| `sleepEDF` | 39 recording ids over 20 subjects |
| `studentlife` | 1,175 session keys over 23 subjects |

Those values become `subject_ids`. The other eight folders ship no subject column, so their records
carry none.

**A subject id states its folder** — `wisdm-1633`, not `1633`. The three folders number their
participants on their own. A bare value merges two of them in any grouped split.

**The record id is positional** — `slip-eval-wisdm-train-000000`. It holds a prefix, the folder, the
split and the place of the row in that split. No folder ships a per-window id. Two builds of one
release give the same ids. A new release invalidates them.

## The tasks this connector builds

| the question | type | count | scope |
| --- | --- | --- | --- |
| which class is this window | `ClassificationTask` | **92,415** *(measured)* | the whole record |

The answer is stored by reference. `target_annotation_ids` names the annotation that holds the
class, so the five long `ptbxl` reports are stored once and not 12,970 times. `target` itself is
unset. That is what "by reference" means.

`target_schema` is the id of the annotation that holds the whole set of classes for that folder, for
example `vocabulary-ptbxl`. One function builds both ends, so the two cannot drift apart.

`prompt` is the folder's own template, verbatim. Two of the templates are wrong, and the connector
keeps them as they are.

The vocabularies hold 2 to 18 values per folder. 92,415 tasks sit on 91,094 records. The three PPG
folders share their windows, so 1,971 of those tasks sit on 650 records.

## Inconsistencies and decisions

### `PPG_CVA`, `PPG_DM` and `PPG_HTN` are one dataset labelled three ways — **Handled**

**Problem.** All three folders hold the same 650 distinct windows. Every pairwise intersection is
650 *(measured)*. They differ only in the diagnosis: stroke, diabetes, hypertension. The card lists
them as three datasets of 657 rows each.

**Decision.** Build 650 records. Each one carries up to three classification tasks. The folder and
the split ride on the task, not on the record.

**Consequence.** The windows are stored once instead of three times, and one record can be asked all
three questions. 650 records carry 1,971 tasks *(measured)*. 643 of them carry three tasks, and 7
carry six.

### Each PPG folder ships 657 rows that hold 650 distinct windows — **Handled**

**Problem.** Seven windows appear twice inside each PPG folder *(measured)*. One of those pairs
straddles the train and test split of its own folder.

**Decision.** Keep both occurrences as tasks on the one record. The task id names the folder, the
split and the row, so the two stay distinct.

**Consequence.** Seven of the 650 PPG records answer the same question twice. For one pair, the same
window is both trained on and tested on.

### `uci_har` has no gyroscope, and the card says it has one — **Handled**

**Problem.** The card lists `uci_har` as "Accelerometer + Gyroscope (6-ch)". Over a sample of
windows, the resultant magnitude of signals 0-2 is 0.0628 and of signals 3-5 is 1.0226 *(measured)*.
Those are body acceleration with gravity removed, and total acceleration with gravity kept.

**Decision.** All six signals are acceleration in g. The connector defines no angular-rate spec.

**Consequence.** A reader who expects three gyroscope channels does not find them. A connector that
believes the card ships three signals of every `uci_har` record in radians per second.

### The signal names are this connector's, not the release's — **Handled**

**Problem.** No file in the release names a signal. The card gives only the sensor kind, the rate
and the class count. Twelve unnamed ECG signals mean nothing without an order.

**Decision.** Name them, and state here that the names are derived. `ptbxl` takes the twelve
standard leads in their published order. The six `uci_har` names were measured. Every other folder
is numbered by position, which asserts nothing. `folders.py` carries the same statement.

**Consequence.** A reader who needs the physical lead of a `ptbxl` signal has this connector's word
for it. If SLIP cut the leads in another order, the names are wrong and the values are not.

### The class index of the release is the order of the vocabulary — **Handled**

**Problem.** Six folders ship an integer `label` beside the readable `text_label`. That integer is
the class index the release trained against. A set of class names does not carry it.

**Decision.** Order each vocabulary by that integer where the folder ships one, and alphabetically
where it ships a string. The index survives as the position in the list.

**Consequence.** A caller who reproduces a published confusion matrix can recover the index. For the
five folders whose `label` is a string, the release states no order, and the alphabetical one is
this connector's.

### `wesad` ships 13 signals of which 10 are copies — **Handled**

**Problem.** Signals 0, 3, 4, 5, 6, 7, 8, 11 and 12 are byte-identical. So are 1 and 9. So are 2 and
10 *(measured)*. Three distinct signals ship as thirteen.

**Decision.** Build all 13, as the release ships them. Warn once. The build compares the signals of
the first window of the folder rather than trusting this entry.

**Consequence.** Ten of the thirteen signals of every `wesad` record carry no information the other
three do not.

### `AsphaltObstacles` carries the wrong prompt — **Handled**

**Problem.** Its template is `The current air quality index (AQI) is $label. ` over the classes
`speed_bump`, `raised_crosswalk`, `raised_markers` and `vertical_patch`. The `Beijing_AQI` template
is a different string, so this is a mis-copy.

**Decision.** Keep it, verbatim. The build emits no warning. To find it, the connector must hold the
judgement that these four classes are not air-quality indices.

**Consequence.** A prompted model is asked about air quality and must answer with a road surface. A
reader who uses this folder must replace the prompt.

### The `wesad` prompt names the wrong device — **Handled**

**Problem.** The template says `Fibit`, misspelled. WESAD used a RespiBAN chest device and an
Empatica E4 wrist device. No Fitbit was involved.

**Decision.** Keep it, verbatim.

**Consequence.** The prompt states a false fact about the equipment.

### `Beijing_AQI` has five classes, not four — **Handled**

**Problem.** The card says 4. The data holds 5 *(measured)*. The extra class is `4` /
`Unhealthy for sensitive groups`, with 50 rows in train and 16 in test.

**Decision.** Convert all five. Warn once. The build compares the distinct classes of the folder
against the count the card states.

**Consequence.** A model trained against the four classes of the card will meet a fifth.

### The `ptbxl` labels are free-text reports, and one is in German — **Handled**

**Problem.** `text_label` is not a class name. It is a full ECG report, one per class, repeated
verbatim across every row of that class. The report for label 0 is in German. Inside the template of
the folder, the result does not read as a sentence.

**Decision.** Keep the reports as the targets. `target_schema` is `vocabulary-ptbxl`, so the
vocabulary has a name even though its members are paragraphs.

**Consequence.** The task targets of `ptbxl` are paragraphs, not labels, and they are not all in one
language. A classification metric over five distinct strings still works.

### Every folder that can leak subjects across its split, does — **Open**

**Problem.** `wisdm` shares all 46 participants between train and test *(measured)*. That column
holds the person, so nothing is assumed. The other two need a reading first. The `participant_id` of
`sleepEDF` is a recording id. 39 recordings fold onto 20 subjects. A split that is clean by
recording still shares 7 of its 8 test subjects with train. The 1,175 values of `studentlife` are shaped
`<subject>_<month>_<day>_<n>`, and the leading token folds them onto 23 subjects, all of which
appear on both sides. **Both readings are this connector's inference** *(derived, not measured)*.
The release never states what those values identify.

**Decision.** Convert the splits as the release states them. This connector does not re-split a
benchmark. `subject_ids` carries the column the folder ships, whatever that column holds. The build
warns for `wisdm` alone, where the column is the person and the overlap is a set intersection. It
stays silent for `sleepEDF` and `studentlife`, where the raw values never repeat across the split.

**Consequence.** A linear-probing score computed on these splits is inflated by subject leakage.
Grouping on `subject_ids` does not repair it. For `sleepEDF` that groups by recording, and for
`studentlife` by a key that is close to one per row. This entry is **Open**. Nobody has decided whether to offer a
grouped split beside the original. Eight of the eleven folders ship no participant column at all.

### Three folders imply an odd window duration — **Handled**

**Problem.** The card states a rate per folder and no window length. The two multiply out to a clean
duration for eight folders: `sleepEDF` 30 s, `wisdm` 10 s, `uci_har` 4 s, `studentlife` 24 h,
`Beijing_AQI` 12 days. Three are odd: `AsphaltObstacles` 7.36 s, the PPG folders 4.17 s and `wesad`
4.29 s *(measured)*.

**Decision.** Use the rate of the card for every folder, unchanged.

**Consequence.** Three folders carry a wall-clock duration that can be wrong. The alternative was to
claim no rate for them. That substitutes the doubt of the converter for a value the release states,
so the doubt is recorded here instead.

### Row counts are the one thing that matches everywhere — **Handled**

**Problem.** None. Every row count matches the card exactly, for all eleven folders and both splits
*(measured)*.

**Decision.** Recorded here, because the rest of this section can leave the impression that nothing
in the card is reliable.

**Consequence.** None.

## Warnings this build emits

| warning | count *(measured)* | why |
| --- | --- | --- |
| a folder's signals are duplicates | 1 | `wesad` ships 13 signals of which only 3 are distinct |
| a folder holds more classes than the card states | 1 | `Beijing_AQI` has 5, the card says 4 |
| a folder's splits share subjects | 1 | `wisdm` has all 46 of its participants on both sides |

`sleepEDF` and `studentlife` leak subjects too, and the build does not count them. Their
`participant_id` is a recording key rather than a subject, and no key sits on both sides, so the
build cannot see the leak. The leakage entry states it instead.

No warning names the mis-copied `AsphaltObstacles` prompt. The entry for it states why.

## What is not built

The pretraining corpus in the same repository, under `data/`. It has a different schema and a
different dataset id.

The recordings these windows were cut from. This release ships fixed-length windows. It ships no way
back to the recording, the session, or the offset inside it.

A subject-grouped alternative to the shipped splits. The leakage entry is **Open**.

The physical units of the seven folders somebody rescaled. Nobody published the constants that restore
them. The other four folders keep their units. `AsphaltObstacles` keeps its scale but not
its zero point.
