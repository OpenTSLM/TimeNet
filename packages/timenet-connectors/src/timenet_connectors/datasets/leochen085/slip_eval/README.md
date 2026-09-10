# SLIP Evaluation Sets

The eleven benchmarks SLIP was evaluated on. One record is one window: a fixed-length slice of
somebody else's recording, with one class attached and the split it belongs to. The same repository
also ships the corpus SLIP was pretrained on; that is the `leochen085/slip` connector, not this one.

- **id**: `leochen085/slip-eval`
- **source**: https://huggingface.co/datasets/LeoChen085/SlipDataset
- **licence**: MIT

MIT is what the release states, and it covers what SLIP added: the windowing, the splits and the
packaging. The studies the windows were cut from — PTB-XL, Sleep-EDF, WESAD, UCI-HAR, WISDM,
StudentLife, PPG-BP and the Beijing air-quality release — keep their own terms. Two of them TimeNet
also reads from their own publishers, where the recordings are whole and the licence is the
publisher's.

## The source of truth

| fact | source of truth | the other reading |
| --- | --- | --- |
| how many classes `Beijing_AQI` has | the data: **5** *(measured)* | the card says 4 |
| how many distinct signals `wesad` has | the data: **3** *(measured)* | the card says 13, and the file does ship 13 columns |
| what `AsphaltObstacles` asks | the data: an air-quality prompt over road-surface classes | the card implies a road-surface question |
| the sampling rate of every folder | the card | the data states none; the rate is only in the card's table |
| whether a folder's values are in a physical unit | the values *(measured)* | the release says nothing at all — not the card, not the files |
| how many rows each split holds | both agree, for all 11 folders and both splits *(measured)* | — |

The data wins where the two disagree, because a reader can re-measure it. The rate is the one fact
the data does not state at all, so the card is the only source there and is used as stated — see
"Three folders imply an odd window duration".

## What the description states

> 11 downstream sensor classification datasets used for linear-probing and zero-shot retrieval
> evaluation, covering activity recognition, clinical diagnosis, stress prediction, and urban
> sensing.

Decides: the question asked of a window is which class it is, so every task is a
`ClassificationTask`.

> The evaluation datasets are curated to cover heterogeneous sensor configurations (varying channel
> counts, sampling rates, and sequence lengths) across four distinct application domains.

Decides: the folder fixes a record's shape, not the release. Signal count and window length are
constant inside every folder and differ between them *(measured)*.

> This dataset should **not** be used for clinical decision-making. It is intended for research
> purposes only.

Decides: nothing in the conversion. It is repeated here because five of the eleven folders are
clinical.

The card's per-folder table — sensor, rate and class count — is the only statement of the sampling
rate anywhere in the release, so the conversion leans on it directly.

## What one record holds

One record is one window.

**Signals.** One `TimeSeries` per inner list of the row's `X` column. The count is constant inside
each folder: 1 for `AsphaltObstacles` and the three PPG folders, 2 for `sleepEDF`, 3 for `wisdm`,
6 for `uci_har`, 7 for `Beijing_AQI`, 10 for `studentlife`, 12 for `ptbxl`, 13 for `wesad`
*(measured)*. Every signal in a window has the same length.

**Seven of the eleven folders ship values somebody already rescaled. Four do not.** This is the
most important paragraph in this file, because the release says nothing about it anywhere.

The four that keep their units *(measured)*:

| folder | unit |
| --- | --- |
| `sleepEDF` | microvolts |
| `wisdm` | g |
| `uci_har` | g, all six signals |
| `AsphaltObstacles` | metres per second squared, with each window's mean removed |

How each was read. **`sleepEDF`** runs -211 to 209 with per-channel standard deviations of 21.17
and 12.94, which is what raw Sleep-EDF EEG looks like. **`wisdm`**'s resultant magnitude over its
three axes has a median of 1.0004 — gravity, to four decimal places. **`uci_har`**'s signals 3-5
have a median resultant magnitude of 1.0226 with gravity in them and its signals 0-2 have 0.0628
with it removed; neither triple is a rate of turn, whatever the card says.
**`AsphaltObstacles`** has every window's mean inside [-0.011, 0.146] while its per-window standard
deviation spans 0.20 to 3.08, so a mean was removed and nothing was scaled. Its global range over
both splits is -10.04 to 47.96, which as metres per second squared is a wheel going near freefall
over a bump and 4.9 g at the impact.

The other seven are dimensionless, by three methods that do not agree *(measured)*:

- **A global per-channel z-score** — `Beijing_AQI` and `studentlife`. Their per-channel means and
  standard deviations sit at 0 and 1 across the whole folder, while per-window means scatter from
  -2.53 to 7.82. That scatter is what a global fit leaves behind. 4.6% of `studentlife`'s
  window-channels are completely flat.
- **A per-window z-score** — `ptbxl` exactly, and the PPG folders approximately. All 155,640 of
  `ptbxl`'s window-channels have mean exactly 0.0000 and standard deviation exactly 1.0000. The PPG
  folders are centred per window but their standard deviations run 0.77 to 1.19, so they were
  divided by something close to but not equal to the standard deviation.
- **Min-max into roughly [0, 1]** — `wesad`. Its three distinct channels run [0.000, 1.163],
  [-0.192, 1.167] and [-0.107, 1.072]. Under 0.07% of values fall outside [0, 1], so the constants
  were fitted on a wider set than this split.

For those seven the connector ships the numbers as this release states them, because that is what
this release is. Nobody published the constants that would undo any of it.

`AsphaltObstacles` is the one to read carefully: the **scale** is real metres per second squared and
the **zero point** is not. A value is a deviation from that window's own mean, not an absolute
magnitude.

**The time axis** is a `RegularAxis` at the rate the card gives for that folder. The data states no
rate, so the card is the only source, and it is used as stated even where the resulting duration
looks wrong.

**Annotations.** None are scoped to a signal, and none sit on the record. Every one is registered
once for the whole dataset and referenced by the tasks that need it, because a record merged from
the three PPG folders belongs to all three and can be `train` for one diagnosis and `test` for
another — so neither the folder nor the split is a property of the record.

| key | value | where it came from |
| --- | --- | --- |
| `source_benchmark` | the folder, e.g. `wisdm` | the folder name |
| `split` | `train` or `test` | the file name |
| `vocabulary` | that folder's classes, in the release's own order | every `text_label` it holds, ordered by its `label` column |
| `label` | one class of that folder's vocabulary | the `text_label` column |
| `vocabulary` | that folder's whole set of classes | every `text_label` the folder holds |

**Subjects.** Three folders ship a `participant_id`: `wisdm` (46 participants), `sleepEDF` (a
recording id, 39 recordings over 20 subjects) and `studentlife` (1,175 keys over 23 subjects)
*(measured)*. Those become `subject_ids`. The other eight ship no subject column, so their records
carry none.

**The record id is positional**: `slip-eval-wisdm-train-000000` — one prefix, then the folder, the
split and the row's place in it. No folder ships a per-window id. A record merged from the three PPG
folders is named for the row that first stated it, so it carries the folder and split of that row
rather than a counter. Two builds of the same release give the same ids; a new release invalidates
them.

**A subject id is qualified by its folder** — `wisdm-1633`, not `1633`. The three folders that ship
a participant column number it independently, and a bare value would silently merge two of them in
any grouped split. Only `wisdm` numbers people: in `sleepEDF` the value names one recording of one
night, and in `studentlife` it is close to one value per row. Read the leakage entry below before
grouping on these.

## The tasks this connector builds

| the question | type | count | scope |
| --- | --- | --- | --- |
| which class is this window | `ClassificationTask` | **92,415** *(measured)* | the whole record |

The answer is stored by reference: `target_annotation_ids` names the annotation holding that class,
so `ptbxl`'s five paragraph-long reports are stored once rather than 12,970 times. `target` itself
is unset, which is what "by reference" means.

`target_schema` is the id of the annotation holding that folder's whole set of classes —
`vocabulary-sleepEDF`, `vocabulary-ptbxl` — so the two ends are built from one function and cannot
drift, and `sleepEDF`'s five classes and `ptbxl`'s five classes stay two vocabularies rather than
one.
`prompt` is the folder's own template, verbatim, including the two that are wrong.

The vocabularies are 2 to 18 values per folder. There are 92,415 tasks over 91,094 records — the
three PPG folders share their windows, so 1,971 of those tasks sit on 650 records.

## Inconsistencies and decisions

### `PPG_CVA`, `PPG_DM` and `PPG_HTN` are one dataset labelled three ways — **Handled**

**Problem.** All three folders hold the same 650 distinct windows. Every pairwise intersection is
650 *(measured)*. They differ only in the diagnosis attached: stroke, diabetes, hypertension. The
card lists them as three datasets of 657 rows each. 657 is exactly PPG-BP's 219 subjects × 3
recordings.

**Decision.** Build 650 records, each carrying three classification tasks, and carry the folder and
the split on the task rather than on the record — a merged window can be `train` for one diagnosis
and `test` for another, so neither is a property of the record.

**Consequence.** The windows are stored once instead of three times, and one record can be asked all
three questions — which is what the data supports. Anyone reproducing a published per-folder number
can still select by the annotation. 650 records carry 1,971 tasks *(measured)*: 643 of them carry
three, and 7 carry six, because of the duplicates in the next entry.

### Each PPG folder ships 657 rows holding 650 distinct windows — **Handled**

**Problem.** Seven windows appear twice within each PPG folder *(measured: after the merge, 643
records carry three tasks and 7 carry six)*. One of the duplicated pairs straddles that folder's own
train and test split.

**Decision.** Keep both occurrences as tasks on the one record. The task id names the folder, the
split and the row, so the two are distinguishable and traceable back to their rows.

**Consequence.** Seven of the 650 PPG records answer the same question twice. Where the duplicate
straddles the split, the same window is both trained on and tested on.

### `uci_har` has no gyroscope, and the card says it has one — **Handled**

**Problem.** The card lists `uci_har` as "Accelerometer + Gyroscope (6-ch)". Measured over 400
windows *(measured)*, the resultant magnitude of signals 0-2 is 0.0628 and of signals 3-5 is 1.0226.
Those are body acceleration with gravity removed and total acceleration with it kept — the two
accelerometer triples UCI-HAR publishes. Nothing in the folder is a rate of turn.

**Decision.** All six signals are acceleration in g. The connector defines no angular-rate spec,
because nothing in this release needs one.

**Consequence.** Anyone expecting three gyroscope channels here will not find them. Had the card
been believed, three of every `uci_har` record's signals would have shipped in radians per second.

### The signal names are this connector's, not the release's — **Handled**

**Problem.** No file in the release names a signal, and the card's table gives only the sensor kind,
the rate and the class count. Twelve unnamed ECG signals and six unnamed accelerometer signals mean
nothing without an order.

**Decision.** Name them, and say here that the names are derived. `ptbxl` takes the twelve standard
leads in PTB-XL's published order. `uci_har`'s six were measured, as the entry above describes.
Every other folder's names are numbered by position, which asserts nothing. `folders.py` carries the
same statement.

**Consequence.** A reader who needs to know which physical lead a `ptbxl` signal is has this
connector's word and PTB-XL's convention, not the release's. If SLIP cut the leads in another order,
the names are wrong and the values are not.

### The release's class index is the vocabulary's order — **Handled**

**Problem.** Six folders ship an integer `label` beside the readable `text_label`. That integer is
the class index the release trained against, and a set of class names does not carry it.

**Decision.** Order each folder's vocabulary annotation by that integer where the folder ships one,
and alphabetically where it ships a string. So the index survives as the position in the list:
`sleepEDF` reads wakefulness, light transitional, stable light, deep slow wave, REM — the release's
order, not an alphabetical one.

**Consequence.** A caller reproducing a published confusion matrix can recover the index. For the
five folders whose `label` is a string, the release states no order and the alphabetical one is this
connector's.

### `wesad` ships 13 signals of which 10 are copies — **Handled**

**Problem.** Signals 0, 3, 4, 5, 6, 7, 8, 11 and 12 are byte-identical to one another; so are 1 and
9; so are 2 and 10 *(measured)*. Three distinct signals are shipped as thirteen. The card says 13
channels, and WESAD itself does record that many.

**Decision.** Build all 13, as the release ships them. Warn once, having compared the signals of the
folder's first window rather than trusting this entry.

**Consequence.** Ten of every `wesad` record's thirteen signals carry no information the other three
do not. Anyone using this folder should use three.

### `AsphaltObstacles` carries the wrong prompt — **Handled**

**Problem.** Its prompt template is `The current air quality index (AQI) is $label. ` over the
classes `speed_bump`, `raised_crosswalk`, `raised_markers` and `vertical_patch`. `Beijing_AQI`'s own
template is a different string, so this is a mis-copy and not a template the two share.

**Decision.** Keep it, verbatim. No warning: detecting it would mean hard-coding the judgement that
these four classes are not air-quality indices, and a connector that encodes that judgement will be
wrong about the next release.

**Consequence.** A prompted model asked this task is asked about air quality and expected to answer
with a road surface. Anyone using this folder should replace the prompt.

### `wesad`'s prompt names the wrong device — **Handled**

**Problem.** The template says `Fibit`, misspelled. WESAD used a RespiBAN chest device and an
Empatica E4 wrist device; no Fitbit was involved.

**Decision.** Keep it, verbatim.

**Consequence.** The prompt states a false fact about the equipment.

### `Beijing_AQI` has five classes, not four — **Handled**

**Problem.** The card says 4. The data holds 5 *(measured)*. The extra is `4` /
`Unhealthy for sensitive groups`, 50 in train and 16 in test.

**Decision.** Convert all five. Warn once, having compared the folder's distinct classes against the
count the card states.

**Consequence.** A model trained against the card's four classes will meet a fifth.

### `ptbxl`'s labels are free-text reports, one of them in German — **Handled**

**Problem.** `text_label` is not a class name but a full free-text ECG report, one per class, reused
verbatim across every row of that class. Label 0's report is in German and the other four are in
English. Substituted into the folder's own prompt template the result does not parse as a sentence.

**Decision.** Keep the reports as the targets. `target_schema` is `vocabulary-ptbxl`, so the
vocabulary is named even though its members are paragraphs.

**Consequence.** `ptbxl`'s task targets are paragraphs, not labels, and they are not all in one
language. A classification metric over five distinct strings still works.

### Every folder that can leak subjects across its split, does — **Open**

**Problem.** `wisdm` shares all 46 participants between train and test *(measured: that column is
the person, so nothing is assumed)*. The other two need a reading before the question can be asked.
`sleepEDF`'s `participant_id` is a recording id: read as Sleep-EDF's own published
`SC4<subject><night>E0` file convention, 39 recordings fold onto 20 subjects, and a split that is
clean by recording then shares 7 of its 8 test subjects with train. `studentlife`'s 1,175 values are
shaped `<subject>_<month>_<day>_<n>`, and taking the leading token as the person folds them onto 23
subjects, all of which appear on both sides. **Both readings are this connector's inference**
*(derived, not measured)*. The release never says what those values identify — not in a file, not in
the card — so 7 of 8 and 23 of 23 are not facts of the kind the row counts are, which anyone
re-measures without assuming anything. Confirming them means going back to Sleep-EDF and StudentLife
themselves. The other eight folders ship no participant column, so nothing can be read there at all
— but the PPG folders' 657 rows is exactly 219 subjects × 3 recordings ungrouped, and `PPG_HTN` has
one byte-identical window in both of its own splits.

**Decision.** Convert the splits as the release states them. The connector does not re-split a
benchmark; it reports what the benchmark is. `subject_ids` carries the column the folder ships,
whatever that column holds. The build warns for `wisdm`, where the column is the person and the
overlap is therefore a set intersection; it stays silent for `sleepEDF` and `studentlife`, where the
column is a recording and a session key, the raw values never repeat across the split, and showing
the leak means decoding the id. This entry carries it instead.

**Consequence.** A linear-probing score computed on these splits as shipped is inflated by subject
leakage, and grouping on `subject_ids` does not repair it: for `sleepEDF` that groups by recording
and still shares 7 of the 8 test subjects, and for `studentlife` it groups by a key that is close to
one per row. This entry is **Open** because nobody has decided whether a grouped split should be
offered beside the original, because deriving a person from those keys is a judgement the release
does not state, and because eight of the eleven folders cannot be checked at all.

### Three folders imply an odd window duration — **Handled**

**Problem.** The card states a rate per folder and no window length. Multiplying out gives a clean
duration for eight folders — `sleepEDF` exactly 30 s, `wisdm` 10 s, `uci_har` 4 s, `studentlife`
24 h, `Beijing_AQI` 12 days — and an odd one for three: `AsphaltObstacles` 7.36 s, the PPG folders
4.17 s, `wesad` 4.29 s *(measured)*. `wesad`'s 3000 samples would be exactly 30 s at 100 Hz rather
than the 700 Hz the card claims. `ptbxl`'s 5.12 s is fine — 512 is a power-of-two crop of PTB-XL's
1000-sample records.

**Decision.** Use the card's rate for every folder, unchanged.

**Consequence.** Three folders carry a wall-clock duration that may be wrong. The alternative was to
claim no rate for them, which would have substituted the converter's doubt for a value the release
states. The doubt is recorded here instead.

### Row counts are the one thing that matches everywhere — **Handled**

**Problem.** None. Every row count matches the card exactly, for all eleven folders and both splits
*(measured)*.

**Decision.** Recorded because the rest of this section may leave the impression that nothing in the
card is reliable.

**Consequence.** None.

## Warnings this build emits

| warning | count *(measured)* | why |
| --- | --- | --- |
| a folder's signals are duplicates | 1 | `wesad` ships 13 signals of which only 3 are distinct |
| a folder holds more classes than the card states | 1 | `Beijing_AQI` has 5, the card says 4 |
| a folder's splits share subjects | 1 | `wisdm` has all 46 of its participants on both sides |

`sleepEDF` and `studentlife` leak subjects too and are not counted here. Their `participant_id` is a
recording key rather than a subject, and no key sits on both sides, so the build cannot see it. The
leakage entry above carries that instead. No warning names the mis-copied `AsphaltObstacles` prompt
either; the entry above says why.

## What is not built

The pretraining corpus in the same repository, under `data/`. It is a different dataset with a
different schema, and it is the `leochen085/slip` connector.

The recordings these windows were cut from. This release ships fixed-length windows and no way back
to the recording, the session or the offset within it. For Sleep-EDF and PTB-XL, TimeNet reads the
whole recordings from their own publishers instead — `physionet/sleep-edfx` and
`physionet/ecg-qa-cot` — and those are the connectors to use when the recording matters.

A subject-grouped alternative to the shipped splits. See the leakage entry above; that is **Open**.

The physical units of the seven folders somebody rescaled. The constants that would restore them are
not published. The other four keep their units, and `AsphaltObstacles` keeps its scale but not its
zero point.
