# Sleep-EDF: the inconsistencies of the release, and the decisions of this connector

Sleep-EDF holds nine inconsistencies. Each one forces a design or preprocessing decision. This
document states those decisions.

*(measured)* marks a number that this connector took from the release itself, and not from the
dataset page.

## The source of truth

Two parts of the release sometimes state the same fact differently. This connector follows one
rule:

**This connector replaces no stated fact. Where two sources state the same fact differently, one
source wins for a named reason, and the connector keeps the other beside it.**


| Fact | Source of truth | The other reading |
| --- | --- | --- |
| age, sex | the subject table | the EDF header, kept in a `demographics_note` |
| lights off | the subject table | — |
| rate, gain of a signal | that file's own EDF header | — |
| unit of a signal | this connector's signal table | the EDF dimension, which the build checks |
| what a signal measures | this connector's signal table | kept as a `prefiltering` annotation |
| sleep stage | the hypnogram | — |
| where the session ends | the later of the signals and the scoring | the connector keeps both |


The subject table wins for the facts about a person because it is the registry of the study, and
published work joins against it.

The signal table wins for the unit because the header does not always state one. It is not a free
choice: the build refuses any file whose header declares a dimension that the release does not
declare for that signal. That check reads the dimension string alone. It cannot see a scale, which
EDF states in the physical range, and the rectal temperature section below states where that
matters.

## The tasks this connector builds

The release ships a scoring and two subject tables. It ships no task. What those become is a
decision, and this section states the decisions that this connector takes. A build gives 484 248
tasks over 197 records *(measured)*.


| the question | type | count | scope |
| --- | --- | --- | --- |
| what stage is this 30 s epoch | `ClassificationTask` | 483 419 | one epoch |
| where did the subject sleep | `TemporalLocalizationTask` | 197 | the whole record |
| how old is this subject | `ScalarPredictionTask` | 197 | the whole record |
| what sex is this subject | `ClassificationTask` | 197 | the whole record |
| was this night drug or placebo | `ClassificationTask` | 44 | the whole record |
| where did the lights go out | `TemporalLocalizationTask` | 194 | the whole record |


`Task.scope` is what separates the two kinds. A scope names a region of the recording. No scope
means the question is about the whole record.

### The scoring

**Sleep staging is one task for each 30 s epoch, and not one for each annotation.** The hypnogram
is run-length encoded, so consecutive epochs that share a label collapse into one entry. One entry
can cover hours. The release holds 28 529 scored entries over 483 419 epochs *(measured)*, a mean
of 16.9 for each entry. The longest single entry covers 1351 epochs.

One task for each entry asks 28 529 questions and not 483 419, and each question covers a stretch
from 30 s to 11 hours. That is a coarser problem than the one the field measures, and not a
smaller version of it.

**The expansion is exact, so nothing rounds.** Every onset in the release sits on a 30 s boundary,
and every duration is a whole multiple of 30 s *(measured over all 28 529 entries)*. An entry
divides into a whole number of epochs, so nothing chooses where an epoch starts.

**The scoring does not tile its recording, and unscored time gets no task.** 26 telemetry scorings
begin after their signals do, and one of them begins 750 s in. Two recordings hold a hole in the
middle: `ST7121J0` skips 30 s and `ST7221J0` skips 1950 s *(measured)*. That time is not wake and
it is not a stage. It carries no label, so it carries no question. A consumer that assumes an
unbroken epoch grid will find these gaps.

**The target is the label that the scorer wrote.** `Sleep stage 4` stays `Sleep stage 4`. The
5-class problem that most papers report merges stage 3 with stage 4 into N3, and drops
`Movement time` and `Sleep stage ?`. That merge is a decision for whoever trains, and not for this
connector. The eight labels count as `Sleep stage W` 290 365, `Sleep stage 2` 88 983,
`Sleep stage R` 34 184, `Sleep stage 1` 25 175, `Sleep stage ?` 25 047, `Sleep stage 3` 12 191,
`Sleep stage 4` 7263 and `Movement time` 211 *(measured)*.

**An epoch task names no signal.** Its `scope` covers the epoch and leaves `time_series_ids`
unset. A sleep-stage annotation names the four signals that the technician read, because that is
what the release states. A task states what a model must answer, which is a different thing. If a
task names four signals, a model cannot read the respiration or the temperature signal. That is
a modeling decision, and this connector does not make it for a consumer.

**Where sleep begins and ends is its own task.** A `TemporalLocalizationTask` asks for the region
and not for the label of one. Its mode is sparse, because the interval does not tile the
recording. This is where the `sleep_period` annotation went. It is the answer to a question and
not a fact that the release states, and a task target is where an answer belongs.

**Where the lights went out is a second region question, and three recordings carry none.** The
subject table states a clock time and no date, so the offset onto the timeline wraps forward
across midnight. `ST7021J0`, `ST7022J0` and `ST7162J0` each started seconds after the lights went
out. The wrap puts the moment about a day later, past the end of the session *(measured)*. The
connector keeps the annotation as it derived it, because the release states the clock time. It
omits the task, because nothing can ask a recording for a moment it does not hold, so 194 of the
197 carry one.

### The subject tables

**A fact about the whole recording becomes a task with no scope.** Age, sex and the drug condition
each carry no span as an annotation, and each becomes a whole-record task. These are the questions
that the two studies asked. The cassette study measured the effect of age on sleep, and the
telemetry study measured the effect of temazepam.

**Age is a scalar prediction and not a classification over strings.** A float with the unit `year`
keeps the type that a regression metric needs. A wrong answer of 34 for a subject of 33 then reads
as a one-year error, and not as two unequal strings.

**Sex carries the decoded letter, and never the code of the subject table.** The cassette table
heads its sex column `sex (F=1)`. The telemetry table codes its own sex column in the opposite
way. A raw code on a task merges two opposite facts under one value.

**Only telemetry carries a condition.** The cassette table states none, and a task with no answer
is not a task.

**Provenance does not become a task.** `study`, `night`, `recording_start_local` and
`demographics_note` carry no span either, so the rule that opens this section takes them in as
well. The connector leaves them out. A question about which study a recording came from asks a
model to recover a fact that the dataset states beside it. A span-less annotation becomes a task
when it states something about the subject or the intervention. It becomes no task when it states
where the recording came from.

### How the connector writes them

**The connector registers the label vocabularies, and does not attach them.** Three closed sets
exist: the eight sleep stages, the two sexes and the two conditions. Each becomes one annotation
that no record carries, and `target_schema` on a task names the set that its target draws from. A
name alone tells a consumer nothing about what is in the set.

**The tasks stream, and the connector does not hold them in memory.** `set_task_stream` exists for
a dataset with far more tasks than records, and this release has about 2450 tasks for each
recording. Streamed tasks are trusted and not validated, so each task carries its own
`record_ids`, and none appears in `Record.task_ids`.

**The stream reads no file.** It expands the sleep-stage annotations that the records already
carry. A second read of the hypnograms can let the tasks and the annotations disagree.

**The connector invents no prompt.** The release states no question in words. A consumer who wants
a prompted form composes it from the task type and the vocabulary, and different consumers word it
differently.

## Inconsistencies

### The two studies filtered the same signal differently — **Handled**

**Problem.** EDF states one `prefiltering` string for each signal, and this release uses that field
to say things that no other field says. The cassette recorder rectified its submental EMG and
low-passed the result at 0.7 Hz, so a cassette EMG value is an amplitude envelope. The telemetry
recorder passed the same muscle through 0.03 to 800 Hz and rectified nothing, so a telemetry EMG
value is a potential. The rates follow the filtering: the cassette EMG runs at 1 Hz and the
telemetry EMG at 100 Hz *(measured)*. The EEG and EOG signals differ too, though less deeply:
cassette states `HP:0.5Hz LP:100Hz [enhanced cassette BW]` and telemetry states
`LP:800Hz HP:0.03Hz`, over the same 100 Hz sampling. Each string is constant inside its study, on
all 153 cassette and all 44 telemetry files *(measured)*.

The release card states the cassette processing in its own words, in the
`Sleep Cassette Study and Data` section of <https://physionet.org/content/sleep-edfx/1.0.0/>:

> The submental-EMG signal was electronically highpass filtered, rectified and low-pass filtered
> after which the resulting EMG envelope expressed in uV rms (root-mean-square) was sampled at 1Hz.

The spec type takes its name from that sentence.

**Decision.** The cassette EMG signal gets the spec type `emg_envelope` and the telemetry one
keeps `emg`, because two spec types is the only thing that stops a filter from returning both.
Every record also carries a `prefiltering` annotation, from a signal name to the string that
signal's own header states. Both studies state the same strings on every recording, so the release
stores two of those annotations *(measured)*.

**Consequence.** A consumer who filters on the spec type `emg` gets the 44 telemetry signals and
not the 153 cassette ones, so pooling an envelope with a broadband signal now takes a deliberate
step. `spec_type` is the writer's primary sort key, so the cassette EMG signals sit elsewhere in
the written order than they did. A consumer who wants what the header states about any other
signal reads the annotation and joins it on the signal name.

### The header does not state the unit of every signal — **Handled**

**Problem.** The EDF physical dimension is not a usable unit on three signals. `Temp rectal`
states `DegC` on 85 of the 153 cassette files and nothing at all on the other 68, and the split
runs through the study rather than between the two studies. `Marker` states `ID+M-E` on all 44
telemetry files, which names a marker convention and not a physical dimension. `Resp oro-nasal`
states nothing on any of the 153 files *(measured)*.

The release card says what that marker string means, in the `Sleep Telemetry Study and Data`
section of <https://physionet.org/content/sleep-edfx/1.0.0/>:

> The physical marker dimension ID+M-E relates to the fact that pressing the marker (M) button
> generated two-second deflections from a baseline value that either identifies the telemetry unit
> (ID = 1 or 2 if positive) or marks an error (E) in the telemetry link if negative.

A telemetry unit number is not a physical quantity, so no unit converts it.

**Decision.** The unit comes from this connector's signal table and never from the header. The
build then checks every header against that table and refuses any file that declares a dimension
the release does not declare for that signal, so a copy whose EEG header read `mV` fails the build
instead of shipping amplitudes that are wrong by a factor of 1000.

**Consequence.** All 153 rectal temperature signals are written in degrees Celsius, and the 68
that declare nothing get a unit the release does not state for them. The `ID+M-E` string reaches no
consumer. The rate and the gain still come from each file's own header. The release states a gain
on every file, but a stated gain is not always a right one, and the rectal temperature section
below states where it is not.

### The scoring reaches past the signals — **Handled**

**Problem.** A hypnogram and the PSG it scores are two files, and nothing makes them agree. 155
of the 197 scorings end after their own signals stop *(measured)*. The last entry pads the file
toward a full day, whatever time the recorder stopped.

**Decision.** The connector keeps both readings. It writes the scoring as the file states that
scoring. The record then declares a session span that covers the later of the two readings. To
trim the scoring is a preprocessing decision, and this connector does not make it.

**Consequence.** A sleep stage names the four signals that the technician read. TimeF thus checks
it against the windows of those signals, and not against the declared span. The last entry of
those 155 recordings falls outside, and `add_annotations` warns one time for each. A consumer that
reads the end of a recording finds labeled epochs with no signal beneath them. This consumer must
decide whether to keep them.

### The header and the table disagree on age or sex — **Handled**

**Problem.** The `patient id` field of an EDF header is anonymous but keeps a sex and an age. This
field differs from the subject table on 24 of the 197 recordings *(measured)*. 21 recordings
differ by a year, which is the age at the recording against the age at enrollment. 3 recordings
contradict the table on sex.

**Decision.** The table wins, because it is the registry of the study and published work joins
against it. The record carries a `demographics_note` that gives both readings. Neither the `age`
nor the `sex` annotation changes.

**Consequence.** A consumer who reads `age` and `sex` gets the answer of the table on every
record. The reading of the header survives only in the note, on those 24 recordings and nowhere
else. The age task takes the table value, so a model that predicts age answers for age at
enrollment.

### The physical range differs between recordings — **Handled**

**Problem.** The release states a different physical range in most recordings: 117 distinct
ranges across the 153 cassette files, and one across all 44 telemetry files *(measured)*. One
stored count thus converts to a different value from file to file.

**Decision.** The connector reads the gain from each file. It applies no fixed gain.

**Consequence.** Two recordings that share a stored count do not share a microvolt value. A
consumer that caches a gain across files, or assumes the cassette study is uniform, gets the
wrong amplitude. The connector writes the decoded value and not the stored count, so the values
plane doubles: 4,356,948,980 values are 8,713,897,960 B as int16 and 17,427,795,920 B as float32
before compression, against 8,714,278,888 B of PSG file on disk *(measured)*.

### The rectal temperature range does not decode to a body temperature — **Handled**

**Problem.** `Temp rectal` declares `physical_range = (0, 30)` on 95 of the 153 cassette files. The
other 58 declare a range that reaches a body temperature, 47 of them `(34, 40)` and the rest one of
six wider ranges. Decoded through each file's own gain, the 95 have per-file medians from 3.57 to
14.47 and the 58 from 35.14 to 37.62 *(measured)*. The header dimension does not predict the split:
40 of the 95 declare `DegC`. The 85-stated and 68-unstated split of the section above is about the
unit and this one is about the scale, so neither predicts the other.

**Decision.** The connector applies the range each file declares and corrects nothing, so those 95
signals are written in degrees Celsius that are not a body temperature. Every EDF reader produces
the same values from the same files. To correct here is to invent a calibration that the release
does not state.

**Consequence.** A consumer who pools rectal temperature across the cassette study gets two
populations. Nothing in the artifact carries the declared header range, so the filter has to be on
the value itself. The dimension check cannot catch this: the release writes both `DegC` and nothing
for this signal, so the table accepts both, and the scale lives in the range and not in the
dimension.

### Lights off can fall outside the session — **Handled**

**Problem.** The subject tables state a clock time and no date, so the offset onto the recording
timeline wraps forward across midnight. Three recordings start seconds after the lights went out,
and the wrap puts the moment about a day later. `ST7021J0`, `ST7022J0` and `ST7162J0` each place
it at 23:59:30, past a session that ends around 8 h *(measured)*.

Each of those three lights-off times falls 30 seconds before its own recording starts. A TimeF
span starts at zero and holds no moment before it, thus the derivation carries the offset forward
by one day. The identical 86 370 s on all three is that one wrap, and not three faults in the
release.

**Decision.** The connector attaches the annotation as it derived it. It clips nothing and drops
nothing, because the release states the clock time and this connector does not correct it. Those
three carry no lights-off task, because nothing validates a streamed task. Such a target ships
56 000 s past the end of its own recording with no warning.

**Consequence.** 194 of the 197 records answer the lights-off question, and three do not. A consumer
that counts tasks thus finds fewer tasks than records. All 197 carry the annotation, and on those
three it names a moment that the recording does not contain. A consumer that reads a lights-off
moment must check it against the session span. A moment past that end means the lights were already
off when the recorder started.

### The two subject tables code sex with opposite meanings — **Handled**

**Problem.** The cassette table heads its sex column `sex (F=1)`. The telemetry table codes its
own sex column in the opposite way. A raw code on a record merges two opposite facts under one
value.

**Decision.** The connector decodes each subject table with its own map before it builds an
annotation. A record never carries a raw code.

**Consequence.** `sex` reads `F` or `M` on every record of both studies, and the sex task draws
from a two-member vocabulary. It is not necessary for a consumer to know the origin of a record.

### The connector does not repair a truncated file — **Handled**

**Problem.** The EDF library warns and repairs a record count that the file cannot support. A
short file then looks like a whole one.

**Decision.** The reader catches the warning that the library gives for a short file, and raises
`TimeFFormatError`. A repair with no error hides a changed release.

**Consequence.** A build stops on a truncated file. It writes no short recording that says nothing
about its own length. A consumer never receives a record whose signals end early with no warning.
Whoever runs the build learns that this copy of the release is not the copy that this connector
expects.

## What this connector does not read

The EDF transducer type field stays unread. Over all 197 headers it is a function of the signal
label with no exception: `Ag-AgCl electrodes` for the four biopotential signals,
`Oral-nasal thermistors` for the airflow, `Rectal thermistor` for the temperature, and
`Marker button` for cassette against `MarkerButton` for telemetry. That is 1,291 fields of 80
bytes, 103,280 B in all *(measured)*. A consumer who has the signal labels rebuilds the field
from them.

`RECORDS-v1` stays unread. The release ships that 1,037 B text file to name the 61-recording subset
the sleep-staging literature reports against, the Sleep-EDF-20 and Sleep-EDF-39 set of 39 cassette
and 22 telemetry recordings. No record is marked as a member. The subset is exactly derivable from
what a record already carries: a cassette recording whose subject number is under 20, or a
telemetry recording whose condition is `placebo`. That rule reproduces the file with no recording
missing and none added, checked against all 197 records and `ST-subjects.xls` *(measured)*.
