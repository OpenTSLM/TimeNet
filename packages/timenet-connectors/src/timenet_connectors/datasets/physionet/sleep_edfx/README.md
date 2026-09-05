# Sleep-EDF: the inconsistencies of the release, and the decisions of this connector

Sleep-EDF holds six inconsistencies. Each one forces a design or preprocessing decision. This
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
| rate, gain, unit of a signal | that file's own EDF header | — |
| sleep stage | the hypnogram | — |
| where the session ends | the later of the signals and the scoring | the connector keeps both |


The subject table wins for the facts about a person because it is the registry of the study, and
published work joins against it.

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
wrong amplitude.

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
