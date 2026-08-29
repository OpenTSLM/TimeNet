# Sleep-EDF: what the release gets wrong, and what this connector does about it

The warning reaches the person who runs the build. This file reaches the person who reads the
data a year later.

It holds two things: the decisions this connector took where the release left a choice open, and
the places where the release disagrees with itself.

Each inconsistency gives the evidence, the decision, and the state. A number this connector
measured over the release itself is marked *(measured)*, so a reader can tell it from one copied
off the dataset's page.

## The source of truth

Two parts of the release sometimes state the same fact differently. The rule this connector
follows:

**A stated fact is never replaced. Where two sources state the same fact differently, one wins by
a named reason, and the other is kept beside it.**

| Fact | Source of truth | The other reading |
| --- | --- | --- |
| age, sex | the subject table | the EDF header, kept in a `demographics_note` |
| lights off | the subject table | — |
| rate, gain, unit of a channel | that file's own EDF header | — |
| sleep stage | the hypnogram | — |
| where the session ends | the later of the signals and the scoring | neither is dropped |

The subject table wins for the facts about a person because it is the registry of the study, and
published work joins against it.

## The tasks this connector builds

The release ships a scoring and two subject tables. It ships no task. What those become is a
decision, and these are the ones taken. A build gives 484 248 tasks over 197 samples *(measured)*.

| the question | type | count | scope |
| --- | --- | --- | --- |
| what stage is this 30 s epoch | `ClassificationTask` | 483 419 | one epoch |
| where did the subject sleep | `TemporalLocalizationTask` | 197 | the whole sample |
| how old is this subject | `ScalarPredictionTask` | 197 | the whole sample |
| what sex is this subject | `ClassificationTask` | 197 | the whole sample |
| was this night drug or placebo | `ClassificationTask` | 44 | the whole sample |
| where did the lights go out | `TemporalLocalizationTask` | 194 | the whole sample |

`Task.scope` is what separates the two kinds. A scope names a region of the recording. No scope
means the question is about the whole sample.

### The scoring

**Sleep staging is one task for each 30 s epoch, not one for each annotation.** The hypnogram is
run-length encoded: consecutive epochs that share a label collapse into one entry, so one entry
can cover hours. The release holds 28 529 scored entries over 483 419 epochs *(measured)*, a mean
of 16.9 epochs for each entry, and its longest single entry covers 1351 of them. One task for each
entry would ask 28 529 questions rather than 483 419, each covering a stretch from 30 s to 11
hours. That is a coarser problem than the one the field measures, not a smaller version of it.

**The expansion is exact, so nothing rounds.** Every onset in the release sits on a 30 s boundary
and every duration is a whole multiple of 30 s *(measured over all 28 529 entries)*. An entry
divides into a whole number of epochs, so no epoch boundary has to be chosen.

**The scoring does not tile its recording, and unscored time gets no task.** 26 telemetry scorings
begin after their signals do, one of them 750 s in, and two recordings hold a hole in the middle:
`ST7121J0` skips 30 s and `ST7221J0` skips 1950 s *(measured)*. That time is not wake and it is
not a stage. It carries no label, so it carries no question. A consumer that assumes an unbroken
epoch grid will find these gaps.

**The target is the label the scorer wrote.** `Sleep stage 4` stays `Sleep stage 4`. The 5-class
problem most papers report merges stage 3 with stage 4 into N3 and drops `Movement time` and
`Sleep stage ?`. That merge is a decision for whoever trains, not for this connector. Over the
release the eight labels fall as `Sleep stage W` 290 365, `Sleep stage 2` 88 983, `Sleep stage R`
34 184, `Sleep stage 1` 25 175, `Sleep stage ?` 25 047, `Sleep stage 3` 12 191, `Sleep stage 4`
7263 and `Movement time` 211 *(measured)*.

**An epoch task names no channel.** Its `scope` covers the epoch and leaves `time_series_ids`
unset. A sleep-stage annotation names the four channels the technician read, because that is what
the release states. A task states what the model is asked, which is a different thing, and naming
four channels would forbid a model from reading the respiration or temperature channel. That is a
modelling decision this connector does not make on a consumer's behalf.

**Where sleep begins and ends is its own task.** A `TemporalLocalizationTask` asks for the region
rather than the label of one, and its mode is sparse because the interval does not tile the
recording. This is where the `sleep_period` annotation went: it is the answer to a question rather
than a fact the release states, and a task target is where an answer belongs.

**Where the lights went out is a second region question, and three recordings carry none.** The
subject table states a clock time and no date, so the offset onto the timeline wraps forward
across midnight. `ST7021J0`, `ST7022J0` and `ST7162J0` each started seconds after the lights went
out, and the wrap puts the moment about a day later, past the end of the session *(measured)*.
The annotation is kept as it was derived, because the release states it. The task is omitted,
because a recording cannot be asked for a moment it does not hold. 194 of the 197 carry one.

### The subject tables

**A fact about the whole recording becomes a task with no scope.** Age, sex and the drug condition
each carry no span as an annotation, and each becomes a whole-sample task. These are the questions
the two studies were recorded to answer: the cassette study measured the effect of age on sleep,
and the telemetry study measured the effect of temazepam.

**Age is a scalar prediction and not a classification over strings.** A float with the unit `year`
keeps the type a regression metric needs, so a wrong answer of 34 for a subject of 33 reads as a
one-year error rather than as two unequal strings.

**Sex carries the decoded letter, never the sheet's code.** The cassette sheet heads its column
`sex (F=1)` and the telemetry sheet codes the same column the other way round. A raw code on a
task would merge two opposite facts under one value.

**Only telemetry carries a condition.** The cassette sheet states none, and a task whose answer is
absent is not a task.

**Provenance does not become a task.** `study`, `night`, `recording_start_local` and
`demographics_note` carry no span either, so the rule above would sweep them in. They are left out
because asking a model which study a recording came from asks it to recover a fact the dataset
already states beside it. A span-less annotation becomes a task when it states something about the
subject or the intervention, and not when it states where the recording came from.

### How they are written

**The label vocabularies are registered, not attached.** Three closed sets exist: the eight sleep
stages, the two sexes and the two conditions. Each is registered as one annotation that no sample
carries, and `target_schema` on a task names the set its target belongs to. A name alone would not
tell a consumer what is in the set.

**The tasks stream; they are not held in memory.** `set_task_stream` exists for a dataset with far
more tasks than samples, and this release has about 2450 tasks for each recording. Streamed tasks
are trusted rather than validated, so each carries its own `sample_ids`, and none appears in
`Sample.task_ids`.

**The stream reads no file.** It expands the sleep-stage annotations the samples already carry.
Reading the hypnograms a second time would let the tasks and the annotations disagree.

**No prompt is invented.** The release states no question in words. A consumer who wants a prompted
form composes it from the task type and the vocabulary, and different consumers word it
differently.

These decisions are recorded before the tasks are built. A reader who finds no task in a version
of this connector has found a version that predates the work, not a contradiction.

## Inconsistencies

### The scoring reaches past the signals — **Handled**

**Evidence.** A hypnogram and the PSG it scores are two files, and nothing makes them agree. Most
scorings end after their own signals stop, because the last entry pads the file to a full day
whatever time the recorder stopped.

**Decision.** Both readings are kept. The scoring is written as the file states it, and the
sample declares a session span that covers the later of the two. The build warns once for each
recording that overruns. Trimming the scoring back would be a preprocessing decision this
connector does not make.

### The header and the table disagree on age or sex — **Handled**

**Evidence.** The `patient id` field of an EDF header is anonymous but keeps a sex and an age.
For some recordings it differs from the subject table: most by one year, which reads as the age
at the recording against the age at enrolment, and a few contradict each other on sex.

**Decision.** The table wins, for the reason above. The sample carries a `demographics_note` that
gives both readings. Neither the `age` nor the `sex` annotation changes.

### The physical range differs between recordings — **Handled**

**Evidence.** The release states a different physical range in most recordings, so one stored
count converts to a different value from file to file.

**Decision.** The gain is read from the file being read. No fixed gain is applied.

### Lights off can fall outside the session — **Handled for tasks, open for the annotation**

**Evidence.** The sheets state a clock time and no date, so the offset onto the recording
timeline wraps forward across midnight. Three recordings start seconds after the lights went out,
and the wrap puts the moment about a day later: `ST7021J0`, `ST7022J0` and `ST7162J0` each place
it at 23:59:30, past a session that ends around 8 h *(measured)*.

**Decision.** The annotation is attached as it was derived. Nothing is clipped and nothing is
dropped, because the release states the clock time and this connector does not correct it.

The task is a different matter. A streamed task is not validated, so a lights-off question on one
of those three would ship a target 56 000 s past the end of its own recording with no warning.
Those three carry no lights-off task. Whether the annotation should also be omitted, or the
offset placed before the start instead, is still not decided.

### The two sheets code sex with opposite meanings — **Handled**

**Evidence.** One sheet heads its column `sex (F=1)`, and the other codes the same column the
other way round. A raw code carried onto a sample would merge two opposite facts under one id.

**Decision.** Each sheet decodes with its own map before any annotation is built. A sample never
carries a raw code.

### A truncated file is not repaired — **Handled**

**Evidence.** The EDF library warns and repairs a record count that the file cannot support.

**Decision.** The reader compares the file's size against what its header claims and raises
`TimeFFormatError`. A silent repair would hide a changed release.
