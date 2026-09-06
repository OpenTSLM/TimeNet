# What a connector may and may not do to its source

Reference for the `add-dataset-connector` skill. These rules decide the design in phase 1 and the
code in phase 4. They hold for every connector.

**A TimeF dataset is a faithful copy of its source, not a cleaned one.** Convert what the source
states, in the shape it states it. Cleaning, balancing and filtering are decisions for whoever uses
the dataset. A connector must not make them on their behalf.

## The data

**Drop nothing the source states.** A record that serves no task is still evidence. Sleep-EDF scores
`Movement time` and `Sleep stage ?`. Neither names a sleep stage, so neither becomes a classification
target. Both still become annotations, and the scored timeline stays whole.

**Keep every sample the source ships.** The short one, the badly scored one and the outlier all stay.

**If an annotation names a series that is not there, build the series. Do not drop the annotation.**
A span carries `time_series_ids`, and a sample refuses one that resolves to nothing. The source
claims the channel exists, so the sample carries it.

- A thermistor comes loose and writes a flat trace, and the scorer still marks an event on it. Keep
  the flat channel. Its emptiness is a fact about the study.
- Build it from what the source recorded, never from invented values. A series must hold at least one
  value, so where the source gives none at all, keep the reference as an annotation value rather than
  deleting it.

**Keep the source's names, labels and units.** Map them onto a spec; do not rename them. `EEG Fpz-Cz`
stays `EEG Fpz-Cz`. The spec says "EEG, microvolts". The channel keeps the montage the technician
wrote.

**Do not resample, interpolate, or fill a gap.** Every series keeps its own time axis, so a source of
mixed rates needs none of it. Cassette holds EEG at 100 Hz beside a rectal temperature at 1 Hz. Both
stay at the rate they were recorded.

**Keep the source's numbers.** Convert to the unit the file states, with the range that file states.
No normalising, no z-scoring, no rounding. The count 220 is 20.6769 uV in one recording and
22.9538 uV in another; the per-file range is part of the data. Read the gain from the file you are
reading, and never apply a fixed one.

**Add a derived fact, never replace a stated one. Where two sources disagree, keep both.** The EDF
header and the subject table disagree about age or sex for 24 of 197 recordings *(measured)*. The
table wins, because published work joins against it, and the sample carries a note giving both
readings.

**Order and identity come from the source.** Channel order from the header, sample id from the
source's own id, so two builds give the same ids. The sample id is `SC4001E0`, not a counter.

**Where TimeF forces a change, make the smallest one, and say so.** These rules bend for a format
invariant and for nothing else. A hypnogram that scores past the end of its signal is neither clipped
nor dropped: its entries are written as the file states them, and the sample declares a `time_span`
reaching the later of the two ends.

## Read the description, not just the headers

**The dataset's description says which series an annotation was derived from. Read it, and scope the
annotation to those series.** No header states this. Only the prose does, and the standard a study
cites is not a safe guess for what that study did.

Sleep-EDF was scored by Rechtschaffen and Kales, but on the `Fpz-Cz` and `Pz-Oz` EEGs and not on the
`C4-A1` and `C3-A2` that the manual prescribes. A stage annotation therefore names the two channels
the release holds. A connector that assumed the standard montage would scope its annotations to two
series that do not exist.

The description also fixes the epoch length, the rater, and the equipment. Take each from it, and not
from the convention of the field.

**Quote the sentence you relied on in the connector's README**, beside the card's `source_url`, so
the next reader can check it rather than trust it. Where the description and the files disagree, that
is an inconsistency: warn, and give it an entry in the README.

## Errors and warnings

**Do not repair a broken file. Raise.** A silent repair hides a changed release. `edfio` warns and
repairs a truncated record count; the reader compares the file size against the header and raises
`TimeFFormatError` instead.

**An unknown channel name raises `TimeFFormatError`. Do not drop it.** A release that adds a channel
must fail loudly, not convert to a sample that is quietly missing a signal.

**Warn on an inconsistency the source ships. Raise only when an artifact is unreadable.** An
inconsistency is a fact about the study; a corrupt file is not. Neither is repaired in silence. The
header and the table disagree about age or sex for 24 recordings: warn, keep both readings, and
convert. Refusing the release over it would be the larger error.

**Do not warn twice.** TimeF warns for itself where it can. `add_annotation` emits
`SpanOutsideWindowWarning` for every span that leaves its window, so a connector that also warned
about the same overrun doubled the output: 314 warnings for 197 recordings *(measured)*. Dropping the
connector's own warning brought them back to 156 *(measured)*. Before you add a warning, find out
whether the format already gives one.

**Warn one time for each kind, with a count and one example.** 155 of 197 scorings end after their
signals stop *(measured)*, because the last entry pads the file toward a full day. That is one
warning, not 155.

**Warn through a module logger**, `_LOG = logging.getLogger(__name__)`, and not through
`warnings.warn`. A warning raised inside a lazy loader never reaches whoever started the build.

**A span outside its window warns, and there is no way to silence it.** `add_annotation` emits
`SpanOutsideWindowWarning` and keeps the span. The keyword that looks like a mute is not one:
`warn_when_outside=False` restores the raise. Choose between a warning and an error, and plan for the
warnings you will get.

Raise TimeNet's own exceptions from `timenet.errors`, never a raw `ValueError`:
`TimeFValidationError` for a violated invariant or bad caller input, `TimeFFormatError` for a corrupt
or unsupported on-disk artifact.

## Write every inconsistency down

The warning reaches the person running the build. The README reaches the person reading the data a
year later. Both are needed.

`references/readme-template.md` is the format, and
`packages/timenet-connectors/src/timenet_connectors/datasets/physionet/sleep_edfx/README.md` is the
worked example.

## Ids

`Annotation.id`, `Task.id`, `Sample.sample_id` and `TimeSeries.time_series_id` all default to
`new_id()`, a UUIDv7. **Do not pass an `id=` unless something resolves the object by that id.** A
generated id is enough for every object nothing looks up, and an invented one is a string somebody
has to keep true.

**Pass `sample_id`, and build it from one `_ID_PREFIX`.** A sample id is load-bearing twice over: a
dataset keys its samples by it to validate a streamed task, and a reader refers to a sample by it
across builds. A generated one would give two builds of one archive two sets of ids that cannot be
compared. The source's own id is the id — `sleep-edfx-SC4001E0`, not a counter.

**Pass `time_series_id`, built on the sample id.** An annotation's `time_series_ids` resolves against
it, so it has to be stable and predictable.

**Let annotation and task ids default.** The exception is a vocabulary annotation whose id a
`ClassificationTask.target_schema` must equal: build both from one function, so the two ends cannot
drift. A **streamed** task never reaches `Sample.task_ids`, so nothing resolves its id, and it must
not state one at all.

**A test asserting an id is not a read.** If an id turns out to be needless, the assertion goes with
it.

**Qualify a subject id by its study.** Cassette subject 1 and telemetry subject 1 are different
people, so `sleep-cassette-00` and not `00`. An unqualified number leaks one person across two
studies in any subject-grouped split.

**Set `start_time` only when the source states a real instant.** Sleep-EDF states a local wall clock
with no timezone, so it stays unset and the clock time becomes an annotation. Inventing a timezone
would be inventing data. `Sample.start_time` refuses a bare `float`, because seconds and microseconds
are both plausible readings of one.

## Windows, and the order that follows from them

`add_annotation` resolves a span's `time_series_ids` against the sample it is attached to, and
measures the span against the window those series give. **The series must therefore exist before
anything can name them.** The order is forced, not chosen: series, then the sample, then annotations,
then tasks.

```python
sample_id = f"{_ID_PREFIX}-{recording.recording_id}"
sample = dataset.add_sample(
    time_series=series,
    sample_id=sample_id,
    subject_ids=(recording.subject_id,),
    time_span=TimeInterval.micros(0, session_end),
)
sample.add_annotations(sleep_stages)
sample.add_annotations(recording_metadata)
```

**Which window a span is measured against depends on the span:**

- A **scoped** span, one that names `time_series_ids`, is measured against the **intersection** of
  those series' windows: the latest start and the earliest end.
- An **unscoped** span is measured against the sample's `time_span` when the sample declares one, and
  against the **union** of its series windows when it does not.

That one rule explains a build's warning count. A sleep stage names the channels it was scored from,
so it is measured against the signals and warns when the scoring runs past them. The metadata
annotations name no series, so they are measured against the `time_span`, which was built to cover
the overrun.

**Attach annotations in one batch.** `add_annotations` validates the whole batch before it attaches
any of it, so a bad hypnogram fails its sample rather than leaving it half annotated.

**Two things are called metadata, and they are not the same.** Dataset metadata is the card,
`dataset.yaml`, read once and passed to `TimeFDataset(metadata=...)`. Sample metadata is annotations,
attached after `add_sample`.

## Tasks

**A task is not an annotation, and there are usually far more of them.** A hypnogram is run-length
encoded: consecutive epochs carrying the same label collapse into one entry with a long duration. A
task is one question, and an epoch task asks one question of each epoch.

| over the whole Sleep-EDF release | *(measured)* |
| --- | --- |
| sleep-stage annotations | 28 529 |
| 30 s epochs they cover | 483 419 |
| mean epochs for each annotation | 16.9 |
| the longest single entry | 1351 epochs |

**Expand the runs before you count tasks.** A connector that made one task per annotation would ask
28 529 questions instead of 483 419, each covering a stretch from 30 s to 11 hours. That is a coarser
problem than the one the field measures, not a smaller version of it.

**Check that the expansion is exact.** Every Sleep-EDF onset sits on a 30 s boundary and every
duration is a whole multiple of 30 s *(measured over all 28 529 entries)*, so an entry divides into a
whole number of epochs. A release with a ragged onset would force a decision about where an epoch
starts. Find out which case you are in.

**The scoring does not tile the recording. Build a task only where the source states a label.** 26
telemetry scorings begin after their signals do, and two recordings hold a hole in the middle
*(measured)*. Unscored time is not wake and it is not a stage. It has no label, so it gets no task.
Filling it would invent a label the technician never wrote.

**`Task.scope` makes a whole-sample label and a 30 s epoch label one type.** A scope of `None` means
the whole sample; a scope of one interval means that epoch. `ClassificationTask` covers both, and no
second task type is needed.

### The count decides the API

- `add_task` / `add_tasks` materialize the tasks in the dataset and validate them. `add_tasks`
  validates the whole batch before attaching any of it.
- `set_task_stream(task_types, source)` streams them and does **not** validate them the way
  `add_task` does. A dataset with far more tasks than samples cannot hold every task in memory.

Sleep-EDF is about 2450 tasks for each sample *(measured)*, so its epoch tasks must stream. **The
plan states the count, so this is decided before any code is written.** Four rules hold for a streamed
task:

- It must already carry its `sample_ids`. Nothing sets them for you.
- It must reference only registered annotations and samples that exist.
- It does not populate `Sample.task_ids`, so nothing resolves its id and it must not state one.
- `source` must give a **fresh iterator on every call**. The writer calls it more than once, first to
  peek at the ids and then to write, so a one-shot generator writes nothing the second time.
- **The stream reads no file.** It expands annotations the samples already carry. A second read of
  the source can disagree with what was written.

Two more hold for every task, streamed or not:

- **`target` and `target_annotation_ids` are exclusive**, and `add_task` enforces it. `target`
  carries the answer inline; `target_annotation_ids` says the answer *is* those stored annotations.
- **`dataset.register_annotations` exists for annotations that tasks reference and no sample
  carries.** Register the annotation before the stream that names it.
- **A closed set is one annotation whose value is the list**, not one annotation per member. One per
  member states that the values exist without stating that they are the whole set.
