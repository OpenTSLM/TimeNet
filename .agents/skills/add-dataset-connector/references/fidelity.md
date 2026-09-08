# What a connector may and may not do to its source

Reference for the `add-dataset-connector` skill. These rules decide the design in phase 1 and the
code in phase 4. They hold for every connector.

**How to read this.** Every rule is general and applies to any source. The examples are marked
*Sleep-EDF:* and come from `physionet/sleep_edfx`, the connector these rules were first written
against. An example illustrates a rule. It never narrows it. A number marked *(measured)* was
counted over that one release and describes it alone — your dataset's numbers will differ, and the
rule will not. When a rule and an example seem to disagree, the rule wins.

**A TimeF dataset is a faithful copy of its source, not a cleaned one.** Convert what the source
states, in the shape it states it. Cleaning, balancing and filtering are decisions for whoever uses
the dataset. A connector must not make them on their behalf.

## Contents

- [The data](#the-data)
- [Read the description, not just the headers](#read-the-description-not-just-the-headers)
- [Errors and warnings](#errors-and-warnings)
- [Write every inconsistency down](#write-every-inconsistency-down)
- [Ids](#ids)
- [Windows, and the order that follows from them](#windows-and-the-order-that-follows-from-them)
- [Tasks](#tasks)

## The data

**Drop nothing the source states.** A record that serves no task is still evidence. A label outside
the set a task predicts is still a label, and the timeline it sits on stays whole.

- *Sleep-EDF:* the scoring uses `Movement time` and `Sleep stage ?`. Neither names a sleep stage, so
  neither becomes a classification target. Both still become annotations.

**Keep every record the source ships.** The short one, the badly scored one and the outlier all stay.

**If an annotation names a series that is not there, build the series. Do not drop the annotation.**
A span carries `time_series_ids`, and a record refuses one that resolves to nothing. The source
claims the signal exists, so the record carries it.

- Build it from what the source recorded, never from invented values. A series must hold at least one
  value. Where the source gives none at all, keep the reference as an annotation value rather than
  deleting it.
- *Any recording modality:* a sensor comes loose and writes a flat trace, and the annotator still
  marks an event on it. Keep the flat signal. Its emptiness is a fact about the study.

**Keep the source's names, labels and units.** Map them onto a spec. Do not rename them. The spec
says what kind of thing a signal measures. The signal keeps the name the instrument or the
technician wrote, no matter how irregular it is.

- *Sleep-EDF:* `EEG Fpz-Cz` stays `EEG Fpz-Cz`, and the spec says "EEG, microvolts". The signal
  keeps its montage.

**Do not resample, interpolate, or fill a gap.** Every series keeps its own time axis, so a source of
mixed rates needs none of it.

- *Sleep-EDF:* one recording holds EEG at 100 Hz beside a rectal temperature at 1 Hz. Both stay at
  the rate they were recorded.

**Keep the source's numbers.** Convert to the unit the file states, with the range that file states.
No normalizing, no z-scoring, no rounding. Read the scaling from the file you are reading, and never
apply a fixed one. A per-file range is part of the data, not a detail to average away.

- *Sleep-EDF:* the same stored count of 220 is 20.6769 uV in one recording and 22.9538 uV in
  another.

**Fidelity is to the release you convert, not to the study upstream of it.** A derived release —
windows cut out of somebody else's recordings, values another group already z-scored — is converted
as the derived thing it is. Those numbers are what *this* source states, so they ship unchanged even
though they are not physical units. The README says what the numbers are and what they were derived
from. Rescaling them back toward the original instrument invents data twice over.

**Add a derived fact, never replace a stated one. Where two sources disagree, keep both.** Say in the
README which one the connector treats as authoritative and why. Carry the other as a note.

- *Sleep-EDF:* the file header and the subject table disagree about age or sex for 24 of 197
  recordings *(measured)*. The table wins, because published work joins against it, and the record
  carries both readings.

**A stated value is used even when it looks wrong.** The rule before this one covers two parts of the
source disagreeing with each other. This rule covers the converter disagreeing with the source, and
it is the harder case, because the arithmetic feels like proof. Write the doubt into the README with
the arithmetic that raised it. Never apply it to the data. If the value is wrong, the release is
wrong, and the README is where a reader finds that out.

- *Any release:* a card states 700 Hz for a 3000-sample window, which is 4.29 s, where 100 Hz gives
  exactly 30 s. Build the axis at the stated 700 Hz. Give the README an entry holding both numbers.

**Order and identity come from the source, where the source states one.** Signal order comes from the
header. A record id comes from the source's own id, under one prefix, so two builds of one release
give the same ids.

**Where the source states no id at all, use a position that is stable under a re-read.** The rule's
content is reproducibility, and a position gives that as long as the release does not change. Build
it from what the release itself is divided into — the shard and the row inside it. Never build it
from a counter that runs across files. A counter changes when a file is added, renamed or read in a
different order. Then say in the README that the id is positional and that a re-release invalidates
it.

- `chengsenwang/tsqa` is the worked example of the fallback: its corpus states no id, so
  `connector.py:53` builds `record_id=f"row-{index}"` and `:49` the matching
  `time_series_id=f"row-{index}-c{signal}"`.

**Where TimeF forces a change, make the smallest one, and say so.** These rules bend for a format
invariant and for nothing else.

- *Sleep-EDF:* a scoring that runs past the end of its signal is neither clipped nor dropped. Its
  entries are written as the file states them, and the record declares a `time_span` reaching the
  later of the two ends.

## Read the description, not just the headers

**The dataset's description says which series an annotation was derived from. Read it, and scope the
annotation to those series.** No header states this. Only the prose does, and the standard a study
cites is not a safe guess for what that study did.

A study cites a standard and then departs from it, and only the prose says where. Scoping an
annotation to the series the standard prescribes, rather than the ones the study used, points it at
series the release does not contain. The record then refuses it.

The description also fixes the window length, the rater, and the equipment. Take each from it, and
not from the convention of the field.

- *Sleep-EDF:* scored by Rechtschaffen and Kales, but on the `Fpz-Cz` and `Pz-Oz` EEGs and not the
  `C4-A1` and `C3-A2` the manual prescribes. A stage annotation names the two signals the release
  actually holds.

**Quote the sentence you relied on in the connector's README**, beside the card's `source_url`. The
next reader can then check it rather than trust it. Where the description and the files disagree,
that is an inconsistency: warn, and give it an entry in the README.

## Errors and warnings

**Do not repair a broken file. Raise.** A silent repair hides a changed release. Parsing libraries
often repair by default and only warn. Check what yours does, and turn its repair into a
`TimeFFormatError`.

- *Sleep-EDF:* `edfio` warns and repairs a truncated record count. The reader compares the file size
  against the header and raises instead.

**An unknown signal name raises `TimeFFormatError`. Do not drop it.** A release that adds a signal
must fail loudly, not convert to a record that is quietly missing a signal.

**Warn on an inconsistency the source ships. Raise only when an artifact is unreadable.** An
inconsistency is a fact about the study. A corrupt file is not. Neither is repaired in silence.
Where a header and a table disagree for a handful of records, warn, keep both readings, and convert.
Refusing the whole release over it is the larger error.

**Do not warn twice.** TimeF warns for itself where it can. Before you add a warning, find out
whether the format already gives one. `add_annotation` emits `SpanOutsideWindowWarning` for every
span that leaves its window, and a connector warning about the same thing doubles the output.

- *Sleep-EDF:* the duplicate warning produced 314 lines for 197 recordings *(measured)*. Without the
  connector's own warning, the count came back to 156.

**Warn one time for each kind, with a count and one example.** A property shared by most of a
release is one fact about the release, not one fact per record.

- *Sleep-EDF:* 155 of 197 scorings end after their signals stop *(measured)*, because the last entry
  pads the file toward a full day. That is one warning, not 155.

**Warn through a module logger**, `_LOG = logging.getLogger(__name__)`, and not through
`warnings.warn`. A warning raised inside a lazy loader never reaches whoever started the build.

**A span outside its window warns, and there is no way to silence it.** `add_annotation` emits
`SpanOutsideWindowWarning` and keeps the span. The keyword that looks like a mute is not one:
`warn_when_outside=False` restores the raise. Choose between a warning and an error. Plan for the
warnings you will get.

Raise TimeNet's own exceptions from `timenet.errors`, never a raw `ValueError`:
`TimeFValidationError` for a violated invariant or bad caller input, `TimeFFormatError` for a corrupt
or unsupported on-disk artifact.

## Write every inconsistency down

The warning reaches the person who runs the build. The README reaches the person who reads the data
a year later. Both are needed.

`references/readme-template.md` is the format, and
`packages/timenet-connectors/src/timenet_connectors/datasets/physionet/sleep_edfx/README.md` is the
worked example.

## Ids

`Annotation.id`, `Task.id`, `Record.record_id` and `TimeSeries.time_series_id` all default to
`new_id()`, a UUIDv7. **Unless something resolves the object by that id, do not pass an `id=`.** A
generated id is enough for every object nothing looks up, and an invented one is a string somebody
has to keep true.

**Pass `record_id`, and build it from one `_ID_PREFIX`.** A record id is load-bearing twice over: a
dataset keys its records by it to validate a streamed task, and a reader refers to a record by it
across builds. A generated one gives two builds of one archive two sets of ids that cannot be
compared. The source's own id is the id, under one prefix. Where the source states none, use a
position that is stable under a re-read — see **The data** above. Never use a counter that runs
across files.

**Pass `time_series_id`, built on the record id.** An annotation's `time_series_ids` resolves against
it, so it must be stable and predictable.

**Let annotation and task ids default.** The exception is a vocabulary annotation whose id a
`ClassificationTask.target_schema` must equal: build both from one function, so the two ends cannot
drift. A **streamed** task never reaches `Record.task_ids`, so nothing resolves its id, and it must
not state one at all.

**A test that asserts an id is not a read.** If an id turns out to be needless, the assertion goes
with it.

**Qualify a subject id by whatever the release numbers separately.** Where a release is split into
parts that each number their subjects from one, a bare number collides. The collision silently
merges two people in any subject-grouped split.

- *Sleep-EDF:* the two studies number subjects independently, so a subject id is
  `sleep-cassette-00`, not `00`.

**Set `start_time` only when the source states a real instant.** A local wall clock with no zone is
not an instant. Leave the field unset. Carry the stated clock time as an annotation. Inventing a
timezone is inventing data. `Record.start_time` refuses a bare `float`, because seconds and
microseconds are both plausible readings of one, and refuses a naive `datetime`. It takes a tz-aware
`datetime` or whole Unix microseconds.

## Windows, and the order that follows from them

`add_annotation` resolves a span's `time_series_ids` against the record it is attached to, and
measures the span against the window those series give. **As a result, the series must exist before
anything can name them.** The order is forced, not chosen: series, then the record, then annotations,
then tasks.

```python
record_id = f"{_ID_PREFIX}-{recording.recording_id}"
record = dataset.add_record(
    time_series=series,
    record_id=record_id,
    subject_ids=(recording.subject_id,),
    time_span=TimeInterval.micros(0, session_end),
)
record.add_annotations(sleep_stages)
record.add_annotations(recording_metadata)
```

**Which window a span is measured against depends on the span:**

- A **scoped** span, one that names `time_series_ids`, is measured against the **intersection** of
  those series' windows: the latest start and the earliest end.
- An **unscoped** span is measured against the record's `time_span` when the record declares one.
  When the record declares none, the span is measured against the **union** of its series windows.
  The union is the case that catches people out. The windows are merged, and a span that lands in a
  **gap** between two of them is outside, even though it sits between the first start and the last
  end. When the session spans a gap, declare a `time_span`.

That one rule explains a build's warning count. A sleep stage names the signals it was scored from,
so it is measured against the signals and warns when the scoring runs past them. The metadata
annotations name no series, so they are measured against the `time_span`, which was built to cover
the overrun.

**Attach annotations in one batch.** `add_annotations` validates the whole batch before it attaches
any of it, so one bad annotation fails its record rather than leaving it half annotated.

**Two things are called metadata, and they are not the same.** Dataset metadata is the card,
`dataset.yaml`, read once and passed to `TimeFDataset(metadata=...)`. Record metadata is annotations,
attached after `add_record`.

## Tasks

**A task is one question asked of a record, together with the answer the source states.** A
per-window task asks that question of each window. An annotation is not a task. An annotation states
a fact about the timeline, and a release usually holds far fewer of them.

Many sources run-length encode their labels. Consecutive windows that carry the same label collapse
into one entry with a long duration, and the file states that entry one time. Count the entries and
the windows before you design.

**Expand the runs before you count tasks.** One task per stored entry asks a different, coarser
question than one task per window. Each entry covers a stretch that can run from one window to
hours. That is not a smaller version of the problem the field measures. It is a different problem.

- *Sleep-EDF:* 28 529 stored entries cover 483 419 windows of 30 s *(measured)*, a mean of 16.9
  windows per entry, and the longest single entry covers 1351.

**Check that the expansion is exact.** An entry divides into a whole number of windows only when two
things hold: every onset sits on a window boundary, and every duration is a whole multiple of one.
Where it does not, you must decide where a window starts. That decision belongs in the README.
Measure which case you are in before you assume.

- *Sleep-EDF:* every onset and duration is a whole multiple of 30 s *(measured over all 28 529
  entries)*, so nothing rounds.

**Labels rarely tile the recording. Build a task only where the source states one.** Unlabeled time
is not the negative class and it is not the default label. It has no label, so it gets no task.
Filling it invents a label nobody wrote.

- *Sleep-EDF:* 26 scorings begin after their signals do, and two recordings hold a hole in the
  middle *(measured)*.

**`Task.scope` makes a whole-record label and a per-window label one type.** A scope of `None` means
the whole record. A scope of one interval means that window. `ClassificationTask` covers both, and no
second task type is needed.

### The count decides the API

- `add_task` / `add_tasks` materialize the tasks in the dataset and validate them. `add_tasks`
  validates the whole batch before attaching any of it.
- `set_task_stream(task_types, source)` streams them and does **not** validate them the way
  `add_task` does. A dataset with far more tasks than records cannot hold every task in memory.

**Stream unless you have a reason not to.** `add_task` rebuilds the set of every registered task id
on each call (`dataset/dataset.py:354`), so adding tasks one at a time is quadratic in the count. One
build of 92 415 tasks ran past twenty minutes and was killed. The same build through
`set_task_stream` took **5.6 seconds** *(measured)*. `add_tasks` is not the escape, because it
attaches one batch to one set of records. A connector with one task per record still makes one call
per record. Use `add_task` only where the tasks are few and you want the cross-task validation it
gives.

**The plan states the count, so this is decided before any code is written.** Where tasks outnumber
records by orders of magnitude, they must stream. Sleep-EDF runs to about 2450 tasks per record
*(measured)*. Five rules hold for a streamed task:

- It must already carry its `record_ids`. Nothing sets them for you.
- It must reference only registered annotations and records that exist.
- It does not populate `Record.task_ids`, so nothing resolves its id and it must not state one.
- `source` must give a **fresh iterator on every call**. It is read after `convert` returns, and
  more than once, so a one-shot generator yields nothing the second time and writes no tasks.
- **The stream holds nothing, and what it reads it can read again.** It can re-open a file. A
  release with more tasks than fit in memory leaves no other way, and
  `physionet/ecg_qa_cot/connector.py:297` streams straight out of its CSVs. What it must not do is
  keep the tasks alive between calls, or read something that can answer differently the second time.

Three more hold for every task, streamed or not:

- **A task needs exactly one of `target` and `target_annotation_ids`.** `target` carries the answer
  inline. `target_annotation_ids` says the answer *is* those stored annotations. `add_task` raises
  `TimeFValidationError` on a task that sets both, and on one that sets neither. The exception is a
  task whose answer is a produced series: it sets `answer_is_record` and names its answer by record
  id, so it sets neither and the check skips it (`dataset/dataset.py:530`).
- **`dataset.register_annotations` exists for annotations that tasks reference and no record
  carries.** Register the annotation before the stream that names it.
- **A closed set is one annotation whose value is the list**, not one annotation per member. One per
  member states that the values exist without stating that they are the whole set.
