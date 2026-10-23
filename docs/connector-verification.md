---
icon: lucide/badge-check
description: "Check that a connector put its source into TimeF correctly, and record how."
tags:
  - guide
  - connectors
---

# Verifying a connector

How do we know a connector put the data into TimeF correctly? Every connector answers that in a
`VERIFICATION.md` beside its `dataset.yaml`. This page explains what goes in it:

- **[Families of connector errors](#families-of-connector-errors):** what can go wrong, and how to
  catch each kind of error.
- **[Verification checklist](#verification-checklist):** every error, in the order to check them.
- **[The verification record](#the-verification-record):** how to write `VERIFICATION.md` from the
  [template](connector-verification-template.md), and what the test checks.
- **[Tooling to build](#tooling-to-build):** shared checks that would make verification routine.

## Families of connector errors

They come in the same order, and under the same names, as the steps of the
[verification checklist](#verification-checklist).

| Family | Typical errors | Silent? |
| --- | --- | --- |
| Source | Wrong release or version, unpinned download, partial or corrupt archive, wrong subset of files, a misread of what the source contains | Often |
| Coverage | Records, subjects or channels dropped or duplicated, wrong filters, counts that don't match the paper, source information left out without a note, values the connector made up | Partly: counts show it if you compare them with something |
| Read path | Lazy loader reads the wrong bytes, precision lost on a dtype change, a build that isn't deterministic | Sometimes |
| Channel meaning | Swapped or mislabeled channels, wrong units or scale, wrong sign convention, raw ADC values without gain and baseline, missing-value markers stored as values, wrong codebook | Yes |
| Time | Wrong time zone or start time, wrong sample rate, resampling artifacts, gaps filled or dropped silently, sources out of sync, labels misaligned with the signal | Yes |
| Identity and joins | Labels, captions or metadata attached to the wrong record: off-by-one ids, relying on file order, colliding ids, a join on the wrong key | **Yes**, and the worst: every shape and count stays the same |
| Labels and targets | Wrong mapping from class code to name, wrong vocabulary order, wrong granularity, target derived wrongly from the source | Yes |
| Splits | Not the authors' splits, the same subject or overlapping windows in train and test, split sizes that don't match the paper | Yes, unless compared with the paper |
| Task construction | Wrong prompt, series placed wrongly in the prompt, wrong task type, a task its inputs can't answer, a field in the wrong role, an input, id or metadata field that reveals the target | Mostly |
| Card and provenance | Wrong license, domain or description, a README that claims what the data doesn't support | Yes, unless a reviewer reads it |

Source, coverage and split errors change counts or shapes, so comparing a stats snapshot with the
paper catches them cheaply. Channel-meaning, time, identity and label errors keep every shape and
count: a swap, a shift or an off-by-one join passes every structural test. Only a comparison with
an independent copy of the data, a physical check, or a join check catches them. Task construction
needs a person to read rendered tasks, and the card needs a reviewer to read it.

The rest of this section goes through each family: what can go wrong, and how to catch it.
The [tooling](#tooling-to-build) these checks call for comes at the end.

### Source

What can go wrong: the connector reads something other than what the paper used. It might be a
newer release, a mirror that changed, a truncated download, or the wrong files from the archive.

How to catch it:

- **Pin every input.** Use an immutable revision where the host has one, such as a Hugging Face
  commit or a versioned release. Otherwise pin a sha256 of the download. The build fails if the
  bytes change.
- **Pin a file inventory.** List each file the connector reads, with its row count, and fail on
  anything missing, unexpected or a different size.
- **Cite the source of truth.** The README says which release the paper used and why the
  connector reads that one. A reviewer checks the claim against the paper or the authors' code.
- **Cross-check releases.** When the authors publish two releases, compare them where they
  overlap. A disagreement means at least one of them isn't what we think.

### Coverage

What can go wrong: some records, subjects or channels never make it into the build, or appear
twice. Usually a filter is wrong, a parse error is swallowed, or a loop exits early. Less visibly,
a field the source or the authors' code provides is left out without a note, or the build holds
a value the source never had, such as an assumed time zone or a default.

How to catch it:

- **Compare counts with the paper.** Records, subjects, tasks per split and channels per record,
  each checked against the number the paper or the source documentation reports. A difference is
  either a bug or something the README must explain.
- **Account for every input row.** Each source row ends up either in the build or in a named
  exclusion bucket, such as "under 100 values". The buckets add up to the input count, and the
  README lists them.
- **Fail loudly.** Never skip a row because it failed to parse. Raise, or count it in a bucket
  that the snapshot tracks.
- **Check uniqueness.** No duplicate record ids, and no two records with identical values unless
  the README explains why.
- **Snapshot.** The [stats report](#stats-report) records these counts, so a PR diff shows any
  change.
- **Inventory every field.** List every file, column and field the source holds, and every input
  the authors' code reads, including what lives only in the code or the paper, such as a forecast
  horizon or a masked label. Mark each one as used, or say in the README why it is dropped.
- **Trace every value.** Every value in the build comes from the source, or from a reference the
  README cites, such as a code table. Anything else, such as a time zone the source doesn't state,
  is an assumption, and the README names it.

### Read path

What can go wrong: the data on disk is right, but loading it isn't. A lazy loader reads the wrong
bytes, values lose precision on a dtype change, or two builds of the same source differ.

How to catch it:

- **Round-trip tests.** Write, read back, and compare values, axes and tasks. Most connectors
  already have these.
- **Compare lazy and eager reads.** For a sample of records, read each value through the lazy
  loader and directly from the source, and compare.
- **Check dtypes.** Show that the chosen dtype holds every source value exactly, or document the
  loss. For example, values with at most two decimals fit in float32.
- **Check determinism.** Build twice and compare checksums.

### Channel meaning

What can go wrong: the values are right but labeled wrongly. Two channels are swapped, a channel
is in mV but labeled µV, a sign convention is flipped, raw ADC counts are stored without gain and
baseline, a missing-value marker such as −999 is stored as a real value, or category codes map to
the wrong names.

How to catch it:

- **Compare with an independent reader.** Read the same record with the source's standard
  library and compare channel by channel ([reference reader](#reference-reader)). A swap shows up
  as a match between the wrong pair of channels.
- **Check plausible ranges.** Each unit has a plausible range: ECG in mV, temperature in °C, heart
  rate in bpm. Values off by 1000× or a fixed factor point to a unit error. Look for repeated
  marker values like −999 or 9999.
- **Check relationships between channels.** Einthoven's law for ECG leads, humidity falling as
  temperature rises, visibility dropping in rain. These catch swaps that ranges miss, such as two
  channels with the same unit.
- **Check codebooks.** Compare category values with the source's documented code table. Look for
  codes the documentation doesn't list.
- **Spot-check upstream.** For a few records, compare with the upstream data the release was
  built from. This settles units and sign conventions directly.

### Time

What can go wrong: the values are right but placed wrongly in time. The time zone or start time
is wrong, the sample rate is wrong, resampling adds artifacts, gaps are filled or dropped
silently, sources drift apart, or labels line up with the wrong part of the signal.

How to catch it:

- **Check the sample rate.** Compare the number of samples with the duration. Compare the rate
  with the source header and the paper.
- **Check the time grid.** Every timestamp sits on its cadence, and gaps are counted and
  documented, not filled.
- **Check the time zone against physics.** Daily cycles such as temperature, light or activity
  peak at a known local time. Regressing the peak hour on longitude tells UTC from local time.
- **Check sync across sources.** Cross-correlate signals that should line up, such as heart rate
  from ECG and PPG, or accelerometer and gyro. The lag should be about 0.
- **Check labels against the signal.** For epoch labels, the signal inside each epoch should
  match the label: delta power in N3 sleep, motion during labeled activity.
- **Overlay with the original.** Plot the original and the TimeF version on the same time axis.
  A shift is obvious there and invisible in either plot alone.

### Identity and joins

What can go wrong: two tables are joined, such as signals with labels, windows with captions, or
recordings with subject metadata, and the join is wrong. Common causes are relying on the order
of rows, an off-by-one id, ids that collide across files, or a join on a key that isn't unique.
Every record still has a label, so nothing looks wrong.

How to catch it:

- **Join on explicit keys, never on order.** Where the source only gives an order, such as
  arrays without ids, check the alignment some other way before relying on it.
- **Check that each joined item agrees with its record.** Most joined items repeat something
  about their record: a caption names the place and dates, an event report names the county, a
  label file names the subject. Parse that and compare it with the record, for every row.
- **Compare with an independent join.** Rebuild the same pairs with the authors' code or a second
  release, and compare them pair by pair.
- **Check that the label predicts the signal.** A model trained on correct pairs does better
  than chance. A model trained on shuffled pairs scores at chance. If ours also scores at chance,
  the join is broken ([model check](#model-check)).
- **Corrupt it on purpose.** Shift the join by one and confirm that a check fails
  ([corruption tests](#corruption-tests)).

### Labels and targets

What can go wrong: the label is attached to the right record but has the wrong value. A class
code maps to the wrong name, the vocabulary order differs from the authors', labels are per
record where they should be per epoch, or the target is computed wrongly from the source.

How to catch it:

- **Compare the mapping with the authors' code.** Take the code → name table from the code that
  trained the paper's models, not from a README, and compare it with ours.
- **Compare label distributions with the paper.** Class balance per split, checked against what
  the paper reports.
- **Regenerate derived targets.** Where a target is built from source fields, such as generated
  captions or text templates, run the authors' code and compare byte for byte.
- **Check that labels agree with the signal.** The physical checks under [Time](#time), and the
  [model check](#model-check).

### Splits

What can go wrong: the splits differ from the authors', or the same subject, or overlapping
windows, appear in both train and test.

How to catch it:

- **Compare split sizes with the paper.** Record and task counts per split.
- **Use the authors' split files where they exist.** Do not regenerate splits.
- **Check for leakage across splits.** Group by subject, recording, anchor or event, and count
  groups that span more than one split. If the authors' splits leak, keep them but document it.
- **Compare membership.** Where the authors publish split files, compare membership id by id,
  not just sizes.

### Task construction

What can go wrong: the data is right, but the task built from it isn't. The prompt is wrong, the
series is placed at the wrong spot in the prompt, or the task type is wrong. The inputs may lack
what the target needs, so no model can answer it. A field may sit in the wrong role, such as
provenance passed as an input or a rationale shown to the model. Or an input, an id or a metadata
field gives away the target.

How to catch it:

- **Read rendered tasks.** A person reads a few tasks exactly as a model would see them
  ([review sheet](#review-sheet)). This is the only reliable check for prompt wording and
  placement.
- **Snapshot rendered tasks.** Commit a few rendered tasks as golden text, so any change to a
  prompt shows up in the PR diff.
- **Check the task can be solved.** For each part of the target, name the input it comes from. A
  caption that names a place needs the place as an input. Run a baseline or the
  [model check](#model-check), and document what no input supports.
- **Map every field to its role.** Signal, record metadata, annotation, task input, target, or
  rationale. Provenance, such as source ids and file names, stays in metadata. Supervision, such
  as a rationale or a report written after the fact, is never an input.
- **Look for leakage.** For every input, ask whether it could contain the target or information
  from after the target. Automated: check that no input annotation contains the target string.
  Manual: the README states what each input contains and when it was recorded.
- **Look for leakage outside the inputs.** A loader can pass a model more than the task's inputs,
  such as the record annotations that come with every sample. Check those fields for the target or
  something derived from it. A model given only those fields should score at chance.
- **Ids sorted by label aren't a leak on their own.** Evaluation shuffles samples, and a model
  doesn't read an id that no input carries. A sorted id matters only when it reaches an input,
  such as a file name placed in a prompt, or when a split is cut from a contiguous range of ids.
- **Cite each prompt.** The README says where each prompt comes from and whether the source
  confirms it.

### Card and provenance

What can go wrong: `dataset.yaml` or the README is wrong. The license, domain or description is
off, or the README claims something the data doesn't support.

How to catch it:

- **Review each claim.** Every number and statement in the README comes from either the code or a
  pass over the data. Each one cites which.
- **Generate numbers where possible.** README figures that come from the stats snapshot can be
  checked automatically. A test fails when the build changes but the README's number doesn't.
- **Check the license against the source.** A reviewer reads the source's license, not a
  summary of it.

## Verification checklist

The checklist goes in this order so that each step only relies on steps already checked. If the
source is wrong, nothing after it matters. If the read path is broken, every value check after it
reads wrong data. Within a step, the cheapest check with the biggest effect comes first.

Steps 1 to 3 are mostly automatic and cheap, and they gate everything after them. Steps 4 to 6
hold the silent, costly errors, which is where comparisons with the original data and physical
checks pay off. Steps 7 to 10 depend on the earlier steps being right.

To verify a connector, answer each item with a status, following
[Choosing a status](#choosing-a-status).

### Step 1. Source: is this the right data?

1. **Wrong release or version.** Compare with the release the paper or the authors' code uses.
2. **Unpinned input.** Pin an immutable revision, or a sha256 of the download.
3. **Partial, corrupt or changed download.** Check the checksum on every build, and check that the
   archive opens. The same check catches a source changed upstream under the same name.
4. **Wrong files from the archive.** Compare the files read with a pinned inventory. Fail on missing
   or unexpected files.
5. **Source structure misread.** Compare where two releases, or the docs and the data, overlap.
   Document any disagreement.

### Step 2. Coverage: is all of it there, only once, and nothing made up?

6. **Counts differ from the source.** Compare records, subjects, sessions and stations with the
   paper or the source's metadata table.
7. **Rows dropped silently.** Account for every input row: it is in the build or in a named
   exclusion bucket.
8. **Wrong filter or threshold.** Compare each filter with the authors' code, and record the count
   in each bucket.
9. **Duplicate records.** Check that record ids are unique, and hash the values to find duplicates.
10. **Missing or extra channels.** Compare the channel list with the source header or the paper,
    including derived channels kept by mistake.
11. **Source information left unused without a note.** List every file, column and field the
    source holds, and every input the authors' code reads. Each one is used, or the README says
    why it is dropped.
12. **Information the connector made up.** List every value the connector didn't copy from the
    source, such as an assumed time zone, a default, a label text or a prompt. Each one cites its
    reference, or the README names it as an assumption.

### Step 3. Read path: does TimeNet return what was written?

13. **Lazy loader reads the wrong bytes or record.** Read a sample through the loader and directly
    from the source, and compare.
14. **Precision lost on a dtype change.** Show that the chosen dtype holds every source value.
15. **Overflow or wraparound in integer conversion.** Compare the minimum and maximum before and
    after.
16. **Write and read don't round-trip.** Compare values, axes and tasks after reading back.
17. **Build isn't deterministic.** Build twice and compare checksums.

### Step 4. Channel meaning: are the values what their labels say?

18. **Raw ADC counts stored without gain and baseline.** Compare with the reference reader, and
    check ranges.
19. **Wrong units or scale.** Check plausible ranges, and look for outliers by 1000× or another
    fixed factor.
20. **Swapped or mislabeled channels.** Compare channel by channel with the reference reader and the
    source header, and check relationships between channels.
21. **Flipped sign convention.** Check upstream, or with a physical relationship.
22. **Missing-value markers stored as values.** Look for repeated marker values and spikes in the
    histogram.
23. **Wrong categorical codebook.** Compare with the documented code table, and flag codes it
    doesn't list.
24. **Implausible upstream values passed through without a note.** Run range checks, and document
    the counts.

### Step 5. Time: is each value placed correctly in time?

25. **Wrong sample rate.** Compare samples ÷ duration with the header and the paper.
26. **Wrong start time or epoch.** Compare with the reference reader and the source metadata.
27. **Wrong time zone.** Check daily cycles against longitude, or check upstream.
28. **Daylight-saving errors.** Look for days of 23 or 25 hours, and for duplicated or missing
    hours.
29. **Gaps or missing values filled or dropped silently.** Check the time grid, and compare NaN
    fractions with the source. Count and document gaps, and never fill them with 0 or interpolation.
30. **Artifacts from resampling or interpolation.** Overlay with the original, and compare spectra.
31. **Off-by-one sample.** Overlay or cross-correlate with the original.
32. **Sources or channels out of sync.** Cross-correlate signals that should line up. The lag should
    be about 0.

### Step 6. Identity and joins: is each item attached to the right record?

33. **Join relies on row order.** Join on explicit keys, or check the alignment independently.
34. **Off-by-one id.** Check that each joined item agrees with its record, for example the dates and
    place a caption names.
35. **Ids or join keys not unique.** Check uniqueness across files and splits, and the key's
    cardinality before joining.
36. **Subject metadata on the wrong recording.** Check that it is consistent across a subject's
    recordings.
37. **Join silently drops rows.** Count rows before and after, with the drops in a named bucket.

### Step 7. Labels and targets: is the label's value right?

38. **Wrong mapping from class code to name, or wrong vocabulary order.** Compare with the authors'
    training code, not the README.
39. **Label distribution differs from the paper.** Compare the class balance in each split.
40. **Wrong granularity, such as per record instead of per epoch.** Compare with the paper's task
    definition.
41. **Derived target computed wrongly.** Regenerate it with the authors' code and compare byte for
    byte.
42. **Label misaligned with, or contradicted by, its signal.** Check that each label agrees with the
    signal inside its epoch or window, with physical checks and a model check against shuffled
    labels.

### Step 8. Splits

43. **Not the authors' splits.** Compare membership id by id with the authors' split files. Without
    split files, compare split sizes with the paper.
44. **Leakage across splits.** Group by subject, recording and event, and count groups that span
    splits. Check windows from one source for overlap in time.
45. **Splits regenerated without a fixed seed.** Check determinism.

### Step 9. Task construction

46. **Wrong task type.** Compare with the paper's task definition.
47. **Task can't be solved from its inputs.** For each part of the target, name the input it comes
    from. Run a baseline or a model check, and document what no input supports.
48. **A field in the wrong role.** Map every field to its role: signal, record metadata,
    annotation, task input, target or rationale. Provenance stays out of the inputs, and
    supervision such as a rationale is never an input.
49. **Wrong prompt, or one without a source.** Cite where it comes from, and snapshot it as golden
    text.
50. **Series placed at the wrong point in the prompt.** Read rendered tasks.
51. **An input reveals the target.** Check that no input contains the target string. Document what
    each input holds.
52. **The target leaks through an id, metadata or another field.** Check the fields a loader
    passes a model besides the task's inputs, such as record annotations, for the target or
    something derived from it. A model given only those fields should score at chance. Ids
    sorted by label aren't a leak unless an input carries them or a split is cut from a
    contiguous range of ids, because evaluation shuffles samples.
53. **An input carries information from after the target.** Compare input timestamps with the
    window.
54. **Answer format doesn't match what the evaluation expects.** Read rendered tasks, and run a
    dummy evaluation.

### Step 10. Card and provenance

55. **Wrong license.** A reviewer reads the source's own license terms.
56. **Wrong domain, tags or description.** Review.
57. **A README number that doesn't match the build.** Generate it from the stats snapshot, or test
    it.
58. **A README claim the data doesn't support.** Each claim cites the code or the pass over the data
    it came from.
59. **Known limitations left undocumented.** Review against the stats report: implausible values,
    duplicates, leaky splits.

## The verification record

Each connector folder holds a `VERIFICATION.md`. Start from the
[template](connector-verification-template.md), and add it in the same commit as the connector.

- **Header.** The checklist version the record answers.
- **References.** The independent copies of the data that exist for this source: another release,
  the authors' preprocessed arrays or code, or the upstream data. Later checks compare against them.
- **Fields.** Every field of the source and of the authors' code, with its role in the build or
  the reason it is dropped, and every value the connector added, with its reference. The
  checklist's items on unused information, made-up values and roles cite this table.
- **Checklist.** One row per item, with a status and the evidence. Evidence names the build check
  or test that enforces the item, the one-off pass that checked it, or why it can't happen. For an
  open item, it says what is left and which planned check closes it.
- **Open items.** The open items, most important first, including decisions for maintainers.
- **Plan.** The checks that would close the open items.

### Choosing a status

Each item gets one of four statuses. The status says what kind of evidence exists, not how
confident the author is.

| Status | Means | The evidence |
| --- | --- | --- |
| ✅ Done | A test or build check in the repo enforces it, every build | Names the test or check in backticks |
| ☑️ Done once | Someone checked it against the real data or a reference, and nothing re-runs it | What was compared with what, over how much, and the result |
| ⚠️ Open | Nothing has checked it yet, or a check found a problem | What is left, and which planned check closes it |
| ⚪ N/A | What the item is about doesn't exist in this source | Why it doesn't exist |

Apply these rules, so that two authors give the same item the same status:

- **An argument isn't a check.** "The cadence check would catch it" or "the code joins by id" is
  a reason to expect the item is fine. If nothing ran, the item is ⚠️ Open, and the argument goes
  in its evidence along with the check that would confirm it.
- **A check counts only for what it tests.** A test on a synthetic fixture shows the code enforces
  a rule. It makes the item ✅ Done only when the item is about that rule, and the rule also runs
  on the real data at build time. It says nothing about what the real values mean: units, time
  zones, or which caption belongs to which window.
- **An item is no more done than what it relies on.** If its check assumes something that an open
  item questions, it is ⚠️ Open too, and its evidence names that item. A daylight-saving check
  can't pass while the time zone is open.
- **N/A is about the source, not the connector.** Use it only when the source has no such thing:
  no joins, no labels, no time zone. When the connector avoids a risk by design, such as joining by
  id instead of by order, the design still needs a check, so the item is ✅ Done, ☑️ Done once or
  ⚠️ Open.
- **A one-off check needs its numbers.** ☑️ Done once says what was compared, how many records,
  and what came out, such as "all 12,000 records, 0 mismatches". A check whose result wasn't
  written down is ⚠️ Open.
- **Split a partial answer.** When part of an item is checked and part isn't, combine the
  statuses with a spaced slash, such as ✅ Done / ⚠️ Open. The evidence covers each part in the
  same order.
- **When in doubt, pick the weaker status.** The reviewer can promote an item. A status that
  claims too much hides an open item.

### What the test checks

`test_every_connector_ships_a_verification_record`, in `timenet-connectors`, enforces the format:

- The file exists for every connector, has the four sections, and has no `TODO` left.
- It names a checklist version between 1 and the template's, and has every item of that version
  exactly once.
- Every status is one of the four, with or without the emoji's variation selector, and every row
  has evidence.
- Every ✅ Done row names its check in backticks, and every test it names exists, either in the
  connector's folder or in the package's tests.

The test can't tell whether the evidence is true. The reviewer checks that.

Connectors that predate the record are listed in the test's `_PENDING_VERIFICATION` set. Remove a
connector from it in the commit that adds its record. Never add one. Synthetic connectors, which
generate their data in code, have no source to check against and are listed in `_SYNTHETIC`
instead.

The record names no people. The connector's PR reviewer reads the answers and checks the risky
ones before the connector merges.

When the checklist changes, bump the template's version. Records on an older version still pass,
with a warning, until someone updates them.

## Why fixture tests aren't enough

Each connector's test runs `convert()` on hand-written synthetic fixtures. Those tests show the
code runs and enforces its own rules. They can't show that the real data was mapped correctly.

The fixture is written by the same person, with the same picture of the source, as the connector.
If the author misreads the source, the fixture repeats the mistake, and the test still passes.
Data can then be wrong for months before anyone notices, usually only once someone plots it next
to the original.

The verification record makes the author say, item by item, how they know the data is right.
Nothing in the repo checks a build against the real source automatically yet.

## Tooling to build

These shared checks would turn many ☑️ Done once and ⚠️ Open answers into ✅ Done. They are
listed roughly in order of value.

### Reference reader

A connector can optionally provide `reference_load(record_id)`. It reads the original data with
the source's standard library, written separately from the connector's own parsing:

- WFDB: `wfdb.rdsamp`
- EDF: `mne`
- CSV: `pandas`
- Other options: the authors' own preprocessed arrays, or a second release of the same data

A shared test samples K records, seeded so the same records come up on every run. It compares the
values with `allclose`, and checks sample rate, channel names, units, length and start time. It
runs on demand under a pytest marker, skipped unless the raw data is present locally, because
downloading it is too slow and some sources need credentials.

The reference must not share code with the connector. A shared bug would make the two agree.

### Stats report

The report lists:

- **Per channel:** range, mean, std, percentiles, NaN fraction and flatline fraction.
- **Per dataset:** number of records, subjects and tasks, the distribution of durations, and the
  balance of labels per split.

Commit it next to `dataset.yaml`, so a PR diff shows any change in what the connector produces.
Check the counts against the numbers the paper reports. Wrong subject counts and skewed class
balance are among the most common integration bugs, and the report catches them in seconds.

### Review sheet

The sheet takes a few samples, picked at random with a fixed seed so the author can't choose good
ones. For each sample it shows:

- the original data and the TimeF output, overlaid on the same axes
- the rendered prompt, with the series placed where they appear in it, and the target

The overlay is what makes the plot useful. A plot of only the TimeF output tells you very little:
it can't show a missing or mislabeled channel, a scaling error, or a shift in time.

### Domain checks

These are checks that rely on how the physical world works, and they catch mislabeled channels
and sync problems. They are written once per domain and then reused by every connector in it.

- **ECG:** Einthoven's relationships between the leads, and R peaks at the same times in every
  lead.
- **Sleep:** spectral power by sleep stage, and the hypnogram aligned with the EEG.
- **Wearables:** heart rate from different sensors agrees, and the accelerometer shows the
  wearer at rest during sleep labels.
- **Weather:** temperature peaks in the local afternoon, and humidity falls as temperature rises.

### Model check

Train a linear probe or a tiny baseline and compare its accuracy with what the paper reports. If
labels are shifted or misaligned against their signals, accuracy drops to chance. A plot will
never catch that.

### Corruption tests

Deliberately corrupt a build:

- swap two channels
- scale one channel by 1000
- shift one channel by a sample
- move every label one record over

Every corruption should make at least one check fail. If none does, the checks have a blind spot.

### Named evidence

The record test checks that every test a ✅ Done row names exists, which keeps ✅ Done honest as
tests are renamed or removed. A build check named in evidence, such as a constant or a function in
the connector, isn't checked yet. Checking it would close the last gap.

### Order of work

**First, because it is cheap:**

- A screenshot from the viewer in the PR, showing prompt and target.
- The [stats report](#stats-report). It is a small amount of code, every connector gets it
  automatically, and it catches scaling problems, wrong counts and missing channels.

**Then:**

- The [reference reader](#reference-reader) harness.
- [Domain checks](#domain-checks), starting with the domains that have the most connectors.
- The [model check](#model-check) on labels.
- The [corruption tests](#corruption-tests).
- [Named evidence](#named-evidence) for build checks, not just tests.
