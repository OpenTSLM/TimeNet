# ARFBench

750 multiple-choice questions about anomalies in Datadog observability metrics. One record is one
question: it carries the one or two metrics the question cites, one signal per tag group, and an
`AnswerTask` holding the question and its answer. The release also ships 414 rendered plots of the
same numbers, which this connector does not convert.

- **id**: `datadog/arfbench`
- **source**: https://huggingface.co/datasets/Datadog/ARFBench
- **licence**: Apache-2.0

Read at commit `1cc8ae54e9633f596b755f7c4bce54ccb0cb9f5a`, not at a branch, so the same code always
reads the same bytes. Every number marked *(measured)* was counted over the whole scope this
connector opens at that commit: 205 Parquet files and 5,518,548 rows. Nothing was sampled. The
counts of the repository as a whole come from the Hub's tree listing at the same commit.

## The source of truth

The release states several facts twice, and the two statements disagree. Each row below names the
one the connector reads and keeps the other beside it.

| fact | source of truth | the other reading |
| --- | --- | --- |
| which metric a file holds | the file name `query_group` joins to | its `query_name` column |
| how many intervals a metric has | the file names | the card's "up to 6 ... intervals" |
| the order of a file's rows | the `epoch` column, after sorting | `__index_level_0__` |
| how many tag groups a file holds | its own `group` column | its own `num_groups` column |
| the candidate answers of a question | `options_str`, valid JSON | `options`, a Python repr |
| how many data points there are | **5,518,548** rows in scope *(measured)* | the paper's "5.38M" |

The data wins in every row, and the reason is the same each time: a reader can re-measure a column
and cannot re-measure a sentence. Each losing reading keeps its own entry under "Inconsistencies
and decisions", with the count of files on which the two disagree *(measured)*. The last row is the
exception: the paper's abstract says "5.38M data points", and no interval choice this connector
could make reproduces it. The first entry below is where the 5,518,548 comes from. The losing
readings stay in this file, because someone who arrives at this dataset through the card needs to
know why the numbers differ.

## What the description states

The prose settles things no file header states.

> Query Group: The unique identifier for the time series referenced in the question. This should be
> used to find the time-series data or image data associated with the question.
> — the release's dataset card, "Dataset Structure"

Decides the join key. A question finds its series through `query_group` and the file name, and never
through the `query_name` column inside the file.

> Time Series data: for each query group there are up to 6 different intervals of the same data.
> — the release's dataset card, "Dataset Structure"

Decides that "up to" is load-bearing. 69 of the 142 metrics publish all six intervals *(measured)*,
so an interval has to be chosen per question rather than pinned once for the dataset.

> Interpolation flags: whether visualizations were interpolated in the original time series seen by
> incident engineers.
> — the release's dataset card, "Dataset Structure"

Decides that the flags describe the picture an engineer looked at during the incident, not the
numbers in the Parquet file. This connector stores the flags and interpolates nothing.

> Time Series plots: A Matplotlib or Plotnine generated png, directly created from the time series
> data.
> — the release's dataset card, "Dataset Structure"

Decides that the 414 images are a rendering of the same numbers, so leaving them out loses no
observation.

> ARFBench consists of 142 unique time series collected from 63 different incident discussion
> threads
> — the release's dataset card, "Dataset Summary"

Decides the shape of the corpus, and the QA table agrees: the 750 questions cite 142 distinct
metrics from 63 distinct incidents *(measured)*.

> Note: the metrics comprising ARFBench were generated from internal monitoring and do not include
> any customer data.
> — the release's dataset card, "Dataset Summary"

Decides nothing in the conversion. It is repeated here because the tag labels read like production
infrastructure, and they are not customer names.

> Missing data points are filled with null.
>
> Null values are excluded.
> — two of the metric descriptions inside the `question` column

Decides what a null `value` means. The release describes a metric per question, and those
descriptions distinguish a metric whose missing points were filled with null from one whose nulls
were excluded. **63** of the 750 questions describe a filled metric and **12** an excluded one, and
those 75 are every question that mentions a null at all *(measured)*. The clause is worded **five**
ways for a filled metric and **two** for an excluded one, and the sentences quoted above are one
wording of each, on **16** and **3** questions *(measured)*. A null is therefore a statement the
release makes, not an artifact.

The description states **no unit for any metric**, and it states no scale. The only account of what
a value means is the English prose inside the question.

## What one record holds

One record is one row of `arfbench-qa.csv`.

**Signals.** One `TimeSeries` per (metric, tag group) pair of the file the question resolves to,
named `{metric}/{tag label}` and ordered by tag label. A question citing two metrics carries the
signals of both, which is why the name is prefixed: two metrics in one question can present the same
tag label, and without the prefix a reader could not tell them apart. Across the scope this is
**10,556 stored series holding 5,518,548 values** *(measured)*.

Values keep the source's `float64`. The largest in scope is about **6.1e10** *(measured)*, and
`float32` carries about seven significant digits, so casting would halve the values plane (about
22 MB before compression) and quantize the largest values to steps of roughly 4,000.

Every signal is **dimensionless**. It is not that these numbers have no unit; it is that the release
does not say what it is.

**The time axis** comes from the file name's interval and the `epoch` column. A signal that holds
every step of its file's grid gets a `RegularAxis` at that interval, placed on the grid by its first
observation. A signal that skips a step carries its own int64 time offsets under an `IrregularAxis`:
**1,182 of the 10,556 series**, holding **1,026,318 values, 18.6% of the total** *(measured)*.

Every record is anchored at `2025-03-07T00:00:00Z`, and not at its own first observation. That is a
derived origin, the UTC day boundary the corpus starts in, and the entry below says where it came
from. The shared zero is what lets one stored copy of a metric serve every question that cites it.
The side effect is that `start_time` no longer distinguishes records, so a reader looks at the axis
to see when an incident happened.

**Annotations**, all scoped to the whole record:

| key | value | where it came from |
| --- | --- | --- |
| `task_category` | one of eight, e.g. `Anomaly Presence` | the `task_category` column |
| `difficulty` | `Tier 1`, `Tier 2` or `Tier 3` | the `difficulty` column |
| `query_group` | the metric ids the question cites | the `query_group` column |
| `interval_s` | the interval this question resolved to | the interval rule, not a column |
| `incident_ids` | the incidents the metrics came from | parsed from the metric ids |
| `interpolate_1`, `interpolate_2` | `true` | the rows that set them |

`query_group` holds the metric ids as the table writes them, and `interval_s` is in seconds.
`incident_ids` is a list because **200 of the 750** questions cite metrics from two different
incidents *(measured)*, and one id per record cannot hold both. The two flags ride only on the
**47** and **18** rows that set them *(measured)*.

**The record id is positional**: `arfbench-000` to `arfbench-749`, the question's row number in the
QA table. The table's own unnamed first column is exactly that number, so nothing is invented and
nothing is carried twice. Two builds of this release give the same ids; a re-release invalidates
them. Every other id this connector states is built from the same `arfbench` prefix: a signal is
`arfbench-{metric}-{interval}-{index}` and a shared candidate-answer list is
`arfbench-options-{digest}`. Record and task annotations carry generated ids, because nothing
resolves them by id.

## The tasks this connector builds

| the question | type | count | scope |
| --- | --- | --- | --- |
| answer this question about these series | `AnswerTask` | **750** *(measured)* | the whole record |

There is one task per QA row, and its `prompt` is the question as the table writes it, including the
embedded newline that splits it across two physical CSV lines: all 750 questions carry one
*(measured)*. The `target` is the single correct option. The candidate answers do not ride on the
task; each task points at a shared `answer_options` annotation, and the 750 tasks reference **278**
of them *(measured)*.

**The tasks stream.** `convert` builds the records and registers the option lists, then hands the
tasks over as a stream that re-reads the QA table. Each task names its own record, and none appears
in `Record.task_ids`, so the record rows on disk carry no task id. A read rebuilds that reverse map
from what each task says, so `tasks_for()` still answers on a dataset read back from disk. The
writer validates every streamed task on its way there. What a stream cannot check is what needs all
750 tasks at once, which here is the duplicate-id check alone: no task in this dataset derives from
another.

## Inconsistencies and decisions

### No single sampling interval answers every question — **Handled**

**Problem.** The release publishes each metric at up to six intervals: 10, 60, 300, 1800, 3600 and
86400 seconds. Only **69 of the 142** metrics publish all six *(measured)*. Pin one interval for the
whole dataset and questions fall off the end: the best single choice, 1800 s, drops **23** of the
750, and the finest, 10 s, drops **154** *(measured)*.

**Decision.** Choose per question: the finest interval that every metric that question cites
publishes. The rule is decidable from the file names alone, which is why `download` lists the
repository before it fetches anything.

**Consequence.** All 750 questions keep their series. They resolve to **596** at 10 s, **105** at
60 s, **18** at 300 s, **24** at 1800 s and **7** at 3600 s *(measured)*; no question resolves to
86400 s. Two questions citing the same metric can therefore read it at two different intervals, and
`interval_s` on each record says which. This also fixes the download scope at **205 of the 748**
series files.

### 239,302 rows publish a null value — **Handled**

**Problem.** **239,302 of the 5,518,548** rows in scope carry a null `value`, **4.34%**, spread over
**3,161 of the 10,556** series *(measured)*. The release means them, as the quoted metric
descriptions above show, and every one of the **63** QA rows that sets an interpolation flag cites a
metric that has null rows *(measured)*.

**Decision.** Keep the row and mark the timestep missing, under `nullable=True`. Not NaN: TimeF
treats NaN as a value that was observed and happens not to be a number, and the release calls these
points missing.

**Consequence.** Nothing published is dropped. Deleting those rows instead would lose **1,369 whole
series** whose every value is null, and it would push the irregular series from 1,182 to **2,167**
and the irregular share of the values from 18.6% to **39.4%** *(measured)*, because a deleted row
leaves a gap that then needs its own int64 time offset. The cost is that the one shared spec makes
all 10,556 series nullable, so a values backend writes a validity bit for every series including the
**7,395** that hold no null *(measured)*. That is about 5.5M highly compressible booleans against
the 8.4 MB of int64 offsets the decision removes. For a consumer it means `to_numpy()` raises on a
series that holds a null; read these with `to_numpy_and_mask()` or `to_arrow()`.

### Row order inside a file is arbitrary — **Handled**

**Problem.** Only **7 of the 205** files in scope arrive sorted by tag group and then by epoch, with
the tag groups in alphabetical order *(measured)*. Neither representation holds a time series until
it sorts.

**Decision.** Sort by tag group, then by epoch, once at build time.

**Consequence.** The sort is charged to whoever reads the raw files on every read, and to this
connector once. Any comparison of the two that does not charge both sides is not measuring the
format.

### `__index_level_0__` does not restore that order, and 5 files ship none — **Handled**

**Problem.** The files carry a pandas index column, which `pandas.read_parquet` restores as an
unnamed index and never as a data column. It is not a `0..n-1` range either: of the **200** files
that carry one, **3** hold a permutation of `0..n-1`, and sorting on it recovers the tag group and
epoch order on **6** *(measured)*. **5 of the 205** files carry no such column at all *(measured)*.

**Decision.** Do not read it. The connector asks Parquet for `epoch`, `group` and `value` and
nothing else, so the column is never read and the five files without it need no special case.

**Consequence.** The largest single saving in this conversion, and the largest bias in any
comparison against the raw files: `__index_level_0__` is **27,542,982 B, 51.42%** of the
column-chunk bytes of the 205 files *(measured)*.

### `query_name` names a different metric than the file it sits in — **Handled**

**Problem.** Each file repeats a `query_name` on every row. It disagrees with the file's own name on
**177 of the 205** files in scope *(measured)*, and the card says a question finds its series
through `query_group`, which matches the file name.

**Decision.** Drop it. Carrying it would invite a wrong join.

**Consequence.** **16,704 B, 0.03%** of the column-chunk bytes *(measured)*. Nothing in the
converted dataset records the release's other name for a metric.

### `num_groups` disagrees with the file's own group count — **Handled**

**Problem.** Each file repeats a `num_groups` on every row. It equals the number of distinct values
in that file's `group` column on only **151 of the 205** files in scope *(measured)*.

**Decision.** Drop it and count the groups from the `group` column, which is what the stored signals
are built from.

**Consequence.** **68,990 B, 0.13%** of the column-chunk bytes *(measured)*. A reader who wants the
release's own count has to go back to the Parquet file.

### The long format repeats the epoch and the tag label on every row — **Handled**

**Problem.** One row is one (epoch, tag group) observation, so both columns are stored once per
value.

**Decision.** One signal per (metric, tag group). The epoch becomes an axis and the tag label
becomes a signal name, each stated once.

**Consequence.** `epoch` is **7,531,283 B (14.06%)** and `group` is **5,509,527 B (10.29%)** of the
column-chunk bytes, 24.35% together *(measured)*. This is the mechanism behind those two savings and
belongs beside them.

### The shared anchor is a derived origin — **Handled**

**Problem.** The release states no zero for its timeline. Every file carries absolute epochs, and
the observations in scope run from **2025-03-07T00:00:10Z to 2025-03-29T23:59:50Z** *(measured)*. A
record anchored at its own first observation would give one metric a different set of time offsets
in every question that cites it, and the stored copy could then serve only one of them.

**Decision.** Anchor every record at `2025-03-07T00:00:00Z`, the UTC day boundary the corpus starts
in. It is 10 s before the earliest observation in scope, so `ANCHOR_US + offset` reproduces every
source epoch exactly and no time offset is negative *(measured)*.

**Consequence.** The whole time axis hangs on a constant in `connector.py` rather than on a fact the
release states, so the constant is part of this connector's contract with the pinned revision. A
re-pin whose corpus starts earlier fails loudly rather than quietly: `RegularAxis.start_index` and
`IrregularAxis.first_us` both refuse a negative start and raise `TimeFValidationError`.

### A metric several questions cite is stored once — **Handled**

**Problem.** The questions overlap heavily. Read naively, per question, the scope holds **52,659**
record-to-series references and **34,972,881** values *(measured)*.

**Decision.** Read a metric once per interval and hand the same `TimeSeries` objects to every record
that cites it. The shared anchor is what makes this legal: two records with the same zero agree
about what a time offset means.

**Consequence.** **10,556 stored series and 5,518,548 stored values**, a factor of **6.34**
*(measured)*. On a full-read throughput comparison this one decision can account for most of a win,
so a comparison lane needs the same per-file cache to be fair.

### A signal name is the metric id and the tag label joined — **Handled**

**Problem.** A metric with no tag grouping carries an empty tag label in the source, and a TimeF
signal name must be non-empty. Two metrics in one question can also present the same tag label.

**Decision.** The name is `{metric}/{tag label}`, and an empty label becomes the literal `value`.

**Consequence.** **21 of the 10,556** series are named `.../value` *(measured)*. Read that as a
placeholder, not as a tag the release wrote. The prefix costs **84,755 B** of name strings across
the scope *(measured)*. A file that carried both an empty label and a real label spelled `value`
would put two tag groups under one name, so the decoder raises `TimeFFormatError` instead of losing
one of them. No file in scope carries that pair *(measured)*.

### Only the files the questions resolve to are downloaded — **Handled**

**Problem.** The repository holds 748 series files, 414 plot images, the QA table and its own card:
**103,352,917 B** in total *(measured)*. The interval rule reaches 205 of the series files.

**Decision.** Fetch the QA table, resolve the interval per question, then fetch exactly those 205
files.

**Consequence.** The raw tree on disk is **55,164,608 B** and is exactly what the loaders read
*(measured)*. **16,878,158 B** of unopened Parquet and **31,302,963 B** of images stay in the
repository. This makes the raw side of any comparison **23 percent smaller** than counting all 748
Parquet files would, and the smaller number is the honest one: it is the bytes something opens.

### The candidate answers are stored once per distinct list — **Handled**

**Problem.** The QA table ships the answer choices twice, as `options_str` (JSON) and `options` (a
Python repr of one-key dicts). The two parse to the same list on all **750** rows *(measured)*. The
750 rows hold only **279** distinct `options_str` strings.

**Decision.** Read `options_str`, drop `options`, and register one `answer_options` annotation per
distinct parsed list, keyed by a hash of the list rather than of the raw text. Each task points at
the annotation it shares.

**Consequence.** **278** stored option lists for 279 distinct strings *(measured)*: one pair differs
only in the whitespace after its commas and parses to the same list, so those two share one
annotation.

### The interpolation flags are stored and nothing is interpolated — **Handled**

**Problem.** The release sets `interpolate_1` on **47** rows and `interpolate_2` on **18**
*(measured)*, and the card says they describe the visualisation an engineer saw. The release does
not ship the interpolated series.

**Decision.** Store the flags as annotations on the rows that set them. Interpolate nothing.

**Consequence.** On those questions the stored series is not what an evaluation following the
release's intent would show a model. The flag is how a reader finds them.

### The QA table's unnamed first column is not carried — **Handled**

**Problem.** `Unnamed: 0` is a pandas index left in the CSV.

**Decision.** Drop it. It is exactly the row number `0..749` *(measured)*, which the record id
already states.

**Consequence.** None: 750 integers that were already stated twice.

### A build decodes the release twice — **Handled**

**Problem.** A signal's axis depends on its values: whether it skips a step of the grid is only
visible after the rows are read and sorted. So `convert` opens all 205 files to work out each
signal's axis and length, and the loaders open them again when the writer asks.

**Decision.** Accept the second read and keep it to one decode per file, with an LRU cache of two
decoded files. Two is enough because the writer asks for series in `source_id` order and `source_id`
is the file, so one file's signals are always requested in a run.

**Consequence.** A build reads about 110 MB rather than 55. The cache is module-level, so it is
shared by every connector instance in a process; a test that writes its own fixture clears it.

### The card declares the observability domain — **Handled**

**Problem.** Infrastructure telemetry is not health, activity or finance, and the domain enum has no
closer entry.

**Decision.** `observability`. The alternative was the catch-all `general`.

**Consequence.** It sets the one domain all 750 records are filed under, so it decides which domain
filter finds this dataset. `general` would be a filter nobody can narrow with.

### A comparison lane must derive the interval map the same way — **Open**

**Problem.** This connector derives which intervals a metric publishes from the Hub's tree listing
of all 748 file names. A reader working from a downloaded copy derives it from the 205 files on
disk. The two agree only because the interval a question resolves to is downloaded for every metric
that question cites, which is an argument and not a check.

**Decision.** Whoever compares this connector against another reader of the same release has to
confirm the two interval maps agree before the comparison means anything. Nothing in this connector
or its tests confirms it.

**Consequence.** Until it is confirmed, do not read a difference between two readers of ARFBench as
a difference between formats. It could be the two sides reading different intervals for some
questions, with no error raised anywhere.

## Warnings this build emits

None. Every inconsistency above holds for the whole release, not for a handful of records, so each
one is decided in code and written down here. A warning per record would repeat one fact 750 times
and add nothing to the counts in this file.

## What is not built

The **414 plot images** under `arfbench-images/`. The card says each is generated directly from the
series data, so they carry no observation the Parquet files do not, and TimeF stores no images.

The **543 series files at the other sampling intervals**. They hold the same metrics the scope
already covers, at an interval no question asks for, so reading them would convert the same
observations twice under two axes.

The **86400 s interval**, by the same rule: no question resolves to it *(measured)*.

The **leaderboard and the released QA model**, which are the benchmark's evaluation harness and not
part of the dataset.

The **link between a metric and the incident thread it was discussed in**. The release states the
incident id inside the metric id and nothing finer, so a record cannot be traced to a conversation,
a service or a team.
