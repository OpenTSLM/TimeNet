# tsaqa/tsaqa-benchmark

TSAQA, the Time Series Analysis Question Answering benchmark of Jing et al.
([arXiv:2601.23204](https://arxiv.org/abs/2601.23204), v2). It has 210,000 closed questions about
series from 13 domains. They span six tasks and three question formats. The
connector reads the Hugging Face release `TSAQA/TSAQA-Benchmark` at commit `883dcb0f`.

## The source data

### What the authors publish

The authors' repository, [SanhornC/TSAQA-Benchmark](https://github.com/SanhornC/TSAQA-Benchmark) at
`e0ea6f6`, holds a README, an MIT license and figures. It has no code to build, prompt or score the
benchmark; its README only loads the three parquet files. So the connector follows the paper and the
released data. Below, "paper" means v2 of the arXiv paper.

The paper (Section 3) defines six tasks in two groups:

| Group | Task | Formats | Series from |
| --- | --- | --- | --- |
| Conventional | Anomaly detection | TF | 6 anomaly detection benchmarks (Appendix A.2) |
| Conventional | Classification | MC | 37 UCR datasets with at most 4 classes (Appendix A.3) |
| Advanced | Characterization | TF, MC | 17 core datasets (Appendix A.1) |
| Advanced | Comparison | TF, MC | Core datasets, two series |
| Advanced | Data transformation | TF, MC | Core datasets |
| Advanced | Temporal relationship | TF, MC, PZ | Core datasets |

TF is true-or-false, MC multiple-choice and PZ puzzling (Section 3.1). GPT-4o writes the
characterization and comparison questions, and a weighted vote of four models sets their answers
(Appendix B.3). The other tasks use templates (Appendix B.5 to B.8). The paper says every series is
z-score normalized and printed to 4 decimals, and, except for classification, is 32 to 256 values long
(Section 3.3). Some are not normalized, as [Answers not computable from the series](#answers-not-computable-from-the-series)
counts.

### Files and columns

The release has three parquet files, `train.parquet`, `val.parquet` and `test.parquet`, with
147,000, 21,000 and 42,000 rows. Each is one row group of nine string columns:

| Column | Holds |
| --- | --- |
| `task` | One of the six tasks, such as `temporal_relationship` |
| `question_type` | `true_or_false`, `multiple_choices` or `ordering`, which 10,000 rows give wrongly |
| `input_ts` | The normalized series, as a JSON list; two lists for comparison |
| `meta_info` | A text that describes the series' source, which the paper calls its description |
| `question` | The question, with its choices. Data transformation and temporal relationship questions print their candidate series as JSON lists |
| `answer` | `T` or `F`, a letter `A` to `D`, or a puzzle order such as `C,B,D,A` |
| `domain` | One of the paper's 13 domains, such as `energy` |
| `dataset` | The source dataset, such as `m5` or `UCR_Classification_Wafer` |
| `raw_ts` | The series before normalization |

For comparison, data transformation and temporal relationship rows, `meta_info`, `domain` and
`dataset` are JSON lists of the same length. No source documents what each entry describes. The
data suggests this pattern, but the connector does not rely on it:

- A comparison names the source of each of its two series.
- A true-or-false data transformation or temporal relationship question names the input's sequence,
  then a distractor's, whatever the answer.
- A multiple-choice one names the input's sequence, then three distractors', in no letter order.
- A puzzle names the one sequence its patches come from.

A check of the release at the pinned revision supports this order for two tasks. Many input series
also appear in an anomaly detection, classification or characterization row, which publishes one
text for each column. Take each such series that has at least two distinct values and one known
context. Entry k of a comparison describes input k in 300 of 300 cases. Entry 0 of a data
transformation describes the input in 120 of 120 cases. A near-constant series looks the same in
several datasets after normalization, so it cannot be checked. Few temporal relationship series can
be checked, so the order is not verified for that task.

## The TimeNet dataset

### Records

Each Record holds one series as one lazy Signal on an ordinal axis, with no unit and no timestamps:
the release gives neither. A row gives a model its input series and then the series its question
prints, so each is a Record of its own.

| Role | Records | Values from |
| --- | --- | --- |
| `input` | 210,000 | `input_ts`, or its first series for a comparison |
| `second_input` | 30,000 | A comparison's second series in `input_ts` |
| `sequence` | 29,969 | The sequence a true-or-false data transformation or temporal relationship question prints |
| `A` to `D` | 4 × 60,031 | The choices of a multiple-choice data transformation or temporal relationship question, and a puzzle's patches |

That is 510,093 Records. A Signal's values are the published decimals as float64, so no value
changes. Its spec says what the values are:

| Spec | Signals | Values |
| --- | --- | --- |
| `tsaqa_series` | 435,051 | A series as published, or a patch of one. [Not every one is z-scored](#normalization) |
| `tsaqa_fourier_transform` | 25,268 | A Fourier transform question's choices or sequence |
| `tsaqa_wavelet_transform` | 24,871 | A wavelet transform question's choices or sequence |
| `tsaqa_first_order_difference` | 24,903 | A first-order difference question's choices or sequence |

A Record's metadata names its row and role: the `file`, the `row`
counted from 0, and the `role`. Metadata is provenance and never model input.

#### Normalization

The paper says every series is z-score normalized. The spec does not say so, because the published
input series do not all follow it. A series counts as z-scored here when its mean is within 0.01 of
0 and its standard deviation is within 0.01 of 1. A flat series has a standard deviation of 0.

| Task | Input series | Z-scored | Flat |
| --- | --- | --- | --- |
| Classification | 30,000 | 30,000 | 0 |
| Characterization | 30,000 | 29,080 | 548 |
| Comparison | 60,000 | 58,119 | 1,193 |
| Data transformation | 30,000 | 28,976 | 622 |
| Anomaly detection | 30,000 | 21 | 1,463 |
| Temporal relationship | 60,000 | 344 | 1,460 |

- **Anomaly detection and temporal relationship.** Almost no input is z-scored on its own. Each may
  have been normalized before it was cut from a longer series. The release does not say.
- **Printed series.** The series a question prints are not z-scored either. Only 194 of the 75,042
  data transformation candidates and 218 of the 195,051 temporal relationship series are. That is
  expected for a transform. For a patch, the release does not say how it was scaled.
- The values are kept as published. No value is rescaled.

**`raw_ts` is not read.** It holds each series before normalization: in every row but 3 anomaly
detection rows and the flat ones, `raw_ts[0]` maps onto the input by one scale and offset. Neither
the paper nor the README mentions the column. Appendix E.1 lists what a model reads, and `raw_ts`
isn't in it. A row's file and row in the task metadata lead back to it.

### Context

Appendix E.1 measures a model's input as `len(ts + description + domain + dataset + task +
question_type + question)`. Appendix E.2 says domain names "are provided to the model as textual
description". The paper doesn't say how these are joined, and no prompt code is published.

The connector keeps `meta_info`, `domain` and `dataset` as published, as annotations of the first
input Record. Each task names them in its `input_annotations`, so the model gets them as context.
A row that publishes one text for each column keeps the texts under those names. A comparison, data
transformation or temporal relationship row publishes lists, and the connector does not split them,
because no source says which entry belongs to which series. It keeps each whole list under
`published_meta_info`, `published_domain` or `published_dataset`, and it writes no `meta_info`,
`domain` or `dataset` key for that row. A row whose three lists differ in length fails. An
annotation key has one value type in TimeF, so the plain keys hold texts and the `published_*` keys
hold lists.

The model also reads the published `task` and `question_type`. Each distinct value is an annotation
of the dataset, and every task names its two in `input_annotations`. The 10,000 rows that publish a
wrong `question_type` keep the published value here. The question format that the connector reads
from the question is separate: it is in the task metadata, and it decides the task type.

### Tasks

Each row is one task. The question's format, read from the question's text, decides its TimeNet
type. Tasks keep the file order, in which the formats alternate. The row's paper task is its `task`
metadata.

| Format | TimeNet type | Target | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| `true_or_false` | `ClassificationTask` | `T` or `F` | 62,757 | 8,931 | 17,993 |
| `multiple_choice` | `AnswerTask` | One letter, `A` to `D` | 63,311 | 9,090 | 18,029 |
| `puzzling` | `AnswerTask` | Four letters, such as `C,B,D,A` | 20,932 | 2,979 | 5,978 |

- **A true-or-false question** is a classification, since `T` and `F` mean the same in every
  question. Its `target_schema` is the `true_or_false_vocabulary` dataset annotation, `["T", "F"]`.
- **A multiple-choice question or a puzzle** is an answer task. Its letters name choices that differ
  from one question to the next. A classification question's choices are its UCR dataset's class
  names, such as `normal process`. The `answer_text` metadata holds the right choice's text when the
  choices are text.
- **A puzzle's answer** lists the patch letters in chronological order. That is the paper's reading
  ("Arrange A, B, C, D in the correct order"), and the release agrees. The other reading takes each
  letter's position from the answer. In 16,012 puzzles the two readings differ and no patch repeats.
  In 13,790 of those, joining the patches in the answer's order gives smaller jumps at the joins.
- **The target** is the published answer, which the paper scores (Section 3.5). TF and MC are scored
  by accuracy, PZ by the share of positions that match. The release has no rationales.

**The prompt** is the published question. The exception is each series it prints, which is a Record:
the prompt shows `<series N>` in its place, the task's N-th input counted from 1. The inputs come first,
so the first printed series is `<series 2>`:

```text
Which of the following choices is most likely the future continuation of
the given time series?
Respond ONLY with the letter of the correct choice (A, B, C, or D).

Choices:
A: <series 2>
B: <series 3>
C: <series 4>
D: <series 5>
```

The placeholder is the connector's choice. The release prints each series as a line of its own, so
the connector reads each such line of a data transformation or temporal relationship question as a
series.

**Task metadata**, never model input:
- `task` and `question_type`, as published. The model reads the same values through
  `input_annotations`;
- `question_format`, read from the question, which routes the row to its task type;
- `file` and `row`;
- `transform`, for a data transformation question;
- `answer_text`, for a multiple-choice question with text choices;
- `source_issues` (see below).

### Splits

The connector keeps the authors' three files as the train, validation and test splits. The paper
splits each task 70/10/20 at random (Section 3.3), and the counts match: each file holds 70, 10 and
20 percent of each task's 30,000 rows (60,000 for temporal relationship).

## Comparing with the paper

**Counts.** The release matches the paper's counts:
- 210,000 rows;
- 30,000 per task, and 60,000 for temporal relationship;
- question types of 84,632 TF, 85,479 MC and 39,889 PZ (Section 3.4).

The PZ count is the release's `ordering` label, which includes 10,000 TF and MC questions (below).

**Domains.** The release names the paper's 13 domains. The dataset card maps them onto TimeNet's:

| Release | Card |
| --- | --- |
| `energy`, `transport`, `finance` | Same name |
| `healthcare`, `biomedical` | `health`, `cardiology` (ECG and heart rate) |
| `sales` | `economics` |
| `nature`, `environmental sensing` | `environment` |
| `web`, `it` | `observability` |
| `manufacturing` | `industrial` |
| `robotics` | `motion` |
| `synthetic` | `general` |

`finance` also covers FRED-MD's macroeconomic series, listed as `economics` too. Mapping `web`
(Wikipedia page views) to `observability` is our choice.

## Data limitations

The connector keeps the release as published. These issues are in the source data and are
documented here, not corrected.

### Question types labelled `ordering`

10,000 temporal relationship rows publish `question_type` `ordering` but ask a true-or-false
(5,049) or multiple-choice (4,951) question with a one-letter answer. The connector builds them as
their question's format, keeps the published label in the `question_type` metadata, and flags them
`question_type_mismatch`. The paper's PZ column (Table 3) likely includes them. For a one-letter
answer, its position score equals accuracy.

### Questions repeated in the release

8,669 rows ask the same question about the same series as another row, with the same answer.
The published `question_type` does not count, since the model never sees it. 4,514 of them have a
copy in another split, so a model can train on a test question:

| Task | Repeated | Across splits |
| --- | --- | --- |
| Classification | 7,227 | 3,440 |
| Anomaly detection | 965 | 632 |
| Temporal relationship | 389 | 418 |
| Data transformation | 84 | 22 |
| Characterization | 4 | 2 |

The "across splits" count includes rows whose copy expects another answer, which is why temporal
relationship has more such rows than repeated ones. Most classification copies come from UCR
datasets that share series: FreezerSmallTrain is drawn from FreezerRegularTrain, and
PhalangesOutlinesCorrect pools the Distal, Middle and Proximal phalanx sets. The copies keep their
own `dataset` and `meta_info`, so they differ only in context.

### Conflicting answers

556 rows ask the same question about the same series as another row, but expect another answer.
532 are puzzles whose patches repeat, as on a flat stretch. The other conflicts are 17 temporal
relationship and 5 data transformation true-or-false questions, and 2 classification questions.

### Ambiguous answers

975 rows have another answer that is as right as the published one, but only the published one
scores. The connector flags them `ambiguous_answer`:

- 935 puzzles have two or more identical patches, so more than one order rebuilds the sequence.
  They include the 532 conflicting puzzles.
- 39 multiple-choice questions print the right choice's series again as another choice: 21 data
  transformation and 18 temporal relationship questions.
- 1 comparison question (test row 28172) offers the right choice's text twice.

### Questions that can't be answered by inputs


The paper says every series is z-score normalized. In 57,333 of the 60,000 comparison and
characterization rows, every input is: its mean is within 0.01 of 0 and its standard deviation
(over all values) is within 0.01 of 1. The other 2,667 rows are not: 1,665 have a flat input whose
values are all 0, and 1,002 have an input with a standard deviation from 0.91 to 0.99. Some of the
questions on z-scored inputs ask about the scale that z-scoring removes. GPT-4o writes them, and a vote of four models sets their answers (Appendix B.3), so an
answer isn't computed from the series. The connector keeps these rows as published, and no source
issue flags them. The table counts only rows whose inputs are all z-scored:

| Question | Train | Val | Test | Total |
| --- | --- | --- | --- | --- |
| Which of two series has the higher variance or standard deviation | 636 | 101 | 206 | 943 |
| Which absolute number describes the values, such as a range or a peak | 469 | 66 | 120 | 655 |

- **Spread comparisons.** In all 943 rows both inputs have standard deviation 1, so the inputs
  can't tell the series apart. The unread `raw_ts` column doesn't explain the answers either. The
  check covers the 536 rows whose question names the two series as "time series 1" and "time series
  2" (or "first" and "second"), so that an answer maps to one series. In 179 of 325 true-or-false rows
  and 116 of 211 multiple-choice rows, the answer follows the raw series' order of standard
  deviation, close to chance. The other 407 rows name their series another way, such as by domain, so
  the check skips them. A spread question about a change inside a series, such as a variance shift,
  can be answered and isn't counted.
- **Absolute numbers.** In 285 of the 655 rows, the right number is one of the input's values or its
  maximum, minimum, mean, standard deviation, variance or range, so the input can answer it. 14 match
  only `raw_ts`. 356 match neither: 115 of them carry a physical unit, such as `From -8.86 mm to 3.19
  mm` or `1.5 kW`, which a z-scored input can't show. The other 241 may be statistics that this count
  doesn't compute, such as the number of peaks.

The counts read each question's text, so a question that words its scale differently isn't counted.
The rules, applied to comparison and characterization rows whose inputs are all z-scored:

- A **spread comparison** is a comparison whose question says "higher", "greater", "larger", "lower",
  "smaller", "less" or "more" next to "variance", "standard deviation", "variability" or "spread". It
  must not contain "shift", "change", "local", "rolling", "within", "half", "segment", "window" or
  "revert".
- An **absolute-number question** is a multiple-choice question that names a statistic (such as
  "standard deviation", "variance", "mean", "range", "maximum", "minimum", "peak" or "amplitude")
  and offers three or more choices that are all numbers, with or without a unit.
- A right number is **found in the input** when it is within 0.05% of one of the input's values, or
  within 0.5% of its maximum, minimum, mean, standard deviation, variance or range. "Found in
  `raw_ts`" uses the same test on `raw_ts`.
- A series is **shared** when its published values are the same, as JSON text, in a row of another
  split.

### Series shared across splits

10,110 rows have an input series that is also the input of a row in another split: 5,764 train, 1,565
validation and 2,781 test rows. 4,514 of them ask the same question about it, and the connector flags
them `repeated_across_splits` (see [Questions repeated in the release](#questions-repeated-in-the-release)).
The other 5,596 ask another question, so a model can train on a test series under another question,
and no source issue flags them:

| Task | Same question | Another question only | Total |
| --- | --- | --- | --- |
| Classification | 3,440 | 1,906 | 5,346 |
| Comparison | 0 | 1,438 | 1,438 |
| Temporal relationship | 408 | 539 | 947 |
| Characterization | 2 | 899 | 901 |
| Data transformation | 22 | 824 | 846 |
| Anomaly detection | 632 | 0 | 632 |

The counts compare each input series exactly, so a comparison counts when either of its two series
is shared. 3,377 distinct series are shared. A series cut from the same source at another offset
isn't counted, because the release names no source sequence.

### Wrong transformation answers

Four true-or-false data transformation rows have an input of zeros and a sequence of zeros, and
the answer `F`. The Fourier transform, the wavelet transform and the first-order difference of a
zero series are all zeros, so `T` is the only answer that fits. They are train rows 106397,
106493, 120546 and 121942. Rows 106397 and 106493 also have `conflicting_answer`. The connector
keeps the published answer and flags the rows `wrong_transform_target`. It does not correct a
label. No multiple-choice row names a nonzero sequence as the answer for an all-zero input.

### Other notes

- **Classification choices.** All 3,999 TwoPatterns questions print each class's case count in its
  choice, such as `A: down-down (1306 cases)`. No other dataset does.
- **Generated series.** Not every series is a measurement:
  - the 5,277 classification rows of the `synthetic` domain come from TwoPatterns, CBF, BME,
    SmoothSubspace and UMD, which are generated UCR sets;
  - the 376 MGAB anomaly detection rows are Mackey-Glass series, listed under `biomedical`.
- **Distractors.** A data transformation question's wrong choices are transforms of other series,
  which the release doesn't hold. A multiple-choice one's `raw_ts` holds five series for four
  choices, and the four after the input don't reproduce the choices.
- **Licenses.** The release is MIT. The series come from public datasets that have their own terms,
  such as M5, PEMS03, the UCR archive and Building Data Genome 2 (paper Appendix A).

## Filtering rows with source issues

Each task lists its row's issues in its metadata under `source_issues`, an empty list when it has
none. The issues are the values of `SourceIssue`:

| Issue | Rows | Meaning |
| --- | --- | --- |
| `question_type_mismatch` | 10,000 | `question_type` names another format than the question has |
| `repeated_question` | 8,669 | Another row asks the same question about the same series, with the same answer |
| `repeated_across_splits` | 4,514 | A copy is in another split |
| `conflicting_answer` | 556 | Another row asks the same question about the same series, with another answer |
| `ambiguous_answer` | 975 | Another answer is as right as the published one |
| `wrong_transform_target` | 4 | The input is all zeros, so its transform is zeros, and the answer says otherwise |

19,381 tasks have at least one issue. The connector keeps them all. No issue flags the
[answers not computable from the series](#answers-not-computable-from-the-series), or the
[series shared across splits](#series-shared-across-splits) under another question.
`without_source_issues` leaves them out, and the connector doesn't call it:

```python
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import (
    SourceIssue,
    without_source_issues,
)

# Leave out the test questions a model may have seen in training,
# the ones whose copies expect another answer, and the ones with
# more than one right answer.
clean = list(
    without_source_issues(
        tasks,
        issues=[
            SourceIssue.REPEATED_ACROSS_SPLITS,
            SourceIssue.CONFLICTING_ANSWER,
            SourceIssue.AMBIGUOUS_ANSWER,
        ],
    )
)
```

Report which issues were left out, and how many tasks remain, with any score.

## Data flow and validation

A build reads every file once to check each row and keep its small fields: its answer, question
format, series lengths, a digest of the series values, and source issues. Source issues need all three splits, so a build without one
of them fails. It checks that every column is text, the task and question type are known, a
multiple-choice question offers A to D in order, the answer fits the question's letters, each choice
or patch of a data transformation or temporal relationship question prints one series, and every
series is a nonempty list of finite numbers. The series values stay in the files until they are
written, and the connector rereads a file when the writer comes back to it. A reread fails if the
lengths or the values of a row's series differ from the build. Everything is in `connector.py`.

The tests read a small fixture release, real rows of the pinned release with all their columns
(`tests/README.md` names each one).

On the full release, run on 2026-10-29: every one of the 210,000 tasks read back from a full build
matches its row:
- the target is the answer;
- every Record's values are the published series;
- the context annotations are the published text.

## Local build

The build downloads the three files, 387 MB, from the Hub at the pinned commit:

```bash
uv run timenet-build build tsaqa/tsaqa-benchmark --out /tmp/timenet
```

On 2026-10-29, building in-process took 79 s and peaked at 4.1 GB of memory, most of it the 510,093
Records. The built dataset takes 123 MB. To load it:

```python
from timenet.client import TimeNet

dataset = TimeNet("/tmp/timenet").load("tsaqa/tsaqa-benchmark")
```
