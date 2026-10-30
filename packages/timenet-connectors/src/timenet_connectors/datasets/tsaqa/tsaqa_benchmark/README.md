# tsaqa/tsaqa-benchmark

TSAQA, the Time Series Analysis Question Answering benchmark of Jing et al.
([arXiv:2601.23204](https://arxiv.org/abs/2601.23204), v2). It has 210,000 closed questions about
z-score normalized series from 13 domains. They span six tasks and three question formats. The
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
(Appendix B.3). The other tasks use templates (Appendix B.5 to B.8). Every series is z-score
normalized and printed to 4 decimals, and, except for classification, is 32 to 256 values long
(Section 3.3).

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
`dataset` are JSON lists:

- A comparison names the source of each of its two series.
- A true-or-false data transformation or temporal relationship question names the input's sequence,
  then a distractor's, whatever the answer.
- A multiple-choice one names the input's sequence, then three distractors', in no letter order.
- A puzzle names the one sequence its patches come from.

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
| `tsaqa_series` | 435,051 | A normalized series, or a patch of one |
| `tsaqa_fourier_transform` | 25,268 | A Fourier transform question's choices or sequence |
| `tsaqa_wavelet_transform` | 24,871 | A wavelet transform question's choices or sequence |
| `tsaqa_first_order_difference` | 24,903 | A first-order difference question's choices or sequence |

A Record's metadata, `records.TSAQARecordMetadata`, names its row and role: the `file`, the `row`
counted from 0, and the `role`. Metadata is provenance and never model input.

**`raw_ts` is not read.** It holds each series before normalization: in every row but 3 anomaly
detection rows and the flat ones, `raw_ts[0]` maps onto the input by one scale and offset. Neither
the paper nor the README mentions the column. Appendix E.1 lists what a model reads, and `raw_ts`
isn't in it. A row's file and row in the task metadata lead back to it.

### Context

Appendix E.1 measures a model's input as `len(ts + description + domain + dataset + task +
question_type + question)`. Appendix E.2 says domain names "are provided to the model as textual
description". The paper doesn't say how these are joined, and no prompt code is published.

The connector keeps `meta_info`, `domain` and `dataset` as published, as annotations of those names
on the input Record. Each task names them in its `input_annotations`, so the model gets them as
context. A comparison's lists hold one entry per series, so each of its input Records gets its own
entry. Every other value is the published text, a JSON list's text included. An annotation key has
one value type in TimeF, so lists aren't decoded. `task` and `question_type` are task metadata, not
context: the question already states both, and 10,000 rows give a wrong `question_type`.

### Tasks

Each row is one task. The question's format, read from the question's text, decides its TimeNet
type, so each format has its own loader method. `tasks()` keeps the file order, in which the formats
alternate. The row's paper task is its `task` metadata.

| Loader method | TimeNet type | Target | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| `true_or_false_tasks()` | `ClassificationTask` | `T` or `F` | 62,757 | 8,931 | 17,993 |
| `multiple_choice_tasks()` | `AnswerTask` | One letter, `A` to `D` | 63,311 | 9,090 | 18,029 |
| `puzzling_tasks()` | `AnswerTask` | Four letters, such as `C,B,D,A` | 20,932 | 2,979 | 5,978 |

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

The placeholder is the connector's choice. `request.authors_question(task)` prints each series back
the way the release does: a zero as `0`, any other value as Python's `repr` of the float. This gives
the published question exactly. Every question of the release reads back this way, and the build
fails if one doesn't.

**Task metadata**, never model input:
- `task` and `question_type`, as published;
- `question_format`, read from the question;
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

8,635 rows ask the same question about the same series as another row, with the same answer.
4,504 of them have a copy in another split, so a model can train on a test question:

| Task | Repeated | Across splits |
| --- | --- | --- |
| Classification | 7,227 | 3,440 |
| Anomaly detection | 965 | 632 |
| Temporal relationship | 355 | 408 |
| Data transformation | 84 | 22 |
| Characterization | 4 | 2 |

The "across splits" count includes rows whose copy expects another answer, which is why temporal
relationship has more such rows than repeated ones. Most classification copies come from UCR
datasets that share series: FreezerSmallTrain is drawn from FreezerRegularTrain, and
PhalangesOutlinesCorrect pools the Distal, Middle and Proximal phalanx sets. The copies keep their
own `dataset` and `meta_info`, so they differ only in context.

### Conflicting answers

550 rows ask the same question about the same series as another row, but expect another answer.
532 are puzzles whose patches repeat, as on a flat stretch. When patches are identical, more than
one order is right, but only the published one scores. 935 puzzles in all have two or more
identical patches. The other conflicts are 11 temporal relationship and 5 data transformation
true-or-false questions, and 2 classification questions.

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
none. The issues are the values of `issues.SourceIssue`:

| Issue | Rows | Meaning |
| --- | --- | --- |
| `question_type_mismatch` | 10,000 | `question_type` names another format than the question has |
| `repeated_question` | 8,635 | Another row asks the same question about the same series, with the same answer |
| `repeated_across_splits` | 4,504 | A copy is in another split |
| `conflicting_answer` | 550 | Another row asks the same question about the same series, with another answer |

18,930 tasks have at least one issue. The connector keeps them all.
`issues.without_source_issues` leaves them out, and the connector doesn't call it:

```python
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import (
    SourceIssue,
    without_source_issues,
)

# Leave out the test questions a model may have seen in training,
# and the ones with more than one right answer.
clean = list(
    without_source_issues(
        tasks,
        issues=[
            SourceIssue.REPEATED_ACROSS_SPLITS,
            SourceIssue.CONFLICTING_ANSWER,
        ],
    )
)
```

Report which issues were left out, and how many tasks remain, with any score.

## Data flow and validation

A build reads each file in stages, one module each:

1. `rows.py` validates every row as a strict `Row`:
   - `questions.py` reads its question against its task's template;
   - the answer must be one of the question's letters;
   - every series must be a nonempty list of finite numbers.

   A row keeps its series' lengths, not their values.
2. `index.py` checks each file's rows against the pinned counts in `release.py`, by task, published
   type and format. It then finds each row's source issues.
3. `records.py` builds the Records. Their values stay in the files. A Signal rereads its row when
   read, holding one file's `input_ts` and `question` columns at a time, and fails if the row no
   longer holds the series it was built from.
4. `tasks.py` and `request.py` build each task. The loader streams them.

The tests read a small fixture release, real rows of the pinned release with all their columns
(`tests/release_fixtures.py` names each one).

On the full release, run on 2026-10-29: every one of the 210,000 tasks read back from a full build
matches its row:
- `authors_question` gives the published question;
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
